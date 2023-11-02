#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_math.h"
#include "forwardKernel.h"

#define BLOCK_X 32
#define BLOCK_Y 32
#define BLOCK_A 1
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

texture<float, cudaTextureType3D, cudaReadModeElementType> volumeTexture;

__global__ void forwardKernel(float* sino, const uint3 volumeSize, const float3 volumeCenter, const uint2 detectorSize, const float2 detectorCenter, const float* projectVector, const uint index,
                              const float volbiasz, const float dSampleInterval, const float dSliceInterval){
    uint3 detectorIdx = make_uint3(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, blockIdx.z*blockDim.z+threadIdx.z);
    if (detectorIdx.x >= detectorSize.x || detectorIdx.y >= detectorSize.y){
        return;
    }

    float detectorX = detectorIdx.x + detectorCenter.x;
    float detectorY = detectorIdx.y + detectorCenter.y;

    float3 sourcePosition = make_float3(projectVector[index*12], projectVector[index*12+1], projectVector[index*12+2]);
    float3 detectorPosition = make_float3(projectVector[index*12+3], projectVector[index*12+4], projectVector[index*12+5]);
    float3 u = make_float3(projectVector[index*12+6], projectVector[index*12+7], projectVector[index*12+8]);
    float3 v = make_float3(projectVector[index*12+9], projectVector[index*12+10], projectVector[index*12+11]);

    float3 detectorPixel = detectorPosition + (0.5f+detectorX) *u + (0.5f+detectorY) * v ;
    float3 rayVector = normalize(detectorPixel - sourcePosition);

    float pixel = 0.0f;
    float alpha0, alpha1;
    float rayVectorDomainDim=fmax(fabs(rayVector.x),fmax(fabs(rayVector.z),fabs(rayVector.y)));
    if (fabs(rayVector.x) == rayVectorDomainDim){
        float volume_min_edge_point = volumeCenter.x * dSampleInterval;
        float volume_max_edge_point = (volumeSize.x + volumeCenter.x) * dSampleInterval;
        alpha0 = (volume_min_edge_point - sourcePosition.x) / rayVector.x;
        alpha1 = (volume_max_edge_point - sourcePosition.x) / rayVector.x;
    }
    else if(fabs(rayVector.y) == rayVectorDomainDim){
        float volume_min_edge_point = volumeCenter.y * dSampleInterval;
        float volume_max_edge_point = (volumeSize.y + volumeCenter.y) * dSampleInterval;
        alpha0 = (volume_min_edge_point - sourcePosition.y) / rayVector.y;
        alpha1 = (volume_max_edge_point - sourcePosition.y) / rayVector.y;
    }
    else {
        float volume_min_edge_point = volumeCenter.z * dSliceInterval + volbiasz;
        float volume_max_edge_point = (volumeSize.z + volumeCenter.z) * dSliceInterval + volbiasz;
        alpha0 = (volume_min_edge_point - sourcePosition.z) / rayVector.z;
        alpha1 = (volume_max_edge_point - sourcePosition.z) / rayVector.z;
    }
    float min_alpha = fmin(alpha0, alpha1) - 3;
    float max_alpha = fmax(alpha0, alpha1) + 3;
    float px, py, pz;
    float step_size = 1;

    while (min_alpha<max_alpha)
    {
        px = sourcePosition.x + min_alpha * rayVector.x;
        py = sourcePosition.y + min_alpha * rayVector.y;
        pz = sourcePosition.z + min_alpha * rayVector.z - volbiasz;
        px /= dSampleInterval;
        py /= dSampleInterval;
        pz /= dSliceInterval;
        px -= volumeCenter.x;
        py -= volumeCenter.y;
        pz -= volumeCenter.z;
        pixel += tex3D(volumeTexture, px + 0.5f, py + 0.5f, pz + 0.5f);
        min_alpha += step_size;
    }
    pixel *= step_size;
    unsigned sinogramIdx = index * detectorSize.x * detectorSize.y + detectorIdx.y * detectorSize.x + detectorIdx.x;
    atomicAdd(&sino[sinogramIdx], pixel);
    //sino[sinogramIdx] = pixel;
}

torch::Tensor forward(torch::Tensor volume, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
                      const float volbiasz, const float dSampleInterval, const float dSliceInterval, const long device){
    CHECK_INPUT(volume);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");

    int angles = projectVector.size(0);
    auto out = torch::zeros({volume.size(0), 1, angles, _detectorSize[1].item<int>(), _detectorSize[0].item<int>()}).to(volume.device());
    float* outPtr = out.data<float>();
    float* volumePtr = volume.data<float>();

    cudaSetDevice(device);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    volumeTexture.addressMode[0] = cudaAddressModeBorder;
    volumeTexture.addressMode[1] = cudaAddressModeBorder;
    volumeTexture.addressMode[2] = cudaAddressModeBorder;
    volumeTexture.filterMode = cudaFilterModeLinear;
    volumeTexture.normalized = false;

    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float2 detectorCenter = make_float2(detectorSize) / -2.0;

    for(int batch = 0;batch < volume.size(0); batch++){
        float* volumePtrPitch = volumePtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;
        float* outPtrPitch = outPtr + angles * detectorSize.x * detectorSize.y * batch;

        cudaExtent m_extent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
        cudaArray *volumeArray;
        cudaMalloc3DArray(&volumeArray, &channelDesc, m_extent);
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)volumePtrPitch, volumeSize.x*sizeof(float), volumeSize.x, volumeSize.y);
        copyParams.dstArray = volumeArray;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.extent = m_extent;
        cudaMemcpy3D(&copyParams);
        cudaBindTextureToArray(volumeTexture, volumeArray, channelDesc);

        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
        const dim3 gridSize = dim3(detectorSize.x / BLOCK_X + 1, detectorSize.y / BLOCK_Y + 1, 1);
        for (int angle = 0; angle < angles; angle++){
           forwardKernel<<<gridSize, blockSize>>>(outPtrPitch, volumeSize, volumeCenter, detectorSize, detectorCenter, (float*)projectVector.data<float>(), angle,
                                                  volbiasz, dSampleInterval, dSliceInterval);
           cudaDeviceSynchronize();
        }

      cudaUnbindTexture(volumeTexture);
      cudaFreeArray(volumeArray);
    }
    return out;
}

__global__ void forwardKernel_P(float* sino, const uint3 volumeSize, const float3 volumeCenter, const uint2 detectorSize, const float2 detectorCenter, const float* projectMatrix, const float* solutionSpace, const uint index,
    const float volbiasz, const float dSampleInterval, const float dSliceInterval) {
    uint3 detectorIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
    if (detectorIdx.x >= detectorSize.x || detectorIdx.y >= detectorSize.y) {
        return;
    }

    float detectorX = detectorIdx.x + detectorCenter.x;
    float detectorY = detectorIdx.y + detectorCenter.y;

    float4 S0 = make_float4(solutionSpace[index * 16], solutionSpace[index * 16 + 1], solutionSpace[index * 16 + 2], solutionSpace[index * 16 + 3]);
    float4 S1 = make_float4(solutionSpace[index * 16 + 4], solutionSpace[index * 16 + 5], solutionSpace[index * 16 + 6], solutionSpace[index * 16 + 7]);
    float4 S2 = make_float4(solutionSpace[index * 16 + 8], solutionSpace[index * 16 + 9], solutionSpace[index * 16 + 10], solutionSpace[index * 16 + 11]);
    float4 S3 = make_float4(solutionSpace[index * 16 + 12], solutionSpace[index * 16 + 13], solutionSpace[index * 16 + 14], solutionSpace[index * 16 + 15]);

    float4 specific_solution = S1 * detectorX + S2 * detectorY + S3;
    float min_z = volbiasz + volumeCenter.z * dSliceInterval;
    float max_z = volbiasz + (volumeCenter.z + volumeSize.z) * dSliceInterval;
    float lambda1 = (specific_solution.z - max_z * specific_solution.w) / (max_z * S0.w - S0.z);
    float lambda2 = (specific_solution.z - min_z * specific_solution.w) / (min_z * S0.w - S0.z);
    float4 pos3d1 = lambda1 * S0 + specific_solution;
    float4 pos3d2 = lambda2 * S0 + specific_solution;
    float3 pos3dreference = make_float3(pos3d1.x / pos3d1.w, pos3d1.y / pos3d1.w, pos3d1.z / pos3d1.w);
    float3 rayVector = normalize(make_float3(pos3d1.x / pos3d1.w, pos3d1.y / pos3d1.w
        , pos3d1.z / pos3d1.w) - make_float3(pos3d2.x / pos3d2.w, pos3d2.y / pos3d2.w
            , pos3d2.z / pos3d2.w));
    //
    float pixel = 0.0f;
    float alpha0, alpha1;
    float rayVectorDomainDim = fmax(fabs(rayVector.x), fmax(fabs(rayVector.z), fabs(rayVector.y)));
    if (fabs(rayVector.x) == rayVectorDomainDim) {
        float volume_min_edge_point = volumeCenter.x * dSampleInterval;
        float volume_max_edge_point = (volumeSize.x + volumeCenter.x) * dSampleInterval;
        alpha0 = (volume_min_edge_point - pos3dreference.x) / rayVector.x;
        alpha1 = (volume_max_edge_point - pos3dreference.x) / rayVector.x;
    }
    else if (fabs(rayVector.y) == rayVectorDomainDim) {
        float volume_min_edge_point = volumeCenter.y * dSampleInterval;
        float volume_max_edge_point = (volumeSize.y + volumeCenter.y) * dSampleInterval;
        alpha0 = (volume_min_edge_point - pos3dreference.y) / rayVector.y;
        alpha1 = (volume_max_edge_point - pos3dreference.y) / rayVector.y;
    }
    else {
        float volume_min_edge_point = volumeCenter.z * dSliceInterval + volbiasz;
        float volume_max_edge_point = (volumeSize.z + volumeCenter.z) * dSliceInterval + volbiasz;
        alpha0 = (volume_min_edge_point - pos3dreference.z) / rayVector.z;
        alpha1 = (volume_max_edge_point - pos3dreference.z) / rayVector.z;
    }
    float min_alpha = fmin(alpha0, alpha1) - 3;
    float max_alpha = fmax(alpha0, alpha1) + 3;
    float px, py, pz;
    float step_size = 1;

    while (min_alpha < max_alpha)
    {
        px = pos3dreference.x + min_alpha * rayVector.x;
        py = pos3dreference.y + min_alpha * rayVector.y;
        pz = pos3dreference.z + min_alpha * rayVector.z - volbiasz;
        px /= dSampleInterval;
        py /= dSampleInterval;
        pz /= dSliceInterval;
        px -= volumeCenter.x;
        py -= volumeCenter.y;
        pz -= volumeCenter.z;
        pixel += tex3D(volumeTexture, px, py, pz);
        min_alpha += step_size;
    }
    pixel *= step_size / fabs(alpha0 - alpha1);
    unsigned sinogramIdx = index * detectorSize.x * detectorSize.y + detectorIdx.y * detectorSize.x + detectorIdx.x;
    sino[sinogramIdx] = pixel;
}

void forward_P(torch::Tensor out,torch::Tensor volume, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectMatrix,
    torch::Tensor solutionSpace, const float volbiasz, const float dSampleInterval, const float dSliceInterval, const long device) {
    CHECK_INPUT(volume);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectMatrix);
    AT_ASSERTM(projectMatrix.size(1) == 12, "project vector's shape must be [angle's number, 12]");
    CHECK_INPUT(solutionSpace);
    AT_ASSERTM(solutionSpace.size(1) == 16, "project vector's shape must be [angle's number, 12]");

    int angles = projectMatrix.size(0);
    float* outPtr = out.data<float>();
    float* volumePtr = volume.data<float>();

    cudaSetDevice(device);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    volumeTexture.addressMode[0] = cudaAddressModeBorder;
    volumeTexture.addressMode[1] = cudaAddressModeBorder;
    volumeTexture.addressMode[2] = cudaAddressModeBorder;
    volumeTexture.filterMode = cudaFilterModeLinear;
    volumeTexture.normalized = false;

    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float2 detectorCenter = make_float2(detectorSize) / -2.0;

    for (int batch = 0; batch < volume.size(0); batch++) {
        float* volumePtrPitch = volumePtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;
        float* outPtrPitch = outPtr + angles * detectorSize.x * detectorSize.y * batch;

        cudaExtent m_extent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
        cudaArray* volumeArray;
        cudaMalloc3DArray(&volumeArray, &channelDesc, m_extent);
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)volumePtrPitch, volumeSize.x * sizeof(float), volumeSize.x, volumeSize.y);
        copyParams.dstArray = volumeArray;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.extent = m_extent;
        cudaMemcpy3D(&copyParams);
        cudaBindTextureToArray(volumeTexture, volumeArray, channelDesc);

        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
        const dim3 gridSize = dim3(detectorSize.x / BLOCK_X + 1, detectorSize.y / BLOCK_Y + 1, 1);
        for (int angle = 0; angle < angles; angle++) {
            forwardKernel_P << <gridSize, blockSize >> > (outPtrPitch, volumeSize, volumeCenter, detectorSize, detectorCenter, (float*)projectMatrix.data<float>(),
                (float*)solutionSpace.data<float>(), angle, volbiasz, dSampleInterval, dSliceInterval);
        }

        cudaUnbindTexture(volumeTexture);
        cudaFreeArray(volumeArray);
    }

}

__global__ void forwardKernel_F(float* sino, const uint3 volumeSize, const float3 volumeCenter, const uint2 detectorSize, const float2 detectorCenter, const float* projectVector, const uint index,
    const float volbiasz, const float dSampleInterval, const float dSliceInterval, const uint systemNum) {
    uint3 detectorIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
    if (detectorIdx.x >= detectorSize.x || detectorIdx.y >= detectorSize.y || detectorIdx.z >= systemNum) {
        return;
    }

    float detectorX = detectorIdx.x + detectorCenter.x;
    float detectorY = detectorIdx.y + detectorCenter.y;

    unsigned projectVectorIdx = detectorIdx.z * 12;
    float3 sourcePosition = make_float3(projectVector[projectVectorIdx], projectVector[projectVectorIdx + 1], projectVector[projectVectorIdx + 2]);
    float3 detectorPosition = make_float3(projectVector[projectVectorIdx + 3], projectVector[projectVectorIdx + 4], projectVector[projectVectorIdx + 5]);
    float3 u = make_float3(projectVector[projectVectorIdx + 6], projectVector[projectVectorIdx + 7], projectVector[projectVectorIdx + 8]);
    float3 v = make_float3(projectVector[projectVectorIdx + 9], projectVector[projectVectorIdx + 10], projectVector[projectVectorIdx + 11]);

    float3 detectorPixel = detectorPosition + (0.5f + detectorX) * u + (0.5f + detectorY) * v;
    float3 sourcePoint = sourcePosition;
    float3 rayVector = normalize(detectorPixel - sourcePoint);

    float pixel = 0.0f;
    float alpha0, alpha1;
    float rayVectorDomainDim = fmax(fabs(rayVector.x), fmax(fabs(rayVector.z), fabs(rayVector.y)));
    if (fabs(rayVector.x) == rayVectorDomainDim) {
        float volume_min_edge_point = volumeCenter.x * dSampleInterval;
        float volume_max_edge_point = (volumeSize.x + volumeCenter.x) * dSampleInterval;
        alpha0 = (volume_min_edge_point - sourcePoint.x) / rayVector.x;
        alpha1 = (volume_max_edge_point - sourcePoint.x) / rayVector.x;
    }
    else if (fabs(rayVector.y) == rayVectorDomainDim) {
        float volume_min_edge_point = volumeCenter.y * dSampleInterval;
        float volume_max_edge_point = (volumeSize.y + volumeCenter.y) * dSampleInterval;
        alpha0 = (volume_min_edge_point - sourcePoint.y) / rayVector.y;
        alpha1 = (volume_max_edge_point - sourcePoint.y) / rayVector.y;
    }
    else {
        float volume_min_edge_point = volumeCenter.z * dSliceInterval + volbiasz;
        float volume_max_edge_point = (volumeSize.z + volumeCenter.z) * dSliceInterval + volbiasz;
        alpha0 = (volume_min_edge_point - sourcePoint.z) / rayVector.z;
        alpha1 = (volume_max_edge_point - sourcePoint.z) / rayVector.z;
    }
    float min_alpha = fmin(alpha0, alpha1) - 3;
    float max_alpha = fmax(alpha0, alpha1) + 3;
    float step_size = 1;
    float px, py, pz;
    float ll = max_alpha - min_alpha;

    while (min_alpha < max_alpha)
    {
        px = sourcePoint.x + min_alpha * rayVector.x;
        py = sourcePoint.y + min_alpha * rayVector.y;
        pz = sourcePoint.z + min_alpha * rayVector.z - volbiasz;
        px /= dSampleInterval;
        py /= dSampleInterval;
        pz /= dSliceInterval;
        px -= volumeCenter.x;
        py -= volumeCenter.y;
        pz -= volumeCenter.z;
        pixel += tex3D(volumeTexture, px + 0.5f, py + 0.5f, pz + 0.5f);
        min_alpha += step_size;
    }
    //pixel *= (step_size / systemNum );
    pixel *= (step_size / systemNum/fabs(alpha0-alpha1));
    unsigned sinogramIdx = index * detectorSize.x * detectorSize.y + detectorIdx.y * detectorSize.x + detectorIdx.x;
    atomicAdd(&sino[sinogramIdx], pixel);
}

void forward_F(torch::Tensor out, torch::Tensor volume, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
    const float volbiasz, const float dSampleInterval, const float dSliceInterval, const int systemNum, const long device) {
    CHECK_INPUT(volume);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");

    int angles = projectVector.size(0) / systemNum;
    out.zero_();
    //auto out = torch::zeros({volume.size(0), 1, angles, _detectorSize[1].item<int>(), _detectorSize[0].item<int>()}).to(volume.device());
    float* outPtr = out.data<float>();
    float* volumePtr = volume.data<float>();

    cudaSetDevice(device);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    volumeTexture.addressMode[0] = cudaAddressModeBorder;
    volumeTexture.addressMode[1] = cudaAddressModeBorder;
    volumeTexture.addressMode[2] = cudaAddressModeBorder;
    volumeTexture.filterMode = cudaFilterModeLinear;
    volumeTexture.normalized = false;

    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float2 detectorCenter = make_float2(detectorSize) / -2.0;

    for (int batch = 0; batch < volume.size(0); batch++) {
        float* volumePtrPitch = volumePtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;
        float* outPtrPitch = outPtr + angles * detectorSize.x * detectorSize.y * batch;

        cudaExtent m_extent = make_cudaExtent(volumeSize.x, volumeSize.y, volumeSize.z);
        cudaArray* volumeArray;
        cudaMalloc3DArray(&volumeArray, &channelDesc, m_extent);
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr((void*)volumePtrPitch, volumeSize.x * sizeof(float), volumeSize.x, volumeSize.y);
        copyParams.dstArray = volumeArray;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.extent = m_extent;
        cudaMemcpy3D(&copyParams);
        cudaBindTextureToArray(volumeTexture, volumeArray, channelDesc);

        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, BLOCK_A);
        const dim3 gridSize = dim3(detectorSize.x / BLOCK_X + 1, detectorSize.y / BLOCK_Y + 1, systemNum / BLOCK_A );
        auto projVec = projectVector.reshape({ angles, systemNum * 12 });
        for (int angle = 0; angle < angles; angle++) {
            forwardKernel_F << <gridSize, blockSize >> > (outPtrPitch, volumeSize, volumeCenter, detectorSize, detectorCenter, (float*)projVec[angle].data<float>(), angle,
                volbiasz, dSampleInterval, dSliceInterval, systemNum);
            cudaDeviceSynchronize();
        }

        cudaUnbindTexture(volumeTexture);
        cudaFreeArray(volumeArray);
    }
    //return out;
}

