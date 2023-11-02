#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_math.h"
#include "cosweightKernel.h"
#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_A 16
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// 存储体块的纹理内存


__global__ void cosweightKernel(float *pSinogramW, float *pSinogram, const float* project_vectors,
                                const uint2 detector_size, const float2 detector_origin ,
                                const uint projidx)
{
    //像素驱动，此核代表一个探测器像素
    uint2 detector_idx = make_uint2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    uint projection_number = projidx;
    if (detector_idx.x >= detector_size.x || detector_idx.y >= detector_size.y)
    {
        return;
    }

    float detectorX = detector_idx.x + detector_origin.x;
    float detectorY = detector_idx.y + detector_origin.y;

    float3 source_pos = make_float3(project_vectors[projection_number*12], project_vectors[projection_number*12+1], project_vectors[projection_number*12+2]);
    float3 detector_pos = make_float3(project_vectors[projection_number*12+3], project_vectors[projection_number*12+4], project_vectors[projection_number*12+5]);
    float3 u = make_float3(project_vectors[projection_number*12+6], project_vectors[projection_number*12+7], project_vectors[projection_number*12+8]);
    float3 v = make_float3(project_vectors[projection_number*12+9], project_vectors[projection_number*12+10], project_vectors[projection_number*12+11]);

    float fpX = dot(source_pos-detector_pos,u);
    float fpY = dot(source_pos-detector_pos,v);
    float dline = length(detector_pos+fpX*u+fpY*v-source_pos);
    float OX = dot(source_pos,u);
    float OY = dot(source_pos,v);
    float sod = length(OX*u+OY*v-source_pos);
    float dsqrtSP = sqrt(dline*dline+(detectorX-fpX)*(detectorX-fpX)+(detectorY-fpY)*(detectorY-fpY));
    float w = dline / dsqrtSP;

    unsigned sinogram_idx = projidx * detector_size.x * detector_size.y + detector_idx.y * detector_size.x + detector_idx.x;
    float val = pSinogram[sinogram_idx];
    pSinogramW[sinogram_idx] = w * val;
    return;
}

torch::Tensor cosweight(torch::Tensor sino, torch::Tensor _detectorSize, torch::Tensor projectVector,const long device){
    CHECK_INPUT(sino);
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");

    int angles = projectVector.size(0);
    auto out = torch::zeros({sino.size(0), 1, angles, _detectorSize[1].item<int>(), _detectorSize[0].item<int>()}).to(sino.device());
    float* outPtr = out.data<float>();
    float* sinoPtr = sino.data<float>();

    // 体块和探测器的大小位置向量化
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float2 detectorCenter = make_float2(detectorSize) / -2.0;
    for(int batch = 0;batch < sino.size(0); batch++){
        float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
        float* outPtrPitch = outPtr + angles * detectorSize.x * detectorSize.y * batch;

        // 以角度为单位做体素驱动的反投影
        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
        const dim3 gridSize = dim3(detectorSize.x / BLOCK_X + 1, detectorSize.y / BLOCK_Y + 1, 1);
        for (int angle = 0; angle < angles; angle++){
           cosweightKernel<<<gridSize, blockSize>>>(outPtrPitch, sinoPtrPitch, (float*)projectVector.data<float>(), detectorSize, detectorCenter, angle);
        }
    }
    return out;
}
