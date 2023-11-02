#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "dtvtools.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_Z 4
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ float gradientForwardX(float* image, unsigned int offset, const uint3 volumeSize) {
    unsigned int width = volumeSize.x;
    if (offset % width + 1 >= width)
        return -image[offset];
    else
        return image[offset + 1] - image[offset];
}

__device__ float gradientForwardY(float* image, unsigned int offset, const uint3 volumeSize) {
    unsigned int width = volumeSize.x;
    unsigned int height = volumeSize.y;
    if (offset % (width * height) + width >= width * height)
        return -image[offset];
    else
        return image[offset + width] - image[offset];
}

__device__ float gradientForwardZ(float* image, unsigned int offset, const uint3 volumeSize) {
    unsigned int width = volumeSize.x;
    unsigned int height = volumeSize.y;
    unsigned int slice = volumeSize.z;
    if (offset % (width * height * slice) + width * height >= width * height * slice)
        return -image[offset];
    else
        return image[offset + width * height] - image[offset];
}

__device__ float gradientBackwardX(float* image, unsigned int offset, const uint3 volumeSize) {
    unsigned int width = volumeSize.x;
    if (offset % width < 1)
        return -image[offset];
    else
        return image[offset - 1] - image[offset];
}

__device__ float gradientBackwardY(float* image, unsigned int offset, const uint3 volumeSize) {
    unsigned int width = volumeSize.x;
    unsigned int height = volumeSize.y;
    if (offset % (width * height) < width)
        return -image[offset];
    else
        return image[offset - width] - image[offset];
}

__device__ float gradientBackwardZ(float* image, unsigned int offset, const uint3 volumeSize) {
    unsigned int width = volumeSize.x;
    unsigned int height = volumeSize.y;
    unsigned int slice = volumeSize.z;
    if (offset % (width * height * slice) < width * height)
        return -image[offset];
    else
        return image[offset - width * height] - image[offset];
}

__global__ void GradKernel(float* f_, float* p, const uint3 volumeSize, int gradientType) {
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (x >= volumeSize.x || y >= volumeSize.y || z >= volumeSize.z) {
        return;
    }

    unsigned int offset = x + y * volumeSize.x + z * volumeSize.x * volumeSize.y;
    switch (gradientType) {
        case 1:
            p[offset] = gradientForwardX(f_, offset, volumeSize);
            break;
        case 2:
            p[offset] = gradientForwardY(f_, offset, volumeSize);
            break;
        case 3:
            p[offset] = gradientForwardZ(f_, offset, volumeSize);
            break;
        case -1:
            p[offset] = gradientBackwardX(f_, offset, volumeSize);
            break;
        case -2:
            p[offset] = gradientBackwardY(f_, offset, volumeSize);
            break;
        case -3:
            p[offset] = gradientBackwardZ(f_, offset, volumeSize);
            break;
        default:
            break;
    }
}

torch::Tensor gradient(torch::Tensor vol, torch::Tensor _volumeSize, int gradientType) {
    CHECK_INPUT(vol);
    //CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");

    auto out = torch::zeros({ vol.size(0), 1, _volumeSize[2].item<int>(), _volumeSize[1].item<int>(), _volumeSize[0].item<int>() }).to(vol.device());
    float* outPtr = out.data_ptr<float>();
    float* volPtr = vol.data_ptr<float>();
    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());

    for (int batch = 0; batch < vol.size(0); batch++) {
        float* volPtrPitch = volPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;
        float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;
        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, BLOCK_Z);
        const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, volumeSize.z / BLOCK_Z + 1);      
        GradKernel<<<gridSize, blockSize>>>(volPtrPitch, outPtrPitch, volumeSize, gradientType);   
        
    }
    return out;
}
