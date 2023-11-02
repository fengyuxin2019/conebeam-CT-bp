#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "backwardKernel.h"

#define BLOCK_X 32
#define BLOCK_Y 32
#define BLOCK_A 1
#define PI 3.14159265359
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

texture<float, cudaTextureType3D, cudaReadModeElementType> sinoTexture;

__device__ float CalcuWeight(float3 pixel, float alpha, float gammamax1, float gammamax2, float Rf,float Rm)
{
	float w, w_ps, w_t;

	float x = pixel.x;
	float y = pixel.y;
    float z = pixel.z;
	float r = sqrt(x * x + y * y);
	float phi = atan2(-1 * x, y);
	float c1 = 1 / tan(gammamax1);
	float c2 = 1 / tan(gammamax2);
	float dr = Rm * Rm / (2 * Rf);  // alpha, Rm, Rf未定
	float theta = alpha - atan2(r * sin(alpha - phi), Rf + r * cos(alpha - phi));
	while (theta < phi - PI)
	{
		theta += 2 * PI;
	}
	while (theta > phi + PI)
	{
		theta -= 2 * PI;
	}
	float theta1 = phi - PI / 2;
	float theta2 = phi + PI / 2;
	float dtheta, angle1, angle2, angle3, angle4;

	//计算体素加权
	w_t = 0;
	w_ps = 0;

    if (Rf - z * c1 < 0)
	{
	    w_t = 1;
		if (r >= z * c1 - Rf)
		{
			dtheta = asin((Rf - z * c1) / r) - atan(sqrt(r * r - (Rf - z * c1)*(Rf - z * c1)) / (z * c1));
			while (dtheta < -PI)
			{
				dtheta += 2 * PI;
			}
			while (dtheta > PI)
			{
				dtheta -= 2 * PI;
			}
			if (dtheta < 0)
			{
				if (abs(theta - phi) <= PI / 2 + dtheta)
				{
					w_ps = 2;// * PI / (PI + 2 * dtheta);
				}
			}
			else
			{
				angle1 = theta1 - dtheta;
				angle2 = theta2 + dtheta;
				angle3 = theta1 + dtheta;
				angle4 = theta2 - dtheta;
				if (theta > angle1 && theta < angle3)
				{
					w_ps = 1 + sin(PI*(theta - angle1 - dtheta) / (2 * dtheta));
				}
				else
				{
					if (theta < angle2 && theta > angle4)
					{
						w_ps = 1 - sin(PI*(theta - angle2 + dtheta) / (2 * dtheta));
					}
					else
					{
						if (theta >= angle3 && theta <= angle4)
						{
							w_ps = 2;
						}
					}
				}
			}
		}
	}
	else
	{
		if (Rf - z * c2 > 0)
		{
			w_t = 1;
			if (r >= Rf - z * c2)
			{
				dtheta = asin((Rf - z * c2) / r) - atan(sqrt(r * r - (Rf - z * c2)*(Rf - z * c2)) / (z * c1));
				while (dtheta < -PI)
				{
					dtheta += 2 * PI;
				}
				while (dtheta > PI)
				{
					dtheta -= 2 * PI;
				}
				if (dtheta > 0)
				{
					if (abs(theta - phi) >= PI / 2 + dtheta)
					{
						w_ps = 2;
					}
				}
				else
				{
					angle1 = theta1 + dtheta;
					angle2 = theta2 - dtheta;
					angle3 = theta1 - dtheta;
					angle4 = theta2 + dtheta;
					if (theta > angle1 && theta < angle3)
					{
						w_ps = 1 - sin(PI*(theta - angle1 - dtheta) / (2 * dtheta));
					}
					else
					{
						if (theta < angle2 && theta > angle4)
						{
							w_ps = 1 + sin(PI*(theta - angle2 + dtheta) / (2 * dtheta));
						}
						else
						{
							if (theta <= angle1 || theta >= angle2)
							{
								w_ps = 2;
							}
						}
					}
				}
			}
		}
		else
		{
			float r1 = Rf - z * c1;
			float r2 = z * c2 - Rf;
			float z1 = (Rf - r) / c1;
			float z2 = (Rf + r) / c2;
			float z0 = 2 * Rf / (c1 + c2);
			if (r >= r1 && r <= r2)
			{
				w_t = 1;
				dtheta = asin((Rf - z * c1) / r) - atan(sqrt(r * r - (Rf - z * c1)*(Rf - z * c1)) / (z * c1));
				while (dtheta < -PI)
				{
					dtheta += 2 * PI;
				}
				while (dtheta > PI)
				{
					dtheta -= 2 * PI;
				}
				if (dtheta < 0)
				{
					if (abs(theta - phi) <= PI / 2 + dtheta)
					{
						w_ps = 2;
					}
				}
				else
				{
					angle1 = theta1 - dtheta;
					angle2 = theta2 + dtheta;
					angle3 = theta1 + dtheta;
					angle4 = theta2 - dtheta;
					if (theta > angle1 && theta < angle3)
					{
						w_ps = 1 + sin(PI*(theta - angle1 - dtheta) / (2 * dtheta));
					}
					else
					{
						if (theta < angle2 && theta > angle4)
						{
							w_ps = 1 - sin(PI*(theta - angle2 + dtheta) / (2 * dtheta));
						}
						else
						{
							if (theta >= angle3 && theta <= angle4)
							{
								w_ps = 2;
							}
						}
					}
				}
			}
			else
			{
				if (r <= r1 && r >= r2)
				{
					w_t = 1;
					dtheta = asin((Rf - z * c2) / r) - atan(sqrt(r * r - (Rf - z * c2)*(Rf - z * c2)) / (z * c2));
					while (dtheta < -PI)
					{
						dtheta += 2 * PI;
					}
					while (dtheta > PI)
					{
						dtheta -= 2 * PI;
					}
					if (dtheta > 0)
					{
						if (abs(theta - phi) >= PI / 2 + dtheta)
						{
							w_ps = 2;
						}
					}
					else
					{
						angle1 = theta1 + dtheta;
						angle2 = theta2 - dtheta;
						angle3 = theta1 - dtheta;
						angle4 = theta2 + dtheta;
						if (theta > angle1 && theta < angle3)
						{
							w_ps = 1 - sin(PI*(theta - angle1 - dtheta) / (2 * dtheta));
						}
						else
						{
							if (theta < angle2 && theta > angle4)
							{
								w_ps = 1 + sin(PI*(theta - angle2 + dtheta) / (2 * dtheta));
							}
							else
							{
								if (theta <= angle1 || theta >= angle2)
								{
									w_ps = 2;
								}
							}
						}
					}
				}
				else
				{
					if (r <= r1 && r <= r2)
					{
						if (r2 >= r1)
						{
							if (r >= r1 - 2 * dr)
							{
								w_t = 0.5 + 0.5 * sin(PI * (r - r1 + dr) / (2 * dr));
								w_ps = 1 + (z/(z1-z0)-z0/(z1-z0)) * cos(theta - phi);
							}
						}
						else
						{
							if (r >= r2 - 2 * dr)
							{
								w_t = 0.5 + 0.5 * sin(PI * (r - r2 + dr) / (2 * dr));
								w_ps = 1 - (z/(z2-z0)-z0/(z2-z0)) * cos(theta - phi);
							}
						}
					}
					else
					{
						w_t = 1;
						float dtheta1 = asin((Rf - z * c1) / r) - atan(sqrt(r * r - (Rf - z * c1)*(Rf - z * c1)) / (z * c1));
						float dtheta2 = asin((Rf - z * c2) / r) - atan(sqrt(r * r - (Rf - z * c2)*(Rf - z * c2)) / (z * c2));
						while (dtheta1 < -PI)
						{
							dtheta1 += 2 * PI;
						}
						while (dtheta1 > PI)
						{
							dtheta1 -= 2 * PI;
						}
						while (dtheta2 < -PI)
						{
							dtheta2 += 2 * PI;
						}
						while (dtheta2 > PI)
						{
							dtheta2 -= 2 * PI;
						}
						if (dtheta1 <= dtheta2)
						{
							w_ps = 0;
						}
						else
						{
							angle1 = theta1 - dtheta1;
							angle2 = theta2 + dtheta1;
							angle3 = theta1 - dtheta2;
							angle4 = theta2 + dtheta2;
							if ((theta > angle1 && theta < angle3) || (theta > angle4 && theta < angle2))
							{
								w_ps = PI / (dtheta1 - dtheta2);
								w_ps = w_ps > 2 ? 2 : w_ps;
							}
						}
					}
				}
			}
		}
	}

	w = 1 - w_t + w_t * w_ps;
    return w;
}

__global__ void backwardKernel(float* volume, const uint3 volumeSize, const uint2 detectorSize, const float* projectVector, const uint index,const int anglesNum,const float3 volumeCenter, const float2 detectorCenter,
                               const float volbiasz, const float dSampleInterval, const float dSliceInterval, const float sourceRadius, const float sourceZpos, const float fBiaz, const float  SID){
    uint3 volumeIdx = make_uint3(blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, blockIdx.z*blockDim.z+threadIdx.z);
    if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y){
        return;
    }

    float gammamax1 = atan(abs(fBiaz) / abs(SID + detectorCenter.x));
    float gammamax2 = atan(abs(fBiaz) / abs(SID + detectorCenter.x + detectorSize.x));

    for(int k=0;k<volumeSize.z;k++){
        float value = 0.0f;
        for(int angleIdx = index;angleIdx < index + BLOCK_A;angleIdx++){
            float3 sourcePosition = make_float3(projectVector[angleIdx*12], projectVector[angleIdx*12+1], projectVector[angleIdx*12+2]);
            float3 detectorPosition = make_float3(projectVector[angleIdx*12+3], projectVector[angleIdx*12+4], projectVector[angleIdx*12+5]);
            float3 u = make_float3(projectVector[angleIdx*12+6], projectVector[angleIdx*12+7], projectVector[angleIdx*12+8]);
            float3 v = make_float3(projectVector[angleIdx*12+9], projectVector[angleIdx*12+10], projectVector[angleIdx*12+11]);
            float3 coordinates = make_float3((volumeCenter.x + volumeIdx.x) * dSampleInterval, (volumeCenter.y + volumeIdx.y) * dSampleInterval,(volumeCenter.z + k) * dSliceInterval + volbiasz);
            float fScale = __fdividef(1.0f, det3(u, v, sourcePosition-coordinates));
            fScale = det3(u, v, sourcePosition-coordinates) == 0 ? 0 : fScale;
            float detectorX = fScale * det3(coordinates-sourcePosition,v,sourcePosition-detectorPosition)-detectorCenter.x;
            float detectorY = fScale * det3(u, coordinates-sourcePosition,sourcePosition-detectorPosition)-detectorCenter.y;
            float fr = fScale * det3(u, v, sourcePosition);

            float alpha = angleIdx * 2 * PI / anglesNum - PI / 2;
            float3 pixel = make_float3((volumeCenter.x + volumeIdx.x) * dSampleInterval, (volumeCenter.y + volumeIdx.y) * dSampleInterval, abs((volumeCenter.z + k) * dSliceInterval + volbiasz - sourceZpos));
            //float weight = CalcuWeight(pixel, alpha, gammamax1, gammamax2, sourceRadius, 0.5 * volumeSize.x * dSampleInterval);
            //value += tex3D(sinoTexture, detectorX, detectorY, angleIdx+0.5f);
			value += fr * fr * tex3D(sinoTexture, detectorX, detectorY, angleIdx + 0.5f);
        }
        int idx = k * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
		atomicAdd(&volume[idx], value / anglesNum );
		//volume[idx] += value * 2 * PI / anglesNum;
    }
}

torch::Tensor backward(torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
                        const float volbiasz, const float dSampleInterval, const float dSliceInterval,
                        const float sourceRadius, const float sourceZpos, const float fBiaz, const float  SID,
                        const long device){
    CHECK_INPUT(sino);
    CHECK_INPUT(_volumeSize);
    AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
    CHECK_INPUT(_detectorSize);
    AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
    CHECK_INPUT(projectVector);
    AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");

    int angles = projectVector.size(0);
    auto out = torch::zeros({sino.size(0), 1, _volumeSize[2].item<int>(), _volumeSize[1].item<int>(), _volumeSize[0].item<int>()}).to(sino.device());
    float* outPtr = out.data<float>();
    float* sinoPtr = sino.data<float>();

    cudaSetDevice(device);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    sinoTexture.addressMode[0] = cudaAddressModeBorder;
    sinoTexture.addressMode[1] = cudaAddressModeBorder;
    sinoTexture.addressMode[2] = cudaAddressModeBorder;
    sinoTexture.filterMode = cudaFilterModeLinear;
    sinoTexture.normalized = false;

    uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
    uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
    float3 volumeCenter = make_float3(volumeSize) / -2.0;
    float2 detectorCenter = make_float2(detectorSize) / -2.0;
    for(int batch = 0;batch < sino.size(0); batch++){
        float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
        float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;

        cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles);
        cudaArray *sinoArray;
        cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, detectorSize.x*sizeof(float), detectorSize.x, detectorSize.y);
        copyParams.dstArray = sinoArray;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.extent = m_extent;
        cudaMemcpy3D(&copyParams);
        cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

        const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
        const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, 1);
        for (int angle = 0; angle < angles; angle+=BLOCK_A){
           backwardKernel<<<gridSize, blockSize>>>(outPtrPitch, volumeSize, detectorSize, (float*)projectVector.data<float>(), angle,angles,volumeCenter,detectorCenter,
                                                   volbiasz, dSampleInterval, dSliceInterval, sourceRadius, sourceZpos, fBiaz, SID);
        }
      cudaUnbindTexture(sinoTexture);
      cudaFreeArray(sinoArray);
    }
    return out;
}

__global__ void backwardKernel_F(float* volume, const uint3 volumeSize, const uint2 detectorSize, const float* projectVector, const uint index, const int anglesNum, const float3 volumeCenter, const float2 detectorCenter,
	const float volbiasz, const float dSampleInterval, const float dSliceInterval, const float sourceRadius, const float sourceZpos, const float fBiaz, const float  SID, const uint systemNum) {
	uint3 volumeIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y || volumeIdx.z >= systemNum) {
		return;
	}

	float gammamax1 = atan(abs(fBiaz) / abs(SID + detectorCenter.x));
	float gammamax2 = atan(abs(fBiaz) / abs(SID + detectorCenter.x + detectorSize.x));
	float alpha = index * 2 * PI / anglesNum - PI / 2;

	unsigned projectVectorIdx = volumeIdx.z * 12;
	float3 sourcePosition = make_float3(projectVector[projectVectorIdx], projectVector[projectVectorIdx + 1], projectVector[projectVectorIdx + 2]);
	float3 detectorPosition = make_float3(projectVector[projectVectorIdx + 3], projectVector[projectVectorIdx + 4], projectVector[projectVectorIdx + 5]);
	float3 u = make_float3(projectVector[projectVectorIdx + 6], projectVector[projectVectorIdx + 7], projectVector[projectVectorIdx + 8]);
	float3 v = make_float3(projectVector[projectVectorIdx + 9], projectVector[projectVectorIdx + 10], projectVector[projectVectorIdx + 11]);

	for (int k = 0; k < volumeSize.z; k++) {
		float3 coordinates = make_float3((volumeCenter.x + volumeIdx.x) * dSampleInterval, (volumeCenter.y + volumeIdx.y) * dSampleInterval, (volumeCenter.z + k) * dSliceInterval + volbiasz);
		float fScale = __fdividef(1.0f, det3(u, v, sourcePosition - coordinates));
		fScale = det3(u, v, sourcePosition - coordinates) == 0 ? 0 : fScale;
		float detectorX = fScale * det3(coordinates - sourcePosition, v, sourcePosition - detectorPosition) - detectorCenter.x;
		float detectorY = fScale * det3(u, coordinates - sourcePosition, sourcePosition - detectorPosition) - detectorCenter.y;
		float fr = fScale * det3(u, v, sourcePosition);
		float3 pixel = make_float3((volumeCenter.x + volumeIdx.x) * dSampleInterval, (volumeCenter.y + volumeIdx.y) * dSampleInterval, abs((volumeCenter.z + k) * dSliceInterval + volbiasz - sourceZpos));
		//float weight = CalcuWeight(pixel, alpha, gammamax1, gammamax2, sourceRadius, 0.5 * volumeSize.x * dSampleInterval);
		float value = tex3D(sinoTexture, detectorX + 0.5f, detectorY + 0.5f, index + 0.5f);
		//float value =  tex3D(sinoTexture, detectorX + 0.5f, detectorY + 0.5f, index + 0.5f);
		int idx = k * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
		atomicAdd(&volume[idx], value / anglesNum / systemNum);
	}
}

void backward_F(torch::Tensor out, torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
	const float volbiasz, const float dSampleInterval, const float dSliceInterval,
	const float sourceRadius, const float sourceZpos, const float fBiaz, const float SID,
	const int systemNum, const long device) {
	CHECK_INPUT(sino);
	CHECK_INPUT(_volumeSize);
	AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
	CHECK_INPUT(_detectorSize);
	AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
	CHECK_INPUT(projectVector);
	AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");
	int angles = projectVector.size(0) / systemNum;
	//auto out = torch::zeros({sino.size(0), 1, _volumeSize[2].item<int>(), _volumeSize[1].item<int>(), _volumeSize[0].item<int>()}).to(sino.device());
	float* outPtr = out.data<float>();
	float* sinoPtr = sino.data<float>();

	cudaSetDevice(device);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	sinoTexture.addressMode[0] = cudaAddressModeBorder;
	sinoTexture.addressMode[1] = cudaAddressModeBorder;
	sinoTexture.addressMode[2] = cudaAddressModeBorder;
	sinoTexture.filterMode = cudaFilterModeLinear;
	sinoTexture.normalized = false;

	uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
	uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
	float3 volumeCenter = make_float3(volumeSize) / -2.0;
	float2 detectorCenter = make_float2(detectorSize) / -2.0;

	for (int batch = 0; batch < sino.size(0); batch++) {
		float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
		float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;

		cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles);
		cudaArray* sinoArray;
		cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, detectorSize.x * sizeof(float), detectorSize.x, detectorSize.y);
		copyParams.dstArray = sinoArray;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		copyParams.extent = m_extent;
		cudaMemcpy3D(&copyParams);
		cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

		const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, BLOCK_A);
		const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, systemNum / BLOCK_A );
		auto projVec = projectVector.reshape({ angles, systemNum * 12 });
		for (int angle = 0; angle < angles; angle++) {
			backwardKernel_F << <gridSize, blockSize >> > (outPtrPitch, volumeSize, detectorSize, (float*)projVec[angle].data<float>(), angle, angles, volumeCenter, detectorCenter,
				volbiasz, dSampleInterval, dSliceInterval, sourceRadius, sourceZpos, fBiaz, SID, systemNum);
			cudaDeviceSynchronize();
		}
		cudaUnbindTexture(sinoTexture);
		cudaFreeArray(sinoArray);
	}
	//return out;
}

__global__ void rotationKernal(float* out, const uint2 detectorSize, const uint2 targetdetectorSize, const float* projectVector,
	const float* targetprojectVector, const uint index, const int anglesNum, const float2 detectorCenter, const float2 targetdetectorCenter)
{ 
	uint3 targetdetectorIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (targetdetectorIdx.x >= targetdetectorSize.x || targetdetectorIdx.y >= targetdetectorSize.y ) {
		return;
	}
	float3 sourcePosition = make_float3(projectVector[0], projectVector[1], projectVector[2]);
	float3 detectorPosition = make_float3(projectVector[3], projectVector[4], projectVector[5]);
	float3 u = make_float3(projectVector[6], projectVector[7], projectVector[8]);
	float3 v = make_float3(projectVector[9], projectVector[10], projectVector[11]);
	float3 targetdetectorPosition = make_float3(targetprojectVector[3], targetprojectVector[4], targetprojectVector[5]);
	float3 targetu = make_float3(targetprojectVector[6], targetprojectVector[7], targetprojectVector[8]);
	float3 targetv = make_float3(targetprojectVector[9], targetprojectVector[10], targetprojectVector[11]);
	float targetdetectorX = targetdetectorIdx.x + targetdetectorCenter.x;
	float targetdetectorY = targetdetectorIdx.y + targetdetectorCenter.y;
	float3 coordinates = targetdetectorPosition+ targetdetectorX* targetu+ targetdetectorY* targetv;
	float fScale = 0;
	if ((det3(u, v, sourcePosition - coordinates)) != 0) { fScale = __fdividef(1.0f, det3(u, v, sourcePosition - coordinates)); }
	float detectorX = fScale * det3(coordinates - sourcePosition, v, sourcePosition - detectorPosition) - detectorCenter.x;
	float detectorY = fScale * det3(u, coordinates - sourcePosition, sourcePosition - detectorPosition) - detectorCenter.y;
	float value = tex3D(sinoTexture, detectorX + 0.5f, detectorY + 0.5f, index + 0.5f);
	unsigned sinogramIdx = index * targetdetectorSize.x * targetdetectorSize.y + targetdetectorIdx.y * targetdetectorSize.x + targetdetectorIdx.x;
	atomicAdd(&out[sinogramIdx], value);
}


void rotation(torch::Tensor out, torch::Tensor sino, torch::Tensor _detectorSize, torch::Tensor _detectorSize1, torch::Tensor projectVector,
	torch::Tensor projectVector1, const long device)
{
	CHECK_INPUT(sino);
	CHECK_INPUT(_detectorSize);
	AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 3");
	CHECK_INPUT(_detectorSize1);
	AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
	CHECK_INPUT(projectVector);
	AT_ASSERTM(projectVector.size(1) == 12, "project vector's shape must be [angle's number, 12]");
	int angles = projectVector.size(0);
	out.zero_();
	float* outPtr = out.data<float>();
	float* sinoPtr = sino.data<float>();

	cudaSetDevice(device);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	sinoTexture.addressMode[0] = cudaAddressModeBorder;
	sinoTexture.addressMode[1] = cudaAddressModeBorder;
	sinoTexture.addressMode[2] = cudaAddressModeBorder;
	sinoTexture.filterMode = cudaFilterModeLinear;
	sinoTexture.normalized = false;

	uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
	float2 detectorCenter = make_float2(detectorSize) / -2.0;
	uint2 detectorSize1 = make_uint2(_detectorSize1[0].item<int>(), _detectorSize1[1].item<int>());
	float2 detectorCenter1 = make_float2(detectorSize1) / -2.0;

	for (int batch = 0; batch < sino.size(0); batch++) {
		float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
		float* outPtrPitch = outPtr + detectorSize1.x * detectorSize1.y * angles * batch;

		cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles);
		cudaArray* sinoArray;
		cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, detectorSize.x * sizeof(float), detectorSize.x, detectorSize.y);
		copyParams.dstArray = sinoArray;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		copyParams.extent = m_extent;
		cudaMemcpy3D(&copyParams);
		cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

		const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, BLOCK_A);
		const dim3 gridSize = dim3(detectorSize1.x / BLOCK_X + 1, detectorSize1.y / BLOCK_Y + 1,  BLOCK_A);
		auto projVec = projectVector.reshape({ angles, 12 });
		auto targetprojVec = projectVector1.reshape({ angles, 12 });
		for (int angle = 0; angle < angles; angle++) {
			rotationKernal << <gridSize, blockSize >> > (outPtrPitch, detectorSize, detectorSize1, (float*)projVec[angle].data<float>(), (float*)targetprojVec[angle].data<float>(),
				angle, angles, detectorCenter, detectorCenter1);
			cudaDeviceSynchronize();
		}
		cudaUnbindTexture(sinoTexture);
		cudaFreeArray(sinoArray);
	}
}

__global__ void backwardKernel_P(float* volume, const uint3 volumeSize, const uint2 detectorSize, const float* projectMatrix, const uint index, const int anglesNum, const float3 volumeCenter, const float2 detectorCenter,
	const float volbiasz, const float dSampleInterval, const float dSliceInterval) {
	uint3 volumeIdx = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
	if (volumeIdx.x >= volumeSize.x || volumeIdx.y >= volumeSize.y) {
		return;
	}
	for (int k = 0; k < volumeSize.z; k++) {
		float value = 0.0f;
		for (int angleIdx = index; angleIdx < index + BLOCK_A; angleIdx++) {
			// coordinates * projectMatrix to get the detector piexl
			// 
			float4 Hrow1 = make_float4(projectMatrix[angleIdx * 12], projectMatrix[angleIdx * 12 + 3], projectMatrix[angleIdx * 12 + 6], projectMatrix[angleIdx * 12 + 9]);
			float4 Hrow2 = make_float4(projectMatrix[angleIdx * 12 + 1], projectMatrix[angleIdx * 12 + 4], projectMatrix[angleIdx * 12 + 7], projectMatrix[angleIdx * 12 + 10]);
			float4 Hrow3 = make_float4(projectMatrix[angleIdx * 12 + 2], projectMatrix[angleIdx * 12 + 5], projectMatrix[angleIdx * 12 + 8], projectMatrix[angleIdx * 12 + 11]);
			float4 coordinates = make_float4((volumeCenter.x + volumeIdx.x) * dSampleInterval, (volumeCenter.y + volumeIdx.y) * dSampleInterval, (volumeCenter.z + k) * dSliceInterval + volbiasz, 1.0);
			float3 pos2D = make_float3(dot(Hrow1, coordinates), dot(Hrow2, coordinates), dot(Hrow3, coordinates));
			float detectorX = 0, detectorY = 0;
			if (pos2D.z != 0) {
				detectorX = pos2D.x / pos2D.z - detectorCenter.x;
				detectorY = pos2D.y / pos2D.z - detectorCenter.y;
			}
			value += tex3D(sinoTexture, detectorX + 0.5f, detectorY + 0.5f, angleIdx + 0.5f);
		}
		int idx = k * volumeSize.x * volumeSize.y + volumeIdx.y * volumeSize.x + volumeIdx.x;
		atomicAdd(&volume[idx], value / anglesNum);
		//volume[idx] += value * 2 * PI / anglesNum;
	}
}

void backward_P(torch::Tensor out,torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectMatrix,
	const float volbiasz, const float dSampleInterval, const float dSliceInterval, const long device) {
	CHECK_INPUT(sino);
	CHECK_INPUT(_volumeSize);
	AT_ASSERTM(_volumeSize.size(0) == 3, "volume size's length must be 3");
	CHECK_INPUT(_detectorSize);
	AT_ASSERTM(_detectorSize.size(0) == 2, "detector size's length must be 2");
	CHECK_INPUT(projectMatrix);
	AT_ASSERTM(projectMatrix.size(1) == 12, "project vector's shape must be [angle's number, 12]");

	int angles = projectMatrix.size(0);
	out.zero_();
	float* outPtr = out.data<float>();
	float* sinoPtr = sino.data<float>();

	cudaSetDevice(device);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	sinoTexture.addressMode[0] = cudaAddressModeBorder;
	sinoTexture.addressMode[1] = cudaAddressModeBorder;
	sinoTexture.addressMode[2] = cudaAddressModeBorder;
	sinoTexture.filterMode = cudaFilterModeLinear;
	sinoTexture.normalized = false;

	uint3 volumeSize = make_uint3(_volumeSize[0].item<int>(), _volumeSize[1].item<int>(), _volumeSize[2].item<int>());
	uint2 detectorSize = make_uint2(_detectorSize[0].item<int>(), _detectorSize[1].item<int>());
	float3 volumeCenter = make_float3(volumeSize) / -2.0;
	float2 detectorCenter = make_float2(detectorSize) / -2.0;
	for (int batch = 0; batch < sino.size(0); batch++) {
		float* sinoPtrPitch = sinoPtr + detectorSize.x * detectorSize.y * angles * batch;
		float* outPtrPitch = outPtr + volumeSize.x * volumeSize.y * volumeSize.z * batch;

		cudaExtent m_extent = make_cudaExtent(detectorSize.x, detectorSize.y, angles);
		cudaArray* sinoArray;
		cudaMalloc3DArray(&sinoArray, &channelDesc, m_extent);
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr((void*)sinoPtrPitch, detectorSize.x * sizeof(float), detectorSize.x, detectorSize.y);
		copyParams.dstArray = sinoArray;
		copyParams.kind = cudaMemcpyDeviceToDevice;
		copyParams.extent = m_extent;
		cudaMemcpy3D(&copyParams);
		cudaBindTextureToArray(sinoTexture, sinoArray, channelDesc);

		const dim3 blockSize = dim3(BLOCK_X, BLOCK_Y, 1);
		const dim3 gridSize = dim3(volumeSize.x / BLOCK_X + 1, volumeSize.y / BLOCK_Y + 1, 1);
		for (int angle = 0; angle < angles; angle += BLOCK_A) {
			backwardKernel_P << <gridSize, blockSize >> > (outPtrPitch, volumeSize, detectorSize, (float*)projectMatrix.data<float>(), angle, angles, volumeCenter, detectorCenter,
				volbiasz, dSampleInterval, dSliceInterval);
		}
		cudaUnbindTexture(sinoTexture);
		cudaFreeArray(sinoArray);
	}
}
