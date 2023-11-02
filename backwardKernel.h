#include <torch/torch.h>

torch::Tensor backward(torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
                       const float volbiasz, const float dSampleInterval, const float dSliceInterval,
                       const float sourceRadius, const float sourceZpos, const float fBiaz, const float  SID,
                       const long device);

void backward_F(torch::Tensor out, torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectVector,
	const float volbiasz, const float dSampleInterval, const float dSliceInterval,
	const float sourceRadius, const float sourceZpos, const float fBiaz, const float SID,
	const int systemNum, const long device);

void backward_P(torch::Tensor out, torch::Tensor sino, torch::Tensor _volumeSize, torch::Tensor _detectorSize, torch::Tensor projectMatrix,
	const float volbiasz, const float dSampleInterval, const float dSliceInterval, const long device);

void rotation(torch::Tensor out, torch::Tensor sino, torch::Tensor _detectorSize, torch::Tensor _detectorSize1, torch::Tensor projectVector,
    torch::Tensor projectVector1, const long device);