#include <torch/torch.h>

torch::Tensor gradient(torch::Tensor vol, torch::Tensor _volumeSize, int gradientType);