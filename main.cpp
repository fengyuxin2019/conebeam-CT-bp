#include <iostream>
#include <fstream>
#include <torch/torch.h>
#include "forwardKernel.h"
#include "backwardKernel.h"
#include "cosweightKernel.h"
#include "dtvtools.h"
#include <ATen/ATen.h>
#include "infer.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cmath>

#include <tuple>
#include <vector>
#include <torch/script.h>
#include <chrono>
#include <Windows.h>
#include <filesystem>
#include <format>
#define PI 3.14159265359
#define imageshape { 1, 1, 100,1600,1600 }
#define sinoshape { 1, 1, 32, 2940, 2304 }
#define sinoshape1 { 1, 1, 32, 5605, 2303 }
//#define sinoshape { 1, 1, 160, 512, 512 }
//#define sinoshape1 { 1, 1, 160, 512, 512 }

std::chrono::high_resolution_clock::time_point print_time_elapsed_and_return_current_time(
    std::chrono::high_resolution_clock::time_point t1, const std::string& event_name) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_span = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - t1);
    std::cout << event_name << " - Time taken: " << time_span.count() << " milliseconds" << std::endl;
    return current_time;
}

torch::Tensor load_raw_file_to_tensor(std::string filename, int batch_size, int volume_depth, int volume_height, int volume_width, int deviceId) {
    // open file
    std::ifstream raw_file(filename, std::ios::binary);

    // get file size
    raw_file.seekg(0, std::ios::end);
    std::streampos file_size = raw_file.tellg();
    raw_file.seekg(0, std::ios::beg);

    // read file data into vector
    std::vector<char> data(file_size);
    raw_file.read(data.data(), file_size);

    // create tensor from vector data
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(data.data(), { batch_size, volume_depth, volume_height, volume_width }, options);
    /*std::cout << input[0][0][19][10];*/
    // move tensor to device
    input = input.to(torch::kCUDA, deviceId);
    return input;
}

torch::Tensor load_raw_file_to_cpu(std::string filename, int batch_size, int volume_depth, int volume_height, int volume_width, int deviceId) {
    // open file
    std::ifstream raw_file(filename, std::ios::binary);

    // get file size
    raw_file.seekg(0, std::ios::end);
    std::streampos file_size = raw_file.tellg();
    raw_file.seekg(0, std::ios::beg);

    // read file data into vector
    std::vector<char> data(file_size);
    raw_file.read(data.data(), file_size);

    // create tensor from vector data
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(data.data(), { batch_size, volume_depth, volume_height, volume_width }, options);
    /*std::cout << input[0][0][19][10];*/
    // move tensor to device
    return input;
}

void write_tensor_to_binary_file(torch::Tensor& tensor, const std::string& filename) {
    // get tensor data pointer
    float* data_ptr = tensor.data<float>();
    // write tensor data to binary file
    std::ofstream outfile(filename, std::ios::binary);
    outfile.write((char*)data_ptr, tensor.numel() * sizeof(float));
    outfile.close();
    //data_ptr = NULL;
}


static void print_cuda_use()
{
    size_t free_byte;
    size_t total_byte;

    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db_1 = (total_db - free_db) / 1024.0 / 1024.0;
    std::cout << "Now used GPU memory " << used_db_1 << "  MB\n";
}

void infer(std::string enginePath, float* dummyInput, float* output)
{
    Infer infer(enginePath, 1 * 1 * 100 * 1600 * 1600);
    //float* dummyInput = new float[2016 * 2016 * 12];
    //for (int i = 0; i < 2016 * 2016 * 12; i++)
    //{
    //    *(dummyInput + i) = 0.0f;
    //}
    print_cuda_use();
    infer.doInference(dummyInput, output, 0);
    /*if (fResult != nullptr)
        std::cout << "Inference success!" << std::endl;*/
        //return fResult;

}

torch::Tensor ramp_filter(int projWidth) {
    torch::Tensor filter = torch::ones({ 1, 1, 1, projWidth });
    int mid = std::floor(projWidth / 2);
    for (int i = 0; i < projWidth; ++i) {

        if ((i - mid) % 2 == 0) {
            filter[0][0][0][i] = 0;
        }
        else {
            filter[0][0][0][i] = -0.5 / (M_PI * M_PI * (i - mid) * (i - mid));
        }
        if (i == mid) {
            filter[0][0][0][i] = 1.0 / 8.0;
        }
    }
    filter = filter;
    return filter;
}
torch::Tensor shepp_logan_filter(int projWidth) {
    torch::Tensor filter = ramp_filter(projWidth);
    int mid = std::floor(projWidth / 2);

    for (int i = 0; i < projWidth; ++i) {
        if (i != mid) {
            filter[0][0][0][i] *= std::sin(M_PI * (i - mid) / (2 * projWidth)) / (M_PI * (i - mid) / (2 * projWidth));
        }
    }
    return filter;
}

torch::Tensor cosine_filter(int projWidth) {
    torch::Tensor filter = ramp_filter(projWidth);
    int mid = std::floor(projWidth / 2);

    for (int i = 0; i < projWidth; ++i) {
        filter[0][0][0][i] *= std::cos(M_PI * (i - mid) / (2 * projWidth));
    }
    return filter;
}

torch::Tensor hamming_filter(int projWidth) {
    torch::Tensor filter = ramp_filter(projWidth);
    int mid = std::floor(projWidth / 2);

    for (int i = 0; i < projWidth; ++i) {
        filter[0][0][0][i] *= 0.54 + 0.46 * std::cos(M_PI * (i - mid) / projWidth);
    }
    return filter;
}

torch::Tensor sobel_filter() {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor sobel_x = torch::tensor({ {{{-1, 0, 1},
                                             {-2, 0, 2},
                                             {-1, 0, 1}}} }, options);
    sobel_x.view({ 1,1,3,3 });
    
    return sobel_x;
}

torch::Tensor gaussian_filter(int kernel_size, float std_dev) {
    torch::Tensor kernel = torch::zeros({ kernel_size, kernel_size });
    int mid = std::floor(kernel_size / 2);
    float sum = 0;
    for (int x = -mid; x <= mid; x++) {
        for (int y = -mid; y <= mid; y++) {
            float value = std::exp(-(x * x + y * y) / (2 * std_dev * std_dev)) / (2 * M_PI * std_dev * std_dev);
            kernel[x + mid][y + mid] = value;
            sum += value;
        }
    }
    kernel.view({1,1,kernel_size, kernel_size });
    return (kernel / sum);
}

torch::Tensor computeSSIM3D(const torch::Tensor& vol1, const torch::Tensor& vol2, double C1 = 6.5025, double C2 = 58.5225) {
    // 注意: vol1 和 vol2 的维度应该是 {1, 1, 1600, 1600, 100}
    auto mu1 = torch::avg_pool3d(vol1, { 3, 3, 3 }, 1, 1).squeeze();
    auto mu2 = torch::avg_pool3d(vol2, { 3, 3, 3 }, 1, 1).squeeze();

    auto sigma1_sq = torch::avg_pool3d(vol1 * vol1, { 3, 3, 3 }, 1, 1) - mu1 * mu1;
    auto sigma2_sq = torch::avg_pool3d(vol2 * vol2, { 3, 3, 3 }, 1, 1) - mu2 * mu2;
    auto sigma12 = torch::avg_pool3d(vol1 * vol2, { 3, 3, 3 }, 1, 1) - mu1 * mu2;

    auto ssim_map = ((2.0 * mu1 * mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2));
    std::cout << "ssim:" << ssim_map.mean().item<double>() << std::endl;
    return ssim_map.mean();
}

void gradientTest() {
    int size = 10;
    int depth = 3;
    torch::Tensor tensor = torch::zeros({ depth, size, size });
    for (int i = 0; i < depth; i++) {
        tensor[i] = torch::ones({ size, size }) * (i + 1);
    }
    tensor = tensor.view({ 1, 1, depth, size, size }).to(torch::kCUDA);
    torch::Tensor _volumeSize = torch::tensor({ size, size, depth }, torch::kInt).to(torch::kCUDA);
    torch::Tensor result = gradient(tensor, _volumeSize, 3);
    std::cout << result << std::endl;
}

torch::Tensor BackGradient(torch::Tensor img, int gradienttype) 
{ 
    auto first_column = img.slice(gradienttype, 0, 1);
    auto second_slice = img.slice(gradienttype, 1);
    return  torch::cat({ second_slice, first_column }, gradienttype);
}

torch::Tensor l1ball(torch::Tensor v, double b) {
    if (v.abs().sum().item<float>() < b) {
        return v;
    }
    double paramLambda = 0.0;
    auto objectValue = torch::relu(v.abs() - paramLambda).sum() - b;
    int iterations = 0;
    double s = std::abs(objectValue.item<float>());
    std::cout << "l1ball target:" << b << std::endl;
    std::cout << "l1ball input:" << v.abs().sum().item<float>() << std::endl;
    while (std::abs(objectValue.item<float>()) > 1e-4 && iterations < 100) {
        objectValue = torch::relu(v.abs() - paramLambda).sum() - b;
        auto difference = (v.abs() - paramLambda > 0).to(torch::kFloat).sum() + 0.001;
        paramLambda += (objectValue / difference).item<float>();
        iterations += 1;
        s = std::abs(difference.item<float>());
    }
    std::cout << "l1ball iteration:" << iterations << std::endl;
    paramLambda = std::max(paramLambda, 0.0);
    auto w = v.sign() * torch::relu(v.abs() - paramLambda);
    std::cout << "l1ball output:" << w.abs().sum().item<float>() << std::endl;
    return w;
}

double power_method_H(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectVector, float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum, int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID)
{
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor x;
    torch::Tensor sino = torch::zeros(sinoshape).to(torch::kCUDA, device);
    torch::Tensor weightMap = torch::ones(imageshape).to(torch::kCUDA, device);
    /*sino = forward(weightMap, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, device);
    weightMap = backward(sino, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, sourceRadius, sourceZpos, fBiaz, SID, device);*/
    forward_F(sino, weightMap, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    backward_F(weightMap, sino, volumeSize, detectorSize, projectVector,
        volbiasz, dSampleInterval, dSliceInterval,
        sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
    weightMap = torch::relu(weightMap)+0.001;
    double maxWeight = torch::max(weightMap).item<double>();
    //weightMap = weightMap.div(maxWeight);
    x = torch::rand(imageshape).to(torch::kCUDA,device);
    auto time = print_time_elapsed_and_return_current_time(start, "start");
    for (int i = 0; i < 10; i++) {
        /*sino = forward(x, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, device);
        x = backward(sino, volumeSize, detectorSize, projectVector,volbiasz, dSampleInterval, dSliceInterval, sourceRadius, sourceZpos, fBiaz, SID, device);*/
        forward_F(sino, x, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
        backward_F(x, sino, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
        //x = torch::div(x, weightMap);
        x = x / x.norm(2);
    }
    //sino = forward(x, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, device);
    forward_F(sino, x, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    double s = sino.norm(2).item<double>();
    sino = sino.cpu().contiguous();
    std::string filename = "../data/sino.raw";
    write_tensor_to_binary_file(sino, filename);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << " seconds" <<"  power_method_H" << std::endl;
    std::cout << "H l2 norm " << s << std::endl;
    return s;
}

double power_method_gradient(int gradienttype,int device, torch::Tensor volumeSize)
{
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor x = torch::rand(imageshape).to(torch::kCUDA, device);
    for (int i = 0; i < 10; i++) {
        x = gradient(x, volumeSize, gradienttype);
        x = gradient(x, volumeSize, -gradienttype);
        x = x / x.norm(2);
    }
    x = gradient(x, volumeSize, gradienttype);
    double s = x.norm(2).item<double>();
    auto time = print_time_elapsed_and_return_current_time(start, "gradient");
    std::cout << "gradienttype:" << gradienttype << std::endl;
    return s;
}

double power_method_L(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectVector, float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum, int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID,double v1,double v2,double v3, double u) 
{
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor x1 = torch::rand(imageshape).to(torch::kCUDA, device);
    torch::Tensor x2 = x1;
    torch::Tensor x3 = x1;
    torch::Tensor x4 = x1;
    torch::Tensor x5 = x1;
    torch::Tensor sino = torch::zeros(sinoshape).to(torch::kCUDA, device);
    torch::Tensor weightMap = torch::ones(imageshape).to(torch::kCUDA, device);
    forward_F(sino, weightMap, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    backward_F(weightMap, sino, volumeSize, detectorSize, projectVector,
        volbiasz, dSampleInterval, dSliceInterval,
        sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
    weightMap = torch::relu(weightMap) + 0.001;
    double maxWeight = torch::max(weightMap).item<double>();
    //weightMap = weightMap.div(maxWeight);
    auto time = print_time_elapsed_and_return_current_time(start, "start");
    for (int i = 0; i < 10; i++) {        
        forward_F(sino, x1, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
        backward_F(x1, sino, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
        //x1 = torch::div(x1, weightMap);
        x2 = v1*gradient(x2, volumeSize, 1);
        x2 = v1*gradient(x2, volumeSize, -1);
        x3 = v2*gradient(x3, volumeSize, 2);
        x3 = v2*gradient(x3, volumeSize, -2);
        x4 = v3*gradient(x4, volumeSize, 3);
        x4 = v3*gradient(x4, volumeSize, -3);
        x5 = u * x5;
        x5 = u * x5;
        torch::Tensor x = x1+x2+x3+x4+x5;
        double xnorm = x.norm(2).item<double>();
        x1 = x / xnorm;
        x2 = x / xnorm;
        x3 = x / xnorm;
        x4 = x / xnorm;
        x5 = x / xnorm;
    }
    forward_F(sino, x1, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    x2 = v1 * gradient(x2, volumeSize, 1);
    x3 = v2 * gradient(x3, volumeSize, 2);
    x4 = v3 * gradient(x4, volumeSize, 3);
    x5 = u * x5;
    sino = sino.view({ -1 });
    x2 = x2.view({ -1 });
    x3 = x3.view({ -1 });
    x4 = x4.view({ -1 });
    x5 = x5.view({ -1 });
    std::vector<torch::Tensor> tensors = {sino,x2,x3,x4,x5};
    torch::Tensor x = torch::cat(tensors,0);
    double s = x.norm(2).item<double>();
    time = print_time_elapsed_and_return_current_time(start, "L norm");
    std::cout << s << std::endl;
    return s;
}

void dtv(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectVector, float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum, int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    //volume.zero_();
    double powerMethodH = power_method_H(real, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, AngleNum, SystemNum, device, sourceRadius, sourceZpos, fBiaz, SID) ;
    double v1 = powerMethodH / power_method_gradient(1, 0, volumeSize);
    double v2 = powerMethodH / power_method_gradient(2, 0, volumeSize);
    double v3 = powerMethodH / power_method_gradient(3, 0, volumeSize);
    double u = powerMethodH; 
    double L = power_method_L(real, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, AngleNum, SystemNum, device, sourceRadius, sourceZpos, fBiaz, SID,v1,v2,v3,u);
    double b = 0.5;
    double tao = b / L;
    double o = 1 / (b * L);
    int projWidth = 2304; 
    int projHeight = 2940;
    torch::Tensor p = torch::zeros(imageshape).to(torch::kCUDA, device);
    torch::Tensor q = torch::zeros(imageshape).to(torch::kCUDA, device);
    torch::Tensor s = torch::zeros(imageshape).to(torch::kCUDA, device);
    torch::Tensor t = torch::zeros(imageshape).to(torch::kCUDA, device);
    //torch::Tensor weightMap = torch::ones(imageshape).to(torch::kCUDA, device);
    torch::Tensor sino = torch::zeros(sinoshape).to(torch::kCUDA, device);
    torch::Tensor residual = torch::zeros(sinoshape).to(torch::kCUDA, device);
    /*forward_F(sino, weightMap, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    backward_F(weightMap, sino, volumeSize, detectorSize, projectVector,
        volbiasz, dSampleInterval, dSliceInterval,
        sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);*/
    torch::Tensor result = torch::zeros(imageshape).to(torch::kCUDA, device);
    //weightMap = torch::relu(weightMap)+0.001;
    //double maxWeight = torch::max(weightMap).item<double>();
    //weightMap = weightMap.div(maxWeight);
    /*backward_F(volume, real, volumeSize, detectorSize, projectVector,
        volbiasz, dSampleInterval, dSliceInterval,
        sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
    volume = volume.div(weightMap);*/
    torch::Tensor f = volume.clone();
    int Niter = 1001;
    double vol_l1 = 0;
    double tx = 6000, ty = 6000, tz = 8000;
    double txrate = 0.2, tyrate = 0.2, tzrate = 0.05;
    /*tx = gradient(volume, volumeSize, 1).norm(1).item<float>();
    ty = gradient(volume, volumeSize, 2).norm(1).item<float>();
    tz = gradient(volume, volumeSize, 3).norm(1).item<float>();*/
    auto time = print_time_elapsed_and_return_current_time(start_time, "start");
    /*for (int i = 0; i < 100; i++) 
    {
        forward_F(sino, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "forward");
        residual = (sino - real) ;
        backward_F(result, residual, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "backward");
        result.div(weightMap);
        volume = volume - 0.1*result;
    }*/

    print_cuda_use();
    for (int i = 0; i < Niter; i++) 
    {
        if (i % 1 == 0) 
        {
            vol_l1 = volume.abs().sum().item<float>();
            //tx = gradient(volume, volumeSize, 1).abs().sum().item<float>();
            //ty = gradient(volume, volumeSize, 2).abs().sum().item<float>();
            //tz = gradient(volume, volumeSize, 3).abs().sum().item<float>();
            //tx = gradient(volume, volumeSize, 1).norm(1).item<float>();
            //ty = gradient(volume, volumeSize, 2).norm(1).item<float>();
            //tz = gradient(volume, volumeSize, 3).norm(1).item<float>();
            tx = vol_l1 * txrate;
            ty = vol_l1 * tyrate;
            tz = vol_l1 * tzrate;
            std::cout << "tx:" << tx << std::endl;
            std::cout << "ty:" << ty << std::endl;
            std::cout << "tz:" << tz << std::endl;
        }
        time = print_time_elapsed_and_return_current_time(time, "before forward");
        forward_F(sino, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "after forward");
        residual = ((sino-real)*o+ residual)/(1+o);
        time = print_time_elapsed_and_return_current_time(time, "residual update");
        //residual = residual.view({ 1, 1, AngleNum, projHeight, projWidth });
        
        p.add_(o * v1 * gradient(volume, volumeSize, 1));
        time = print_time_elapsed_and_return_current_time(time, "p1");
        std::cout << "p:" << p.abs().sum().item<float>() << std::endl;
        q.add_(o * v2 * gradient(volume, volumeSize, 2));
        time = print_time_elapsed_and_return_current_time(time, "q1");
        
        s.add_(o * v3 * gradient(volume, volumeSize, 3));
        time = print_time_elapsed_and_return_current_time(time, "s1");
        
        p = p - o * (p.sign()).mul(l1ball(p.abs() / o, v1 * tx));
        time = print_time_elapsed_and_return_current_time(time, "p2");
        std::cout << "p:" << p.abs().sum().item<float>() << std::endl;
        q = q - o * (q.sign()).mul(l1ball(q.abs() / o, v2 * ty));
        time = print_time_elapsed_and_return_current_time(time, "q2");
        
        s = s - o * (s.sign()).mul(l1ball(s.abs() / o, v3 * tz));
        time = print_time_elapsed_and_return_current_time(time, "s2");
        std::cout << "volume:" << volume.sum().item<float>() << std::endl;
        t = -((-t - o * u * volume).relu());
        time = print_time_elapsed_and_return_current_time(time, "t");
        std::cout << "t:" << t.abs().sum().item<float>() << std::endl;
        result.zero_();
        time = print_time_elapsed_and_return_current_time(time, "before backward");
        backward_F(result, residual, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
        std::cout << "result:" << result.sum().item<float>() << std::endl;
        time = print_time_elapsed_and_return_current_time(time, "after backward");
        //result.div(weightMap);
        print_cuda_use();
        //real= real.cpu().contiguous();
        //sino = sino.cpu().contiguous();
        //residual = residual.cpu().contiguous();
        print_cuda_use();
        //volume.copy_(f - 2*tao*(result+v1*gradient(p, volumeSize, -1)+ v2*gradient(q, volumeSize, -2)+ v3*gradient(s, volumeSize, -3)+u*t));
        result.add_(v1 * gradient(p, volumeSize, -1));
        //result.add_(v1 * BackGradient(p,0));
        std::cout << "result:" << result.sum().item<float>() << std::endl;
        time = print_time_elapsed_and_return_current_time(time, "addp");
        result.add_(v2 * gradient(q, volumeSize, -2));
        std::cout << "result:" << result.sum().item<float>() << std::endl;
        time = print_time_elapsed_and_return_current_time(time, "addq");
        result.add_(v3 * gradient(s, volumeSize, -3));
        std::cout << "result:" << result.sum().item<float>() << std::endl;
        time = print_time_elapsed_and_return_current_time(time, "adds");
        result.add_(u * t);
        std::cout << "result:" << result.sum().item<float>() << std::endl;
        time = print_time_elapsed_and_return_current_time(time, "addt");
        volume.copy_(f-2*tao* result);
        time = print_time_elapsed_and_return_current_time(time, "update volume");
        print_cuda_use();
        //f = f - tao * (result + v1 * gradient(p, volumeSize, -1) + v2 * gradient(q, volumeSize, -2) + v3 * gradient(s, volumeSize, -3) + u * t);
        f.sub_(tao * result);
        time = print_time_elapsed_and_return_current_time(time, "update f");
        print_cuda_use();
        //real = real.to(torch::kCUDA, device);
        //sino = sino.to(torch::kCUDA, device);
        //residual = residual.to(torch::kCUDA, device);
        //volume = volume -  tao * (result + v1 * gradient(p, volumeSize, -1) + v2 * gradient(q, volumeSize, -2) + v3 * gradient(s, volumeSize, -3) );
        //volume = volume - 2*tao * result;
        time = print_time_elapsed_and_return_current_time(time, "iter");
        std::cout << i << std::endl;
        print_cuda_use();
        if (i % 5 == 0)
        {
            volume = volume.cpu().contiguous();
            std::string filename = "../data/test/dtvResults/"+ std::to_string(i)+".raw";
            write_tensor_to_binary_file(volume, filename);
            volume = volume.to(torch::kCUDA, device);
            time = print_time_elapsed_and_return_current_time(time, "save");
        }
    }
    /*volume = volume.cpu().contiguous();
    std::string filename = "../data/test/dtvLabel/high/1/dtv.raw" ;
    write_tensor_to_binary_file(volume, filename);*/
}

torch::Tensor bp(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectMatrix, torch::Tensor solutionSpace,float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum, int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    volume.zero_();
    int projWidth = 2304;
    int projHeight = 2940;
    torch::Tensor residual = torch::zeros(sinoshape).to(torch::kCUDA, device);
    torch::Tensor result = torch::zeros(imageshape).to(torch::kCUDA, device);
    auto time = print_time_elapsed_and_return_current_time(start_time, "start");
    for (int i = 0; i < 1; i++)
    {
        //forward_P(sino, volume, volumeSize, detectorSize, projectMatrix,solutionSpace, volbiasz, dSampleInterval, dSliceInterval, device);
        
        forward_F(residual, volume, volumeSize, detectorSize, projectMatrix, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "forward");
        //residual.copy_(residual - real) ;
        // 
        residual.sub_(real);
        //backward_P(result, residual, volumeSize, detectorSize, projectMatrix, volbiasz, dSampleInterval, dSliceInterval, device);
        backward_F(result, residual, volumeSize, detectorSize, projectMatrix,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "backward");

        //result.div(weightMap);
        volume .sub_(result);
    }
    volume = volume.cpu().contiguous();
    std::string filename = "../data/sirt3.raw";
    write_tensor_to_binary_file(volume, filename);
    volume = volume.to(torch::kCUDA, device);
    time = print_time_elapsed_and_return_current_time(time, "save");
    return volume;
}

void fdk(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectVector, float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum, int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID) 
{
    int projWidth = 2303;
    int projHeight = 5605;
    torch::Tensor ramp = ramp_filter(projWidth).to(torch::kCUDA, device);
    torch::nn::functional::Conv2dFuncOptions conv_options;
    conv_options.stride({ 1, 1 }).padding({ 0, static_cast<int>(projWidth / 2) });
    torch::Tensor sino = torch::zeros(sinoshape1).to(torch::kCUDA, device);
    sino.copy_(real);
    real = real.cpu().contiguous();

    //torch::Tensor residual = torch::zeros(sinoshape1).to(torch::kCUDA, device);
    //torch::Tensor result = torch::zeros(imageshape).to(torch::kCUDA, device);
    //sino = forward(volume, volumeSize, detectorSize, projectVector, volbiasz, 1.0, 1.0, device);
    //residual = sino - real;
    //residual = residual.view({ 1, 1, AngleNum, projHeight, projWidth });
    //residual = cosweight(residual, detectorSize, projectVector, 0);
    // Assuming 'residual' and 'x' tensors are already defined and in GPU memory
    //residual = residual.view({ 1, 1, AngleNum * projHeight, projWidth });
    //residual = torch::nn::functional::conv2d((residual), ramp, conv_options);
    // Assuming 'residual' and 'volume' tensors are already defined and in GPU memory
    //residual = residual.view({ 1, 1, AngleNum, projHeight, projWidth });

    sino = sino.view({ 1, 1, AngleNum, projHeight, projWidth });
    sino = cosweight(sino, detectorSize, projectVector, 0);
    sino = sino.view({ 1, 1, AngleNum * projHeight, projWidth });
    sino = torch::nn::functional::conv2d((sino), ramp, conv_options);
    sino = sino.view({ 1, 1, AngleNum, projHeight, projWidth });
    volume.copy_(backward(sino, volumeSize, detectorSize, projectVector,
        volbiasz, dSampleInterval, dSliceInterval,
        sourceRadius, sourceZpos, fBiaz, SID, device));
    volume = volume.cpu().contiguous();
    std::string filename = "../data/test/shenzhenhigh/results/1/fdkResults.raw";
    write_tensor_to_binary_file(volume, filename);
    volume = volume.to(torch::kCUDA, device);
    real = real.to(torch::kCUDA, device);
}

void dtvnet(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectVector, float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum,int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID)
{
    float ntx[5] = { -0.0264,-0.0265,-0.0266,-0.0269,-0.0271 };
    float nty[5] = { -0.0239,-0.0240,-0.0242,-0.0246,-0.0248 };
    float ntz[5] = { -0.0266,-0.0267,-0.0269,-0.0272,-0.0274 };
    float nt[5] = { -1.1668,-1.1352,-1.0398,-0.8805,-0.8839 };
    auto start_time = std::chrono::high_resolution_clock::now();
    int projWidth = 1629; // Replace with the appropriate value from shenzhenDetectorSize[0]
    int projHeight = 1629;
    torch::Tensor ramp = shepp_logan_filter(projWidth).to(torch::kCUDA, device);
    auto time = print_time_elapsed_and_return_current_time(start_time, "calculate ramp");
    // 
    torch::nn::functional::Conv2dFuncOptions conv_options;
    conv_options.stride({ 1, 1 }).padding({ 0, static_cast<int>(projWidth / 2) });
    time = print_time_elapsed_and_return_current_time(time, "prepare convoptions");
    torch::Tensor p = torch::zeros(imageshape).to(torch::kCUDA, device);
    torch::Tensor q = torch::zeros(imageshape).to(torch::kCUDA, device);
    torch::Tensor s = torch::zeros(imageshape).to(torch::kCUDA, device);
    torch::Tensor Pnew = torch::zeros(imageshape).to(torch::kCUDA, device);
    torch::Tensor resultgrad = torch::zeros(imageshape).to(torch::kCUDA, device);

    //calculate weight map
    torch::Tensor weightMap = torch::ones(imageshape).to(torch::kCUDA, device);
    torch::Tensor sino = torch::zeros(sinoshape).to(torch::kCUDA, device);
    /*sino = forward(weightMap, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, device);
    weightMap = backward(sino, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, device);*/
    forward_F(sino, weightMap, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    backward_F(weightMap, sino, volumeSize, detectorSize, projectVector,
        volbiasz, dSampleInterval, dSliceInterval,
        sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
    weightMap = torch::relu(weightMap)+0.001;
    double maxWeight = torch::max(weightMap).item<double>();
    weightMap = weightMap.div(maxWeight);
    torch::Tensor result = torch::zeros(imageshape).to(torch::kCUDA, device);
    torch::Tensor residual = torch::zeros(sinoshape).to(torch::kCUDA, device);
    for (int i = 0; i < 20; i++) {
        // 调用 forward 函数
        sino.zero_();
        residual.zero_();
        //sino = forward(volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, device);
        forward_F(sino, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "forward");
        residual = sino - real;
        residual = residual.view({ 1, 1, AngleNum, projHeight, projWidth });
        result.zero_();
        time = print_time_elapsed_and_return_current_time(time, "backward before");
        /*result = backward(residual, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, device);*/
        backward_F(result, residual, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);
        time = print_time_elapsed_and_return_current_time(time, "backward");
        volume = volume - result;
    }

    // Apply ReLU activation
    result = torch::relu(volume);
    result=torch::div(result,weightMap);
    volume = result;
    time = print_time_elapsed_and_return_current_time(time, "div");
    result = result.cpu().contiguous();
    std::string filename = "../data/fdk_fuck.raw";
    write_tensor_to_binary_file(result, filename);
    result = result.to(torch::kCUDA, device);
    weightMap = weightMap.cpu().contiguous();
    filename = "../data/weightMap.raw";
    write_tensor_to_binary_file(weightMap, filename);
    weightMap = weightMap.to(torch::kCUDA, device);
    std::cout << weightMap.device() << std::endl;
    print_cuda_use();
    c10::cuda::CUDACachingAllocator::emptyCache();
    time = print_time_elapsed_and_return_current_time(time, "saveFDK");
    torch::Tensor t = volume;
    for (int i = 0; i < 5; i++)
    {
        time = print_time_elapsed_and_return_current_time(time, "iteration begin");
        std::cout << "Iteration " << i << std::endl;
        //torch::Device device(torch::kCUDA, 0);
        // 调用 forward 函数
        sino.zero_();
        //sino = forward(volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, device);
        forward_F(sino, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, 0);
        time = print_time_elapsed_and_return_current_time(time, "forward");
        torch::Tensor residual = sino - real;
        //residual = residual.view({ 1, 1, AngleNum, projWidth, projWidth });
        //residual = cosweight(residual, detectorSize, projectVector, 0);
        //// Assuming 'residual' and 'x' tensors are already defined and in GPU memory
        //residual = residual.view({ 1, 1, AngleNum * projWidth, projWidth });
        //residual = torch::nn::functional::conv2d((residual), ramp, conv_options);
        //time = print_time_elapsed_and_return_current_time(time, "conv");
        //// Assuming 'residual' and 'volume' tensors are already defined and in GPU memory
        residual = residual.view({ 1, 1, AngleNum, projWidth, projWidth });
        result.zero_();
        time = print_time_elapsed_and_return_current_time(time, "backward before");
        /*result = backward(residual, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, device);*/
        backward_F(result, residual, volumeSize, detectorSize, projectVector,
            volbiasz, dSampleInterval, dSliceInterval,
            sourceRadius, sourceZpos, fBiaz, SID, SystemNum, 0);
        result = volume - result;
        // Apply ReLU activation
        result = torch::relu(result);
        result.div(weightMap);
        time = print_time_elapsed_and_return_current_time(time, "backward");
        time = print_time_elapsed_and_return_current_time(time, "grad");

        c10::cuda::CUDACachingAllocator::emptyCache();

        print_cuda_use();
        resultgrad = gradient(result, volumeSize, 1);
        infer("../engine/8channel/gradx.engine", resultgrad.data<float>(), Pnew.data<float>());
        p = p + ntx[i] * (p - Pnew);

        resultgrad = gradient(result, volumeSize, 2);
        infer("../engine/8channel/grady.engine", resultgrad.data<float>(), Pnew.data<float>());
        q = q + nty[i] * (q - Pnew);

        resultgrad = gradient(result, volumeSize, 3);
        infer("../engine/8channel/gradz.engine", resultgrad.data<float>(), Pnew.data<float>());
        s = s + ntz[i] * (s - Pnew);

        infer("../engine/8channel/ae1600.engine", result.data<float>(), Pnew.data<float>());
        auto z_ = t + nt[i] * (t - Pnew);

        time = print_time_elapsed_and_return_current_time(time, "network");

        torch::Tensor Size2 = torch::tensor({ 1600, 1600, 100 }, torch::kInt);
        t = gradient(p, Size2, -1);
        t += gradient(q, Size2, -2);
        t += gradient(s, Size2, -3);
        t += z_;
        time = print_time_elapsed_and_return_current_time(time, "grad");
        volume = t;
        time = print_time_elapsed_and_return_current_time(time, "iter");
        if (i == 4)
        {

            std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
            std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
            auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            // 将时间结构转换为本地时间
            struct tm localTime;
            localtime_s(&localTime, &nowTime);

            // 输出实时时间
            char timeStr[30];
            strftime(timeStr, sizeof(timeStr), "%Y-%m-%d-%H-%M-%S", &localTime);
            std::string timeNow = timeStr;

            filename = "../data/" + timeNow + ".raw";
            std::cout << filename << std::endl;
            t = t.cpu();
            write_tensor_to_binary_file(t, filename);
            std::cout << "done" << std::endl;
        }
    }
    auto endtime = print_time_elapsed_and_return_current_time(start_time, "end");
}

void flySimulation(torch::Tensor real, torch::Tensor volume, torch::Tensor volumeSize, torch::Tensor detectorSize, torch::Tensor projectVector, float volbiasz, float dSampleInterval, float dSliceInterval, int AngleNum, int SystemNum, int device, float sourceRadius, float sourceZpos, float fBiaz, float SID)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    int rotationnum = SystemNum;
    double sourceR = sourceRadius,detR= SID;
    double sourceZ = sourceZpos, detZ = sourceZpos + fBiaz;
    torch::Tensor sourcepos = torch::zeros({3,rotationnum * AngleNum });
    torch::Tensor detcenpos = torch::zeros({ 3,rotationnum * AngleNum });
    torch::Tensor dir_u = torch::zeros({ 3,rotationnum * AngleNum });
    torch::Tensor dir_v = torch::zeros({ 3,rotationnum * AngleNum });
    torch::Tensor blurImg = torch::zeros({1,1,160,2940,2304}).to(torch::kCUDA, device);
    torch::Tensor labelImg = torch::zeros({ 1,1,160,2940,2304 }).to(torch::kCUDA, device);
    for (int i = 0; i < rotationnum * AngleNum;i++) 
    {
            double Angle = (i) * 2 * M_PI / (rotationnum * AngleNum);
            torch::Tensor R2 = torch::tensor({ {std::cos(Angle), -std::sin(Angle), 0.0},
                                              {std::sin(Angle), std::cos(Angle), 0.0},
                                              {0.0, 0.0, 1.0} }, torch::kFloat);

            sourcepos.slice(1, i, i + 1) = torch::tensor({ {-std::cos(Angle) * sourceR},
                                                           {-std::sin(Angle) * sourceR },
                                                           { sourceZ} }, torch::kFloat);
            dir_u.slice(1, i, i + 1) = torch::matmul(R2, torch::tensor({ {1.0}, {0.0}, {0.0} }, torch::kFloat));
            dir_v.slice(1, i, i + 1) = torch::matmul(R2, torch::tensor({ {0.0}, {1.0}, {0.0} }, torch::kFloat));

            detcenpos.slice(1, i, i + 1) = torch::tensor({ {std::cos(Angle) * detR},
                                                           {std::sin(Angle) * detR },
                                                           {detZ} }, torch::kFloat);
    }
    torch::Tensor proj_vec_F = torch::cat({ sourcepos.t(), detcenpos.t(), dir_u.t(), dir_v.t()}, 1);
    std::cout << proj_vec_F.sizes() << std::endl;
    std::string resultsFolder = "../data/label";
    std::string blurFolder = "../data/blur";
    std::filesystem::create_directories(blurFolder);
    std::filesystem::create_directories(resultsFolder);
    auto time = print_time_elapsed_and_return_current_time(start_time, "start");
    for (int i = 0; i < AngleNum; i++) 
    {
        time = print_time_elapsed_and_return_current_time(time, "iter");
        std::cout << i << std::endl;
        auto img = torch::zeros({ 2940,2304 }).to(torch::kCUDA, device);
        auto img1 = torch::zeros({ 2940,2304 }).to(torch::kCUDA, device);
        for (int j = 0; j < rotationnum; j++)
        {
             int index = i * rotationnum + j ;
             auto sourcepos_col = sourcepos.slice(1, index, index + 1).squeeze();
             auto detcenpos_col = detcenpos.slice(1, index, index + 1).squeeze();
             auto dir_u_col = dir_u.slice(1, index, index + 1).squeeze();
             auto dir_v_col = dir_v.slice(1, index, index + 1).squeeze();
             torch::Tensor proj_vec = torch::cat({ sourcepos_col, detcenpos_col, dir_u_col, dir_v_col });
             proj_vec = proj_vec.to(torch::kCUDA, device);
             proj_vec = proj_vec.view({ 1,12 });
             forward_F(real, volume, volumeSize, detectorSize, proj_vec, volbiasz, dSampleInterval, dSliceInterval, 1, device);
             img1 = real.index({ torch::indexing::Slice(0,1),
                                    torch::indexing::Slice(0,1),
                                    torch::indexing::Slice(0,1),
                                    torch::indexing::Slice(),
                                    torch::indexing::Slice() }).squeeze();
             img = img + img1;
             if (j == 0) {
                 labelImg.index({ torch::indexing::Slice(),torch::indexing::Slice(),i,torch::indexing::Slice(),torch::indexing::Slice() }) = img1;
             }
        }
        img = img / 100;
        blurImg.index({ torch::indexing::Slice(),torch::indexing::Slice(),i,torch::indexing::Slice(),torch::indexing::Slice() }) = img;
    }
    std::string filename = "010.raw";
    blurImg = blurImg.cpu().contiguous();
    write_tensor_to_binary_file(blurImg, blurFolder + '/'+filename);
    labelImg = labelImg.cpu().contiguous();
    write_tensor_to_binary_file(labelImg, resultsFolder + '/' + filename);
    write_tensor_to_binary_file(proj_vec_F, "E:/deblur2D/deblur2D/data/proj_vec_F.raw");
}

int main() {
    // 设置参数
    // 湖熟低倍
    //float volbiasz = -50.0f /*192.0f*/;
    //float dSampleInterval = 0.438f;//0.438f;
    //float dSliceInterval = 1.0f;
    //float sourceRadius = 905  /*663.4656*/;
    //float sourceZpos = -1774 /* -1314*/;
    //float fBiaz = 4018  /*3597*/;
    //float SID = 2115  /*1774.6*/;
    //float volbiasz = -870.0f /*192.0f*/;
    //float dSampleInterval = /*1.0f*/0.3238f;
    //float dSliceInterval = 1.0f;
    //float sourceRadius = 663.4656  /*663.4656*/;
    //float sourceZpos = -1164.3368 /* -1314*/;
    //float fBiaz = 3595.6166  /*3597*/;
    //float SID = 1774.5977  /*1774.6*/;
    //中北
    //float volbiasz = -20.0f /*192.0f*/;
    //float dSampleInterval = 0.5184f;
    //float dSliceInterval = 3.0f;
    //float sourceRadius = /*905*/  663.4656;
    //float sourceZpos = /*-1774*/  -1314;
    //float fBiaz = /*4018*/  4299;
    //float SID = /*2115*/  2127;
    //深圳低倍
    /*float volbiasz = 0.0f;
    float dSampleInterval = 0.5184f;
    float dSliceInterval = 1.0f;
    float sourceRadius = 909.8944;
    float sourceZpos = -2206.5688;
    float fBiaz = 4256.3759;
    float SID = 2127.6967;*/
    //深圳高倍
    float volbiasz = -55.0f ;
    float dSampleInterval = 0.125f;
    float dSliceInterval = 0.50f;
    float sourceRadius = 302.8957 ;
    float sourceZpos = -582.5868 ;
    float fBiaz = 4299.0880 ;
    float SID = 1806.7501  ;
    //仿真do
    //float volbiasz = 100.0f;
    //float dSampleInterval = 0.25f;
    //float dSliceInterval = 0.25f;
    //float sourceRadius = 303.0303;
    //float sourceZpos = -582.5868;
    //float fBiaz = 4299.0880;
    //float SID = 2020.2;
    long device = 0;
    const int AngleNum = 32;
    int projWidth = 2304; // Replace with the appropriate value from shenzhenDetectorSize[0]
    int projHeight = 2940;
    //int projWidth = 2303; 
    //int projHeight = 2605;
    const int SystemNum = 1;
    //
    auto start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor volumeSize = torch::tensor({ 1600, 1600, 100 }, torch::kInt).to(torch::kCUDA, device);
    torch::Tensor detectorSize = torch::tensor({ projWidth, projHeight }, torch::kInt).to(torch::kCUDA, device);
    torch::Tensor detectorSize_r = torch::tensor({ 2303, 5605 }, torch::kInt).to(torch::kCUDA, device);
    torch::Tensor projectVector = load_raw_file_to_tensor(
        "../data/test/shenzhenhigh/results/1/proj_vec.raw",
        1, 1, AngleNum * SystemNum, 12, device);
    projectVector = projectVector.view({ AngleNum * SystemNum, 12 });
    torch::Tensor projectVector_r = load_raw_file_to_tensor(
        "../data/test/shenzhenhigh/results/1/proj_vec_r1.raw",
        1, 1, AngleNum * 1, 12, device);
    projectVector_r = projectVector_r.view({ AngleNum * 1, 12 });
    /*torch::Tensor projectMatrix = load_raw_file_to_tensor(
        "../data/test/highF/H.raw",
        1, 1, AngleNum , 12, device);
    projectMatrix = projectMatrix.view({ AngleNum , 12 });
    torch::Tensor SolutionSpace = load_raw_file_to_tensor(
        "../data/test/highF/S.raw",
        1, 1, AngleNum, 16, device);
    SolutionSpace = SolutionSpace.view({ AngleNum , 16 });*/ 
    //
    //torch::Tensor volume = load_raw_file_to_tensor("../data/voxel.raw",1,100,1600,1600, device);
    //volume = volume.view({ 1,1,100,1600,1600 });
    torch::Tensor real = load_raw_file_to_tensor("../data/test/shenzhenhigh/results/10/gM.raw", 1, AngleNum, projHeight, projWidth, device);
    //real = real / real.norm(2);
    //torch::Tensor real_r = load_raw_file_to_tensor("../data/test/shenzhenhigh/simulation/circle/800_800_100/sino_r.raw", 1, AngleNum, 2605, 2303, device);
    torch::Tensor volume = torch::ones(imageshape).to(torch::kCUDA, device);
    real = real.view({ 1, 1, AngleNum , projHeight, projWidth });
    //real_r = real_r.view({ 1, 1, AngleNum , 2605, 2303 });
    
    
    /*backward_F(volume, real, volumeSize, detectorSize, projectVector,
        volbiasz, dSampleInterval, dSliceInterval,
        sourceRadius, sourceZpos, fBiaz, SID, SystemNum, device);*/
    //torch::Tensor real = torch::zeros(sinoshape).to(torch::kCUDA, device);
    torch::Tensor real_r = torch::zeros(sinoshape1).to(torch::kCUDA, device);
    //rotation(real_r,real,  detectorSize, detectorSize_r, projectVector,projectVector_r, device);
    //forward_F(real_r, volume, volumeSize, detectorSize_r, projectVector_r, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    
    //fdk(real_r, volume, volumeSize, detectorSize_r, projectVector_r, volbiasz, dSampleInterval, dSliceInterval, AngleNum, SystemNum, device, sourceRadius, sourceZpos, fBiaz, SID);
    //forward_F(real, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    //forward_F(real_r, volume, volumeSize, detectorSize_r, projectVector_r, volbiasz, dSampleInterval, dSliceInterval, SystemNum, device);
    
    /*std::string filename1 = "../data/test/shenzhenhigh/simulation/32/sino.raw";
    real = real.cpu().contiguous();
    write_tensor_to_binary_file(real, filename1);
    real = real.to(torch::kCUDA, device);*/
    
    //std::string filename2 = "../data/test/shenzhenhigh/1/sino_r_3.raw";
    //real_r = real_r.cpu().contiguous();
    //write_tensor_to_binary_file(real_r, filename2);
    //real_r = real_r.to(torch::kCUDA, device); 
    
    //flySimulation(real, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, AngleNum, SystemNum, device, sourceRadius, sourceZpos, fBiaz, SID);
    //bp(real, volume, volumeSize, detectorSize, projectVector,SolutionSpace, volbiasz, dSampleInterval, dSliceInterval, AngleNum, SystemNum, device, sourceRadius, sourceZpos, fBiaz, SID);
    dtv(real, volume, volumeSize, detectorSize, projectVector, volbiasz, dSampleInterval, dSliceInterval, AngleNum, SystemNum, device, sourceRadius, sourceZpos, fBiaz, SID);
    //dtvnet(real, volume,  volumeSize,  detectorSize,  projectVector,  volbiasz, dSampleInterval,  dSliceInterval,  AngleNum,  SystemNum,  device,  sourceRadius,  sourceZpos,  fBiaz,  SID);
}
