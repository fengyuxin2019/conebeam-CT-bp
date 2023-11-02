//
// 
//

#ifndef INC_23_03_22_SWIN_DEPLOYMENT_INFER_H
#define INC_23_03_22_SWIN_DEPLOYMENT_INFER_H

#include "NvInfer.h"
#include "logging.h"
#include <fstream>
#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <chrono>
#include "cuda_runtime_api.h"
//#include <opencv2/opencv.hpp>

using namespace nvinfer1;


class Infer {
public:
    Infer(std::string trtEnginePath, int inputSize);

    ~Infer();

    void doInference(float* fInput, float* fOutput, int i);

private:
    std::string m_enginePath;
    Logger m_gLogger;
    IRuntime* m_runtime;
    ICudaEngine* m_engine;
    IExecutionContext* m_context;
    cudaStream_t m_stream;

    int m_inputSize;
    int m_outputSize;
    int m_inputIndex;
    int m_outputIndex;

    float* m_blob;
    float* m_prob;
    void* m_buffers[2];

    int initialize();

    int doSingleInfer();
    int doSingleInfercon();

    static void checkStatus(cudaError status)
    {
        do
        {
            auto ret = (status);
            if (ret != 0)
            {
                std::cerr << "Cuda failure: " << ret << std::endl;
                abort();
            }
        } while (0);
    }

};


#endif //INC_23_03_22_SWIN_DEPLOYMENT_INFER_H
