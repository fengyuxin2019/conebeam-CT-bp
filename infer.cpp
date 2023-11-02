//
// Created by 单淳劼 on 2022/7/14.
//

#include "infer.h"

Infer::Infer(std::string trtEnginePath, int inputSize)
    : m_enginePath(trtEnginePath)
    , m_inputSize(inputSize) // `1` & `3` means batch_size and channels
    , m_outputSize(inputSize)
{
    initialize();
}

Infer::~Infer()
{
    delete m_context;
    delete m_engine;
    delete m_runtime;
    m_blob = NULL;
    m_prob = NULL;
    //delete[] m_blob;  //libtorch delete the memory so don't need this
    //delete[] m_prob;
}

int Infer::initialize()
{
    cudaSetDevice(0);
    char* trtModelStream{ nullptr };
    size_t size{ 0 };

    std::ifstream file(m_enginePath, std::ios::binary);
    std::cout << "[I] Swin Transformer model creating...\n";
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    // test for half FP16 support
    //nvinfer1::IBuilder* builder = createInferBuilder(m_gLogger);

    //if (builder != NULL)
    //{
    //    bool mEnableFP16 = builder->platformHasFastFp16();
    //    std::cout << mEnableFP16;
    //    builder->destroy();
    //}



    m_runtime = createInferRuntime(m_gLogger);
    assert(m_runtime != nullptr);


    std::cout << "[I] Swin Transformer engine creating...\n";
    m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size);
    assert(m_engine != nullptr);
    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);
    delete[] trtModelStream;

    auto out_dims = m_engine->getBindingDimensions(1);

    //m_blob = new float[m_inputSize];
    //m_prob = new float[m_outputSize];

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(m_engine->getNbBindings() == 2);
    std::cout << "[I] Cuda buffer creating...\n";

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    m_inputIndex = m_engine->getBindingIndex("input.1");
    assert(m_engine->getBindingDataType(m_inputIndex) == nvinfer1::DataType::kFLOAT);
    m_outputIndex = m_engine->getBindingIndex("53");
    assert(m_engine->getBindingDataType(m_outputIndex) == nvinfer1::DataType::kFLOAT);
    //int mBatchSize = m_engine->getMaxBatchSize();

    // Create GPU buffers on device
    checkStatus(cudaMalloc(&m_buffers[m_inputIndex], m_inputSize * sizeof(float)));
    checkStatus(cudaMalloc(&m_buffers[m_outputIndex], m_outputSize * sizeof(float)));

    // Create stream
    std::cout << "[I] Cuda stream creating...\n";
    checkStatus(cudaStreamCreate(&m_stream));

    std::cout << "[I] Swin-Transformer engine created!\n";

    return 0;
}

void Infer::doInference(float* fInput, float* fOutput, int i)
{
    m_blob = fInput;
    m_prob = fOutput;
    int ret;
    if (i == 0)
        ret = doSingleInfer();
    else
        ret = doSingleInfercon();
    if (ret)
    {
        std::cout << "[I] Inference phase ends.\n";
        //return m_prob;
    }
    else
    {
        //return nullptr;
    }
}


int Infer::doSingleInfer()
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    checkStatus(cudaMemcpyAsync(m_buffers[m_inputIndex], m_blob, m_inputSize * sizeof(float), cudaMemcpyDeviceToDevice, m_stream));
    m_context->enqueueV2(m_buffers, m_stream, nullptr);
    checkStatus(cudaMemcpyAsync(m_prob, m_buffers[m_outputIndex], m_outputSize * sizeof(float), cudaMemcpyDeviceToDevice, m_stream));
    cudaStreamSynchronize(m_stream);
    auto end = std::chrono::system_clock::now();

    std::cout << "[I] Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    cudaFree(m_buffers[m_inputIndex]);
    cudaFree(m_buffers[m_outputIndex]);
    return 1;
}

int Infer::doSingleInfercon()
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    checkStatus(cudaMemcpyAsync(m_buffers[m_inputIndex], m_blob, m_inputSize * sizeof(float), cudaMemcpyHostToDevice, m_stream));
    m_context->enqueueV2(m_buffers, m_stream, nullptr);
    checkStatus(cudaMemcpyAsync(m_prob, m_buffers[m_outputIndex], m_outputSize * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
    cudaStreamSynchronize(m_stream);
    auto end = std::chrono::system_clock::now();

    std::cout << "[I] Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    return 1;
}