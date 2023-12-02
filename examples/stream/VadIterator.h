#ifndef VAD_ITERATOR_H
#define VAD_ITERATOR_H

#include <vector>
#include <cstring>
#include "onnxruntime_cxx_api.h"

class VadIterator
{
public:
    VadIterator(const std::string ModelPath, int Sample_rate, int frame_size, float Threshold, float minimum_speech_duration);
    
    void init_engine_threads(int inter_threads, int intra_threads);
    void init_onnx_model(const std::string &model_path);
    void reset_states();
    void bytes_to_float_tensor(const char *pcm_bytes);
    bool predict(const std::vector<float> &data);

private:
    // OnnxRuntime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    // model config
    int64_t window_size_samples;
    int sample_rate;
    int sr_per_ms;
    float threshold;
    float minimum_speech_duration; // Minimum speech duration in miliseconds

    float output;

    // Onnx model - Inputs
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char *> input_node_names = {"input", "sr", "h", "c"};
    std::vector<float> input;
    std::vector<int64_t> sr;
    unsigned int size_hc = 2 * 1 * 64;
    std::vector<float> _h;
    std::vector<float> _c;

    int64_t input_node_dims[2] = {};
    const int64_t sr_node_dims[1] = {1};
    const int64_t hc_node_dims[3] = {2, 1, 64};

    // Outputs
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char *> output_node_names = {"output", "hn", "cn"};
};

#endif // VAD_ITERATOR_H
