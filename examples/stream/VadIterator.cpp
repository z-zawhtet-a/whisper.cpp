#include "VadIterator.h"
#include <iostream>

// Constructor
VadIterator::VadIterator(const std::string ModelPath, int Sample_rate, int frame_size, float Threshold, float minimum_speech_duration)
{
    init_onnx_model(ModelPath);
    sample_rate = Sample_rate;
    sr_per_ms = sample_rate / 1000;
    threshold = Threshold;
    minimum_speech_duration = minimum_speech_duration;
    window_size_samples = frame_size * sr_per_ms;

    input.resize(window_size_samples);
    input_node_dims[0] = 1;
    input_node_dims[1] = window_size_samples;
    _h.resize(size_hc);
    _c.resize(size_hc);
    sr.resize(1);
    sr[0] = sample_rate;
}
void VadIterator::init_engine_threads(int inter_threads, int intra_threads)
{
    // The method should be called in each thread/proc in multi-thread/proc work
    session_options.SetIntraOpNumThreads(intra_threads);
    session_options.SetInterOpNumThreads(inter_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

void VadIterator::init_onnx_model(const std::string &model_path)
{
    // Init threads = 1 for
    init_engine_threads(1, 1);
    // Load model
    session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
}

void VadIterator::reset_states()
{
    // Call reset before each audio start
    std::memset(_h.data(), 0.0f, _h.size() * sizeof(float));
    std::memset(_c.data(), 0.0f, _c.size() * sizeof(float));
}

// Call it in predict func for raw bytes input.
void VadIterator::bytes_to_float_tensor(const char *pcm_bytes)
{
    const int16_t *in_data = reinterpret_cast<const int16_t *>(pcm_bytes);
    for (int i = 0; i < window_size_samples; i++)
    {
        input[i] = static_cast<float>(in_data[i]) / 32768; // int16_t normalized to float
    }
}

bool VadIterator::predict(const std::vector<float> &data)
{
    // bytes_to_float_tensor(data);
    int chunks = data.size() / window_size_samples;
    unsigned int total_speech_duration = 0; // Counter for total speech duration

    for (int i = 0; i < chunks; ++i)
    {
        // Extract the 64ms chunk from the 2000ms data
        auto chunk_start = data.begin() + i * window_size_samples;
        auto chunk_end = chunk_start + window_size_samples;
        std::vector<float> chunk_data(chunk_start, chunk_end);

        // Infer
        // Create ort tensors
        input.assign(data.begin(), data.end());
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
        Ort::Value h_ort = Ort::Value::CreateTensor<float>(
            memory_info, _h.data(), _h.size(), hc_node_dims, 3);
        Ort::Value c_ort = Ort::Value::CreateTensor<float>(
            memory_info, _c.data(), _c.size(), hc_node_dims, 3);

        // Clear and add inputs
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(sr_ort));
        ort_inputs.emplace_back(std::move(h_ort));
        ort_inputs.emplace_back(std::move(c_ort));

        // Infer
        ort_outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        // Output probability & update h,c recursively
        float output = ort_outputs[0].GetTensorMutableData<float>()[0];
        float *hn = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_h.data(), hn, size_hc * sizeof(float));
        float *cn = ort_outputs[2].GetTensorMutableData<float>();
        std::memcpy(_c.data(), cn, size_hc * sizeof(float));

        // Check if output exceeds threshold
        if (output >= threshold)
        {
            total_speech_duration += 64; // Add 64ms for each detected speech chunk
        }

        // Break early if total speech duration exceeds minimum speech duration
        if (total_speech_duration >= minimum_speech_duration)
        {
            return true;
        }
    }

    // Check if the total speech duration meets the minimum speech duration threshold
    return total_speech_duration >= minimum_speech_duration;
}

