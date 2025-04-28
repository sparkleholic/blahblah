#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <cxxopts.hpp>

// Include tokenizers-cpp if available
#ifdef USE_TOKENIZERS_CPP
#include <tokenizers_cpp.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

// Structure to hold model configuration
struct ModelConfig {
    int num_hidden_layers = 16;
    int num_attention_heads = 32;
    int num_key_value_heads = 8;
    int head_dim = 64;
    int hidden_size = 2048;
    int intermediate_size = 8192;  // Added this parameter
    std::vector<int> eos_token_ids = {128001, 128008, 128009};
    int bos_token_id = 128000;
    bool attention_bias = false;
    float attention_dropout = 0.0f;
    int max_position_embeddings = 131072;
    float rope_theta = 500000.0f;
    std::string hidden_act = "silu";
};

// Function to check ONNX Runtime version compatibility
void check_ort_version() {
    // In ONNX Runtime 1.21, we don't need to check the API version
    // Just make sure we're using the correct API
    std::cout << "Using ONNX Runtime version: " << ORT_API_VERSION << std::endl;
}

// Function to check ONNX Runtime extensions
bool check_onnx_extensions() {
    std::cout << "Checking ONNX Runtime extensions..." << std::endl;
    // In C++, we can't easily check for extensions like in Python
    // We'll assume they're available and let the runtime handle errors
    return true;
}

// Simple tokenizer implementation for when tokenizers-cpp is not available
class SimpleTokenizer {
public:
    SimpleTokenizer(const std::string& model_path) {
        // Load special tokens map
        std::ifstream special_tokens_file(model_path + "/special_tokens_map.json");
        if (!special_tokens_file.is_open()) {
            throw std::runtime_error("Failed to open special_tokens_map.json in " + model_path);
        }
        json special_tokens_json;
        special_tokens_file >> special_tokens_json;
        
        // Load tokenizer config
        std::ifstream tokenizer_config_file(model_path + "/tokenizer_config.json");
        if (!tokenizer_config_file.is_open()) {
            throw std::runtime_error("Failed to open tokenizer_config.json in " + model_path);
        }
        json tokenizer_config_json;
        tokenizer_config_file >> tokenizer_config_json;
        
        // Load vocabulary
        std::ifstream vocab_file(model_path + "/tokenizer.json");
        if (!vocab_file.is_open()) {
            throw std::runtime_error("Failed to open tokenizer.json in " + model_path);
        }
        json vocab_json;
        vocab_file >> vocab_json;
        
        // Extract vocabulary from the correct path in the JSON structure
        if (vocab_json.contains("model") && vocab_json["model"].contains("vocab")) {
            for (const auto& [token, id] : vocab_json["model"]["vocab"].items()) {
                vocab_[token] = id.get<int>();
                id_to_token_[id.get<int>()] = token;
            }
        } else {
            throw std::runtime_error("Invalid tokenizer.json format: missing model.vocab");
        }
        
        // Set special tokens from config
        // First try to get from tokenizer_config.json
        if (tokenizer_config_json.contains("bos_token_id")) {
            bos_token_id_ = tokenizer_config_json["bos_token_id"].get<int>();
        } else {
            // Fallback to hardcoded value from config.json
            bos_token_id_ = 128000;
        }
        
        if (tokenizer_config_json.contains("eos_token_id")) {
            if (tokenizer_config_json["eos_token_id"].is_array()) {
                eos_token_ids_ = tokenizer_config_json["eos_token_id"].get<std::vector<int>>();
            } else {
                // If it's a single value, convert to vector
                eos_token_ids_ = {tokenizer_config_json["eos_token_id"].get<int>()};
            }
        } else {
            // Fallback to hardcoded values from config.json
            eos_token_ids_ = {128001, 128008, 128009};
        }
        
        // Store special token strings
        if (special_tokens_json.contains("bos_token") && 
            special_tokens_json["bos_token"].contains("content")) {
            bos_token_ = special_tokens_json["bos_token"]["content"].get<std::string>();
        } else {
            bos_token_ = "<|begin_of_text|>";
        }
        
        if (special_tokens_json.contains("eos_token") && 
            special_tokens_json["eos_token"].contains("content")) {
            eos_token_ = special_tokens_json["eos_token"]["content"].get<std::string>();
        } else {
            eos_token_ = "<|eot_id|>";
        }
        
        std::cout << "Tokenizer initialized with " << vocab_.size() << " tokens" << std::endl;
        std::cout << "BOS token: " << bos_token_ << " (ID: " << bos_token_id_ << ")" << std::endl;
        std::cout << "EOS token: " << eos_token_ << " (IDs: ";
        for (int id : eos_token_ids_) {
            std::cout << id << " ";
        }
        std::cout << ")" << std::endl;
    }
    
    std::vector<int> encode(const std::string& text) const {
        std::vector<int> tokens;
        
        // Add BOS token
        tokens.push_back(bos_token_id_);
        
        // Simple tokenization: split by whitespace and look up in vocabulary
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            auto it = vocab_.find(word);
            if (it != vocab_.end()) {
                tokens.push_back(it->second);
            } else {
                // For unknown tokens, try to split into characters
                for (char c : word) {
                    std::string char_token(1, c);
                    auto char_it = vocab_.find(char_token);
                    if (char_it != vocab_.end()) {
                        tokens.push_back(char_it->second);
                    } else {
                        // If character not found, use a default token ID
                        tokens.push_back(0);
                    }
                }
            }
        }
        
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens) const {
        std::string text;
        for (int token : tokens) {
            // Skip special tokens
            if (token == bos_token_id_ || 
                std::find(eos_token_ids_.begin(), eos_token_ids_.end(), token) != eos_token_ids_.end()) {
                continue;
            }
            
            auto it = id_to_token_.find(token);
            if (it != id_to_token_.end()) {
                text += it->second + " ";
            }
        }
        return text;
    }
    
private:
    std::map<std::string, int> vocab_;
    std::map<int, std::string> id_to_token_;
    int bos_token_id_;
    std::vector<int> eos_token_ids_;
    std::string bos_token_;
    std::string eos_token_;
};

// Function to load model configuration
ModelConfig load_config(const std::string& model_path) {
    ModelConfig config;
    std::ifstream config_file(fs::path(model_path) / "config.json");
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to open config.json");
    }
    
    json j;
    config_file >> j;
    
    config.num_hidden_layers = j["num_hidden_layers"];
    config.num_attention_heads = j["num_attention_heads"];
    config.num_key_value_heads = j["num_key_value_heads"];
    config.head_dim = j["head_dim"];
    config.hidden_size = j["hidden_size"];
    config.intermediate_size = j["intermediate_size"];
    config.eos_token_ids = j["eos_token_id"].get<std::vector<int>>();
    config.bos_token_id = j["bos_token_id"];
    config.attention_bias = j["attention_bias"];
    config.attention_dropout = j["attention_dropout"];
    config.max_position_embeddings = j["max_position_embeddings"];
    config.rope_theta = j["rope_theta"];
    config.hidden_act = j["hidden_act"];
    
    return config;
}

// Function to create ONNX session
Ort::Session create_session(const std::string& model_path) {
    if (!check_onnx_extensions()) {
        throw std::runtime_error("ONNX Runtime extensions not available");
    }
    
    // Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Create environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model-qa");
    
    // Create session
    std::string model_path_str = (fs::path(model_path) / "model.onnx").string();
    std::cout << "Loading model from: " << model_path_str << std::endl;
    
    try {
        Ort::Session session(env, model_path_str.c_str(), session_options);
        std::cout << "Model loaded successfully!" << std::endl;
        return session;
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to create ONNX session: " << e.what() << std::endl;
        if (std::string(e.what()).find("MatMulNBits") != std::string::npos) {
            std::cerr << "\nThe model requires quantized operations support." << std::endl;
            std::cerr << "Please make sure you have the correct ONNX Runtime version installed." << std::endl;
        }
        throw;
    }
}

// Function to initialize KV cache
std::vector<Ort::Value> initialize_kv_cache(
    Ort::Session& session,
    const ModelConfig& config,
    int batch_size = 1,
    int seq_length = 1
) {
    std::vector<Ort::Value> kv_cache;
    kv_cache.reserve(config.num_hidden_layers * 2);  // Pre-allocate space for key and value tensors
    
    // Create memory info
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator,
        OrtMemTypeDefault
    );
    
    // Initialize KV cache for each layer
    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        // Calculate tensor size for GQA
        size_t tensor_size = static_cast<size_t>(batch_size) * 
                            static_cast<size_t>(config.num_key_value_heads) * 
                            static_cast<size_t>(seq_length) * 
                            static_cast<size_t>(config.head_dim);
        
        // Create shape for GQA
        std::vector<int64_t> shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(config.num_key_value_heads),
            static_cast<int64_t>(seq_length),
            static_cast<int64_t>(config.head_dim)
        };
        
        // Initialize tensor data with zeros
        std::vector<float> tensor_data(tensor_size, 0.0f);
        
        // Create key tensor
        auto key_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            tensor_data.data(),
            tensor_data.size(),
            shape.data(),
            shape.size()
        );
        
        // Create value tensor with the same data (will be updated during inference)
        auto value_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            tensor_data.data(),
            tensor_data.size(),
            shape.data(),
            shape.size()
        );
        
        // Store tensors
        kv_cache.push_back(std::move(key_tensor));
        kv_cache.push_back(std::move(value_tensor));
    }
    
    return kv_cache;
}

// Function to generate text
template<typename TokenizerType>
std::vector<int> generate_text(
    Ort::Session& session,
    const TokenizerType& tokenizer,
    const std::string& prompt,
    const ModelConfig& config,
    int max_new_tokens = 40,
    float top_p = 0.95f,
    float temperature = 0.8f,
    float repetition_penalty = 1.0f
) {
    try {
        std::cout << "Starting text generation..." << std::endl;
        
        // Tokenize input prompt
        std::vector<int64_t> input_ids;
#ifdef USE_TOKENIZERS_CPP
        if constexpr (std::is_same_v<TokenizerType, tokenizers::Tokenizer>) {
            std::cout << "Using tokenizers-cpp for encoding..." << std::endl;
            auto encoded = const_cast<TokenizerType&>(tokenizer).Encode(prompt);
            input_ids.assign(encoded.begin(), encoded.end());
        } else {
            std::cout << "Using custom tokenizer for encoding..." << std::endl;
            auto tokens = tokenizer.encode(prompt);
            input_ids.assign(tokens.begin(), tokens.end());
        }
#else
        std::cout << "Using custom tokenizer for encoding..." << std::endl;
        auto tokens = tokenizer.encode(prompt);
        input_ids.assign(tokens.begin(), tokens.end());
#endif

        std::cout << "Input tokens: ";
        for (auto id : input_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // Add BOS token if not present
        if (input_ids.empty() || input_ids[0] != config.bos_token_id) {
            std::cout << "Adding BOS token: " << config.bos_token_id << std::endl;
            input_ids.insert(input_ids.begin(), config.bos_token_id);
        }

        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Get input and output names from the model
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input names
        size_t num_input_nodes = session.GetInputCount();
        std::vector<std::string> input_name_strings;
        std::vector<const char*> input_names;
        input_name_strings.reserve(num_input_nodes);
        input_names.reserve(num_input_nodes);
        
        std::cout << "\nGetting input names..." << std::endl;
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_name_strings.push_back(input_name.get());
            input_names.push_back(input_name_strings.back().c_str());
            std::cout << "Input " << i << ": " << input_name.get() << std::endl;
        }
        
        // Get output names
        size_t num_output_nodes = session.GetOutputCount();
        std::vector<std::string> output_name_strings;
        std::vector<const char*> output_names;
        output_name_strings.reserve(num_output_nodes);
        output_names.reserve(num_output_nodes);
        
        std::cout << "\nGetting output names..." << std::endl;
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_name_strings.push_back(output_name.get());
            output_names.push_back(output_name_strings.back().c_str());
            std::cout << "Output " << i << ": " << output_name.get() << std::endl;
        }
        
        // Initialize KV cache
        std::cout << "\nInitializing KV cache..." << std::endl;
        std::vector<Ort::Value> kv_cache = initialize_kv_cache(session, config, 1, input_ids.size());
        std::cout << "Created " << kv_cache.size() << " KV cache tensors" << std::endl;
        
        std::vector<int> output_ids;
        int current_seq_length = input_ids.size();
        
        for (int i = 0; i < max_new_tokens; ++i) {
            std::cout << "\nGenerating token " << i + 1 << "/" << max_new_tokens << std::endl;
            
            // Create input tensors
            std::vector<Ort::Value> input_tensors;
            input_tensors.reserve(num_input_nodes);  // Pre-allocate space
            
            // Create memory info
            auto memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );
            
            // Ensure input shape is reasonable
            if (input_ids.size() > config.max_position_embeddings) {
                throw std::runtime_error("Input sequence too long: " + std::to_string(input_ids.size()) + 
                                       " (max: " + std::to_string(config.max_position_embeddings) + ")");
            }
            
            // Create input_ids tensor
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
            std::cout << "Creating input tensor with shape [" << input_shape[0] << ", " << input_shape[1] << "]" << std::endl;
            
            // Create a copy of input_ids to ensure data remains valid
            std::vector<int64_t> input_ids_copy(input_ids.begin(), input_ids.end());
            
            auto input_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info,
                input_ids_copy.data(),
                input_ids_copy.size(),
                input_shape.data(),
                input_shape.size()
            );
            input_tensors.push_back(std::move(input_tensor));
            
            // Add attention_mask tensor if needed
            if (num_input_nodes > 1) {
                std::cout << "Creating attention mask tensor..." << std::endl;
                std::vector<int64_t> attention_mask(input_ids.size(), 1);
                auto attention_tensor = Ort::Value::CreateTensor<int64_t>(
                    memory_info,
                    attention_mask.data(),
                    attention_mask.size(),
                    input_shape.data(),
                    input_shape.size()
                );
                input_tensors.push_back(std::move(attention_tensor));
            }
            
            // Add KV cache tensors
            if (!kv_cache.empty()) {
                std::cout << "Adding " << kv_cache.size() << " KV cache tensors..." << std::endl;
                
                // Move KV cache tensors to input tensors
                for (auto& tensor : kv_cache) {
                    input_tensors.push_back(std::move(tensor));
                }
                kv_cache.clear();
            }
            
            // Run inference with error handling
            try {
                std::cout << "Running inference..." << std::endl;
                
                // Set run options for better error handling
                Ort::RunOptions run_options;
                run_options.SetRunLogVerbosityLevel(ORT_LOGGING_LEVEL_VERBOSE);
                
                // Verify input tensor count matches input names
                if (input_tensors.size() != input_names.size()) {
                    throw std::runtime_error("Input tensor count mismatch: expected " + 
                                           std::to_string(input_names.size()) + 
                                           ", got " + std::to_string(input_tensors.size()));
                }
                
                // Verify output names are valid
                if (output_names.empty()) {
                    throw std::runtime_error("No output names available");
                }
                
                std::cout << "Starting ONNX Runtime inference..." << std::endl;
                
                // Create a vector to hold the output tensors
                std::vector<Ort::Value> output_tensors;
                
                // Run inference with explicit error handling
                try {
                    output_tensors = session.Run(
                        run_options,
                        input_names.data(),
                        input_tensors.data(),
                        input_tensors.size(),
                        output_names.data(),
                        output_names.size()
                    );
                } catch (const Ort::Exception& e) {
                    std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
                    throw;
                }
                
                std::cout << "Inference completed successfully" << std::endl;
                
                // Verify output tensor count
                if (output_tensors.empty()) {
                    throw std::runtime_error("No output tensors received");
                }
                
                // Get next token probabilities
                auto& logits_tensor = output_tensors[0];
                auto logits_info = logits_tensor.GetTensorTypeAndShapeInfo();
                
                // Verify logits tensor shape
                auto logits_shape = logits_info.GetShape();
                if (logits_shape.size() != 3) {
                    throw std::runtime_error("Invalid logits tensor shape: expected 3 dimensions");
                }
                
                // Get logits data
                float* logits = logits_tensor.GetTensorMutableData<float>();
                int vocab_size = logits_info.GetElementCount() / (logits_shape[0] * logits_shape[1]);
                std::cout << "Vocab size: " << vocab_size << std::endl;
                
                // Apply temperature
                std::vector<float> probs(vocab_size);
                float max_logit = *std::max_element(logits, logits + vocab_size);
                float sum_exp = 0.0f;
                
                // Compute softmax with temperature
                for (int j = 0; j < vocab_size; ++j) {
                    probs[j] = std::exp((logits[j] - max_logit) / temperature);
                    sum_exp += probs[j];
                }
                
                // Normalize probabilities
                for (int j = 0; j < vocab_size; ++j) {
                    probs[j] /= sum_exp;
                }
                
                // Apply top-p sampling
                std::vector<std::pair<float, int>> sorted_probs;
                sorted_probs.reserve(vocab_size);
                
                for (int j = 0; j < vocab_size; ++j) {
                    sorted_probs.emplace_back(probs[j], j);
                }
                std::sort(sorted_probs.begin(), sorted_probs.end(), std::greater<>());
                
                float cumulative_prob = 0.0f;
                int selected_token = -1;
                
                for (const auto& [prob, token] : sorted_probs) {
                    cumulative_prob += prob;
                    if (cumulative_prob >= top_p) {
                        selected_token = token;
                        break;
                    }
                }
                
                if (selected_token == -1) {
                    selected_token = sorted_probs[0].second;
                }
                
                std::cout << "Selected token: " << selected_token << std::endl;
                
                // Add token to sequence
                input_ids.push_back(selected_token);
                output_ids.push_back(selected_token);
                current_seq_length++;
                
                // Check for EOS token
                if (std::find(
                    config.eos_token_ids.begin(),
                    config.eos_token_ids.end(),
                    selected_token
                ) != config.eos_token_ids.end()) {
                    std::cout << "EOS token found, stopping generation" << std::endl;
                    break;
                }
                
                // Update KV cache with new outputs
                if (output_tensors.size() > 1) {
                    std::cout << "Updating KV cache..." << std::endl;
                    kv_cache.reserve(output_tensors.size() - 1);
                    
                    // Each layer has a key and value tensor in the output
                    for (size_t j = 1; j < output_tensors.size(); j += 2) {
                        if (j + 1 >= output_tensors.size()) {
                            throw std::runtime_error("Invalid output tensor count for KV cache update");
                        }
                        
                        // Get tensor info
                        auto key_info = output_tensors[j].GetTensorTypeAndShapeInfo();
                        auto value_info = output_tensors[j + 1].GetTensorTypeAndShapeInfo();
                        
                        // Verify shapes match
                        auto key_shape = key_info.GetShape();
                        auto value_shape = value_info.GetShape();
                        if (key_shape != value_shape) {
                            throw std::runtime_error("Key and value tensor shapes don't match");
                        }
                        
                        // Create new tensors with the same memory info
                        auto memory_info = Ort::MemoryInfo::CreateCpu(
                            OrtArenaAllocator,
                            OrtMemTypeDefault
                        );
                        
                        // Get data pointers
                        float* key_data = output_tensors[j].GetTensorMutableData<float>();
                        float* value_data = output_tensors[j + 1].GetTensorMutableData<float>();
                        
                        // Create new tensors
                        auto new_key = Ort::Value::CreateTensor<float>(
                            memory_info,
                            key_data,
                            key_info.GetElementCount(),
                            key_shape.data(),
                            key_shape.size()
                        );
                        
                        auto new_value = Ort::Value::CreateTensor<float>(
                            memory_info,
                            value_data,
                            value_info.GetElementCount(),
                            value_shape.data(),
                            value_shape.size()
                        );
                        
                        // Add to new KV cache
                        kv_cache.push_back(std::move(new_key));
                        kv_cache.push_back(std::move(new_value));
                    }
                    
                    std::cout << "KV cache updated with " << kv_cache.size() << " tensors" << std::endl;
                }
            } catch (const Ort::Exception& e) {
                std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
                throw;
            }
        }
        
        return output_ids;
    } catch (const std::exception& e) {
        std::cerr << "Error in generate_text: " << e.what() << std::endl;
        throw;
    }
}

int main(int argc, char* argv[]) {
    try {
        cxxopts::Options options("model-qa", "Run LLaMA model inference");
        options.add_options()
            ("m,model", "Path to model directory", cxxopts::value<std::string>())
            ("p,prompt", "Input prompt", cxxopts::value<std::string>())
            ("k,max_tokens", "Maximum number of tokens to generate", cxxopts::value<int>()->default_value("40"))
            ("t,temperature", "Sampling temperature", cxxopts::value<float>()->default_value("0.8"))
            ("top-p", "Top-p sampling parameter", cxxopts::value<float>()->default_value("0.95"))
            ("r,repetition_penalty", "Repetition penalty", cxxopts::value<float>()->default_value("1.0"))
            ("h,help", "Print usage");
            
        auto result = options.parse(argc, argv);
        
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }
        
        if (!result.count("model") || !result.count("prompt")) {
            std::cerr << "Error: Model path and prompt are required" << std::endl;
            std::cout << options.help() << std::endl;
            return 1;
        }
        
        std::string model_path = result["model"].as<std::string>();
        std::string prompt = result["prompt"].as<std::string>();
        int max_new_tokens = result["max_tokens"].as<int>();
        float temperature = result["temperature"].as<float>();
        float top_p = result["top-p"].as<float>();
        float repetition_penalty = result["repetition_penalty"].as<float>();
        
        // Load model configuration
        std::cout << "Loading model configuration..." << std::endl;
        ModelConfig config = load_config(model_path);
        
        // Create ONNX session
        std::cout << "Creating ONNX session..." << std::endl;
        Ort::Session session = create_session(model_path);
        
        // Initialize tokenizer with the correct path
        std::cout << "Initializing tokenizer..." << std::endl;
        SimpleTokenizer tokenizer(model_path);
        
        // Generate text
        std::cout << "Generating text..." << std::endl;
        std::vector<int> output_tokens = generate_text(
            session,
            tokenizer,
            prompt,
            config,
            max_new_tokens,
            top_p,
            temperature,
            repetition_penalty
        );
        
        // Decode and print output
        std::string output = tokenizer.decode(output_tokens);
        std::cout << "\nGenerated text:" << std::endl;
        std::cout << output << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 