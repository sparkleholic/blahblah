#!/usr/bin/env python3

import argparse
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import sys
from transformers import AutoTokenizer


def check_onnx_extensions():
    """Check if required ONNX Runtime extensions are available."""
    try:
        # Try to import the extensions
        import onnxruntime_extensions
        print(f"ONNX Runtime Extensions version: "
              f"{onnxruntime_extensions.__version__}")
        return True
    except ImportError:
        print("Error: Required ONNX Runtime extensions not found.")
        print("Please install them using:")
        print("pip install onnxruntime-extensions==0.12.0")
        return False


def load_config(model_path):
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def create_session(model_path):
    if not check_onnx_extensions():
        sys.exit(1)
        
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session_options.intra_op_num_threads = 1
    
    # Try to use CPU provider with specific settings
    providers = [
        ("CPUExecutionProvider", {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cpu_memory_arena_cfg": {
                "arena_extend_strategy": "kNextPowerOfTwo",
                "initial_chunk_size_bytes": 2 * 1024 * 1024,
                "max_dead_bytes_per_chunk": 2 * 1024 * 1024,
                "initial_arena_size_bytes": 2 * 1024 * 1024,
                "max_arena_size_bytes": 2 * 1024 * 1024 * 1024,
            }
        })
    ]
    
    model_path_str = str(Path(model_path) / "model.onnx")
    print(f"Loading model from: {model_path_str}")
    
    try:
        session = ort.InferenceSession(
            model_path_str,
            session_options,
            providers=providers
        )
        print("Model loaded successfully!")
        print("Available providers:", ort.get_available_providers())
        print("Session providers:", session.get_providers())
        return session
    except ort.capi.onnxruntime_pybind11_state.Fail as e:
        error_msg = str(e)
        print(f"Error loading model: {error_msg}")
        if "MatMulNBits" in error_msg:
            print("\nThe model requires quantized operations support.")
            print("Please try installing a specific version of "
                  "onnxruntime-extensions:")
            print("pip install onnxruntime-extensions==0.12.0")
            print("\nIf the issue persists, you might need to:")
            print("1. Install a newer version of ONNX Runtime:")
            print("   pip install --upgrade onnxruntime")
            print("2. Install the CPU-specific version:")
            print("   pip install onnxruntime-cpu")
            print("3. Make sure you have the latest onnxruntime-extensions:")
            print("   pip install --upgrade onnxruntime-extensions")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)


def initialize_kv_cache(session, batch_size=1, seq_length=1, num_layers=16):
    """Initialize the key-value cache for all layers."""
    kv_cache = {}
    num_key_value_heads = 8  # From config.json
    head_dim = 64  # From config.json
    
    for layer_idx in range(num_layers):
        # Initialize key cache
        kv_cache[f'past_key_values.{layer_idx}.key'] = np.zeros(
            (batch_size, num_key_value_heads, seq_length, head_dim),
            dtype=np.float32
        )
        # Initialize value cache
        kv_cache[f'past_key_values.{layer_idx}.value'] = np.zeros(
            (batch_size, num_key_value_heads, seq_length, head_dim),
            dtype=np.float32
        )
    
    return kv_cache


def generate_text(
    session,
    tokenizer,
    prompt,
    max_new_tokens=40,
    top_p=0.95,
    temperature=0.8,
    repetition_penalty=1.0
):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    attention_mask = np.ones_like(input_ids)
    
    # Initialize KV cache
    kv_cache = initialize_kv_cache(session)
    
    for _ in range(max_new_tokens):
        # Prepare input feed
        input_feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kv_cache  # Add all KV cache tensors
        }
        
        # Run inference
        outputs = session.run(None, input_feed)
        
        # Get next token probabilities
        next_token_logits = outputs[0][0, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply top-p sampling
        sorted_logits, sorted_indices = (
            np.sort(next_token_logits)[::-1],
            np.argsort(next_token_logits)[::-1]
        )
        probs = np.exp(sorted_logits) / np.sum(np.exp(sorted_logits))
        cumulative_probs = np.cumsum(probs)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample next token
        probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
        next_token = np.random.choice(len(probs), p=probs)
        
        # Add token to sequence
        input_ids = np.concatenate(
            [input_ids, np.array([[next_token]])],
            axis=1
        )
        attention_mask = np.concatenate(
            [attention_mask, np.array([[1]])],
            axis=1
        )
        
        # Update KV cache with new outputs
        # Note: This is a simplified version. In a real implementation,
        # you would need to properly update the KV cache based on the
        # model's output format
        for layer_idx in range(16):  # num_layers from config
            kv_cache[f'past_key_values.{layer_idx}.key'] = outputs[
                layer_idx * 2 + 1
            ]
            kv_cache[f'past_key_values.{layer_idx}.value'] = outputs[
                layer_idx * 2 + 2
            ]
        
        # Check for EOS token
        eos_tokens = [128001, 128008, 128009]  # EOS tokens from config
        if next_token in eos_tokens:
            break
    
    return input_ids[0].tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Run Llama model with ONNX Runtime"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Path to the model directory"
    )
    parser.add_argument(
        "-k",
        "--max_new_tokens",
        type=int,
        default=40,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "-p",
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "-r",
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="<|user|>Tell me a joke<|end|><|assistant|>",
        help="Input prompt for the model"
    )
    
    args = parser.parse_args()
    
    # Print ONNX Runtime version
    print(f"ONNX Runtime version: {ort.__version__}")
    
    # Load model and create session
    session = create_session(args.model_path)
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        sys.exit(1)
    
    # Generate text
    output_ids = generate_text(
        session,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty
    )
    
    # Convert token IDs to text
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main() 