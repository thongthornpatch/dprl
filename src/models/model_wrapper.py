"""
Model Wrapper for Code Generation

Handles loading and using language models for code generation.
Supports both small models (GPT-2) for testing and larger models (CodeGen) for training.
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelWrapper:
    """
    Wrapper for language models used in RL training.

    Handles:
    - Model and tokenizer loading
    - Code generation with various sampling strategies
    - Device management (CPU/GPU)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        max_length: int = 512,
        torch_dtype: str = "float32",
    ):
        """
        Initialize model wrapper.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "Salesforce/codegen-1B-mono")
            device: Device to use ("cpu" or "cuda")
            max_length: Maximum sequence length
            torch_dtype: Torch dtype ("float32", "float16", "bfloat16")
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float32)

        print(f"Loading model: {model_name}")
        print(f"  Device: {device}")
        print(f"  Dtype: {torch_dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
        ).to(device)

        # Set to eval mode initially
        self.model.eval()

        print(f"Model loaded successfully")
        print(f"Parameters: {self.count_parameters() / 1e6:.1f}M")

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate code completions for a prompt.

        Args:
            prompt: Input prompt (problem description)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling (vs greedy)
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated code strings
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode outputs
        # Remove the input prompt from outputs
        input_length = inputs['input_ids'].shape[1]
        generated_sequences = []

        for output in outputs:
            # Decode only the generated part
            generated_text = self.tokenizer.decode(
                output[input_length:],
                skip_special_tokens=True,
            )
            generated_sequences.append(generated_text)

        return generated_sequences

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[List[str]]:
        """
        Generate code for multiple prompts.

        Args:
            prompts: List of prompts
            **kwargs: Generation parameters (passed to generate())

        Returns:
            List of lists of generated sequences (one list per prompt)
        """
        results = []
        for prompt in prompts:
            sequences = self.generate(prompt, **kwargs)
            results.append(sequences)
        return results

    def get_model(self):
        """Get the underlying model (for training)."""
        return self.model

    def get_tokenizer(self):
        """Get the tokenizer."""
        return self.tokenizer

    def train_mode(self):
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def save(self, save_path: str):
        """
        Save model and tokenizer.

        Args:
            save_path: Directory to save to
        """
        print(f"Saving model to: {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved")

    @classmethod
    def load(cls, load_path: str, device: str = "cpu"):
        """
        Load a saved model.

        Args:
            load_path: Directory to load from
            device: Device to load to

        Returns:
            ModelWrapper instance
        """
        print(f"Loading model from: {load_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(load_path)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(load_path).to(device)

        # Create wrapper
        wrapper = cls.__new__(cls)
        wrapper.model = model
        wrapper.tokenizer = tokenizer
        wrapper.device = device
        wrapper.model_name = load_path

        print(f"Model loaded")
        return wrapper


if __name__ == "__main__":
    # Test the model wrapper
    print("="*80)
    print("Testing Model Wrapper")
    print("="*80)

    # Test with GPT-2 (small model for testing)
    print("\n--- Test 1: Load GPT-2 ---")
    try:
        model = ModelWrapper(
            model_name="gpt2",
            device="cpu",
            max_length=256,
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    # Test generation
    print("\n--- Test 2: Generate Code ---")
    prompt = "def fibonacci(n):\n    # Calculate fibonacci number\n"

    try:
        generated = model.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            do_sample=True,
            num_return_sequences=2,
        )

        print(f"Generated {len(generated)} sequences:")
        for i, seq in enumerate(generated):
            print(f"\nSequence {i+1}:")
            print(seq[:100] + "..." if len(seq) > 100 else seq)

        print("\nGeneration successful")
    except Exception as e:
        print(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Test batch generation
    print("\n--- Test 3: Batch Generation ---")
    prompts = [
        "def add(a, b):\n",
        "def multiply(x, y):\n",
    ]

    try:
        batch_results = model.generate_batch(
            prompts,
            max_new_tokens=30,
            num_return_sequences=1,
        )

        print(f"Generated for {len(batch_results)} prompts:")
        for i, results in enumerate(batch_results):
            print(f"\nPrompt {i+1}: {prompts[i].strip()}")
            print(f"  Output: {results[0][:50]}...")

        print("\nBatch generation successful")
    except Exception as e:
        print(f"Batch generation failed: {e}")

    print("\n" + "="*80)
    print("All model wrapper tests completed!")
    print("="*80)
