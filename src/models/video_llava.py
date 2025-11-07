"""
Video-LLaVA Model Wrapper for Vietnamese Traffic QA
Using Qwen2-VL as an alternative (better multilingual support)

Supports two modes:
- FULL MODE: Qwen2.5-VL-7B for high accuracy (requires 16GB+ GPU)
- LITE MODE: Qwen2.5-VL-3B for weaker machines (requires 8GB+ GPU)
"""

import torch
import torch.nn as nn
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VietnameseTrafficVQAModel(nn.Module):
    """
    Video QA Model for Vietnamese Traffic
    
    Supports two modes:
    - FULL: Qwen2.5-VL-7B-Instruct (high accuracy, requires 16GB+ GPU)
    - LITE: Qwen2.5-VL-3B-Instruct (good accuracy, requires 8GB+ GPU)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        mode: str = "full",  # "full" or "lite"
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = False  # Enable CPU offload for weak machines
    ):
        super().__init__()
        
        # Select model based on mode
        if mode == "lite":
            model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
            logger.info("LITE MODE: Using Qwen2.5-VL-3B for weaker machines")
            
            # Adjust LoRA for smaller model
            lora_r = max(8, lora_r // 2)
            lora_alpha = max(16, lora_alpha // 2)
        else:
            model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            logger.info("FULL MODE: Using Qwen2-VL-7B for maximum accuracy")
        
        self.model_name = model_name
        self.mode = mode
        self.use_lora = use_lora
        
        logger.info(f"Loading model: {model_name}")
        
        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=cpu_offload  # Enable CPU offload if needed
            )
            logger.info("Using 4-bit quantization")
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=cpu_offload
            )
            logger.info("Using 8-bit quantization")
        
        # Adjust device_map for weak machines
        if cpu_offload:
            device_map = "balanced"  # More balanced GPU/CPU distribution
            logger.info("CPU offload enabled - using balanced device map")
        
        # Load base model
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True  # Reduce CPU memory usage
            )
        except Exception as e:
            logger.error(f"Failed to load model with device_map={device_map}")
            logger.info("Retrying with CPU offload enabled...")
            
            # Retry with CPU offload
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,  # Use 8-bit as fallback
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="balanced",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("Successfully loaded with CPU offload")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Apply LoRA if requested
        if use_lora:
            logger.info("Applying LoRA...")
            
            # Prepare model for k-bit training if quantized
            if load_in_4bit or load_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # LoRA configuration
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        logger.info("Model loaded successfully!")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )
        
        return outputs
    
    def prepare_inputs(
        self,
        frames: torch.Tensor,  # (B, T, C, H, W)
        prompts: List[str],
        answers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model
        """
        batch_size = frames.shape[0]
        
        # Convert frames to list of numpy arrays for processor
        # Qwen2-VL expects videos as List[np.ndarray] where each video is (T, H, W, C)
        videos_list = []
        for i in range(batch_size):
            video = frames[i]  # (T, C, H, W)
            video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
            video = video.cpu().numpy()
            # Processor expects values in [0, 255]
            if video.max() <= 1.0:
                video = (video * 255).astype('uint8')
            videos_list.append(video)
        
        # Format prompts with video tokens
        formatted_prompts = []
        for prompt, answer in zip(prompts, answers or [None] * batch_size):
            text = f"<|im_start|>system\nBạn là một trợ lý AI chuyên về luật giao thông Việt Nam.<|im_end|>\n"
            text += f"<|im_start|>user\n<video>\n{prompt}<|im_end|>\n"
            text += "<|im_start|>assistant\n"
            
            if answer is not None:
                text += f"{answer}<|im_end|>"
            
            formatted_prompts.append(text)
        
        # Process inputs
        inputs = self.processor(
            text=formatted_prompts,
            videos=videos_list,
            padding=True,
            return_tensors="pt"
        )
        
        return inputs
    
    def generate(
        self,
        frames: torch.Tensor,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False
    ) -> List[str]:
        """
        Generate answers
        """
        # Prepare inputs without answers (for generation)
        batch_size = frames.shape[0]
        
        # Convert frames to list of numpy arrays
        videos_list = []
        for i in range(batch_size):
            video = frames[i]  # (T, C, H, W)
            video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
            video = video.cpu().numpy()
            # Ensure values in [0, 255]
            if video.max() <= 1.0:
                video = (video * 255).astype('uint8')
            videos_list.append(video)
        
        # Format prompts
        formatted_prompts = []
        for prompt in prompts:
            text = f"<|im_start|>system\nBạn là một trợ lý AI chuyên về luật giao thông Việt Nam.<|im_end|>\n"
            text += f"<|im_start|>user\n<video>\n{prompt}<|im_end|>\n"
            text += "<|im_start|>assistant\n"
            formatted_prompts.append(text)
        
        # Process inputs
        inputs = self.processor(
            text=formatted_prompts,
            videos=videos_list,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode
        generated_texts = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_texts
    
    def save_pretrained(self, output_dir: str):
        """Save model"""
        logger.info(f"Saving model to {output_dir}")
        
        if self.use_lora:
            # Save LoRA adapters only
            self.model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)
        
        self.processor.save_pretrained(output_dir)
        logger.info("Model saved successfully!")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """Load fine-tuned model"""
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        # This will be implemented after training
        # For now, load base model
        model = cls(
            model_name=model_path,
            use_lora=False,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        
        return model


def extract_answer_letter(generated_text: str, choices: List[str]) -> str:
    """
    Extract answer letter (A, B, C, D) from generated text
    """
    text = generated_text.strip().upper()
    
    # Direct letter match
    for letter in ['A', 'B', 'C', 'D']:
        if text.startswith(letter):
            return letter
    
    # Search in text
    for i, choice in enumerate(choices):
        letter = chr(65 + i)  # A, B, C, D
        choice_text = choice.split('.', 1)[-1].strip()
        if choice_text.lower() in generated_text.lower():
            return letter
    
    # Default to A if no match
    logger.warning(f"Could not extract answer from: {generated_text}")
    return 'A'