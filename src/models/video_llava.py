"""
Video-LLaVA Model Wrapper for Vietnamese Traffic QA
Using Qwen2-VL as an alternative (better multilingual support)

Supports two modes:
- FULL MODE: Qwen2.5-VL-7B for high accuracy (requires 16GB+ GPU)
- LITE MODE: Qwen2-VL-2B for weaker machines (requires 8GB+ GPU)
"""

import torch
import torch.nn as nn
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from typing import List, Dict, Optional
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VietnameseTrafficVQAModel(nn.Module):
    """
    Video QA Model for Vietnamese Traffic
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        mode: str = "full",  # "full" or "lite"
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = False
    ):
        super().__init__()
        
        # Select model based on mode
        if mode == "lite":
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            logger.info("LITE MODE: Using Qwen2-VL-2B for weaker machines")
            lora_r = max(8, lora_r // 2)
            lora_alpha = max(16, lora_alpha // 2)
        else:
            model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            logger.info("FULL MODE: Using Qwen2.5-VL-7B for maximum accuracy")

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
                llm_int8_enable_fp32_cpu_offload=cpu_offload
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
            device_map = "balanced"
            logger.info("CPU offload enabled - using balanced device map")
        
        # Load base model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Apply LoRA if requested
        if use_lora:
            logger.info("Applying LoRA...")
            
            if load_in_4bit or load_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
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
        
        logger.info("Model loaded successfully!")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass through the model"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def prepare_inputs(
        self, 
        frames: torch.Tensor,  # (B, T, C, H, W) normalized [0, 1]
        prompts: List[str], 
        answers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model
        """
        batch_size = frames.shape[0]
        
        # Convert frames from torch tensor to list of numpy arrays for processor
        # Input: (B, T, C, H, W) in [0, 1]
        # Output: List of [List of numpy arrays (H, W, C) in [0, 255]]
        videos_list = []
        for i in range(batch_size):
            video = frames[i]  # (T, C, H, W)
            video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
            video = (video * 255.0).clamp(0, 255).byte()  # Convert to [0, 255]
            video = video.cpu().numpy()
            
            # Convert to list of frames
            video_frames = [video[j] for j in range(video.shape[0])]
            videos_list.append(video_frames)
        
        # Create messages in Qwen2-VL format
        messages_list = []
        for idx, prompt in enumerate(prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": videos_list[idx],
                            "fps": 1.0
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Add assistant response if answer is provided (for training)
            if answers is not None and answers[idx] is not None:
                messages.append({
                    "role": "assistant", 
                    "content": answers[idx]
                })
            
            messages_list.append(messages)
        
        # Apply chat template to get formatted text
        texts = [
            self.processor.apply_chat_template(
                msg, 
                tokenize=False, 
                add_generation_prompt=(answers is None)
            ) 
            for msg in messages_list
        ]
        
        # Process vision info using qwen_vl_utils
        image_inputs_list = []
        video_inputs_list = []
        for messages in messages_list:
            image_inputs, video_inputs = process_vision_info(messages)
            image_inputs_list.append(image_inputs)
            video_inputs_list.append(video_inputs)
        
        # Flatten lists (process_vision_info returns lists of lists)
        all_images = [img for sublist in image_inputs_list for img in (sublist or [])]
        all_videos = [vid for sublist in video_inputs_list for vid in (sublist or [])]
        
        # Process inputs with the processor
        inputs = self.processor(
            text=texts,
            images=all_images if all_images else None,
            videos=all_videos if all_videos else None,
            padding=True,
            return_tensors="pt"
        )
        
        # Create labels for training
        if answers is not None:
            labels = inputs["input_ids"].clone()
            
            # Mask padding tokens
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            # Find assistant response start and mask everything before it
            for i in range(batch_size):
                # Find where assistant response starts
                # Look for the pattern after the last user message
                input_text = self.processor.tokenizer.decode(inputs["input_ids"][i])
                
                # Find the position where assistant starts responding
                assistant_start = input_text.find("assistant\n")
                if assistant_start != -1:
                    # Tokenize up to assistant start
                    prefix_ids = self.processor.tokenizer.encode(
                        input_text[:assistant_start + len("assistant\n")],
                        add_special_tokens=False
                    )
                    # Mask everything up to assistant response
                    labels[i, :len(prefix_ids)] = -100
            
            inputs["labels"] = labels
        
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
        Generate responses for given frames and prompts
        """
        inputs = self.prepare_inputs(frames, prompts, answers=None)
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
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
        
        # Decode only the generated part (skip input)
        generated_texts = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return generated_texts
    
    def save_pretrained(self, output_dir: str):
        """Save model and processor"""
        logger.info(f"Saving model to {output_dir}")
        
        if self.use_lora:
            # Save only LoRA adapters
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
        
        model = cls(
            model_name=model_path,
            use_lora=False,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        
        return model

def extract_answer_letter(generated_text: str, choices: List[str]) -> str:
    """Extract answer letter from generated text"""
    text = generated_text.strip().upper()
    
    # Check if starts with A, B, C, or D
    for letter in ['A', 'B', 'C', 'D']:
        if text.startswith(letter):
            return letter
    
    # Check if any choice text appears in generated text
    for i, choice in enumerate(choices):
        letter = chr(65 + i)  # A=65, B=66, C=67, D=68
        choice_text = choice.split('.', 1)[-1].strip()
        if choice_text.lower() in generated_text.lower():
            return letter
    
    logger.warning(f"Could not extract answer from: {generated_text}")
    return 'A'  # Default to A