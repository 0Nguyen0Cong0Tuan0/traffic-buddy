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
    - LITE: Qwen2-VL-2B-Instruct (good accuracy, requires 8GB+ GPU)
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
        cpu_offload: bool = False  # Enable CPU offload for weak machines
    ):
        super().__init__()
        
        # Select model based on mode
        if mode == "lite":
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
            logger.info("🔧 LITE MODE: Using Qwen2-VL-2B for weaker machines")
            # Adjust LoRA for smaller model
            lora_r = max(8, lora_r // 2)
            lora_alpha = max(16, lora_alpha // 2)
        else:
            if "2.5" not in model_name:
                model_name = "Qwen/Qwen2-VL-7B-Instruct"
            logger.info("🚀 FULL MODE: Using Qwen2-VL-7B for maximum accuracy")
        
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
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            logger.error(f"Failed to load model with device_map={device_map}")
            logger.info("Retrying with CPU offload enabled...")
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="balanced",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            logger.info("✓ Successfully loaded with CPU offload")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Debug: Print tokenizer special tokens
        logger.info(f"Tokenizer special tokens: {self.processor.tokenizer.special_tokens_map}")
        logger.info(f"Tokenizer vocab size: {len(self.processor.tokenizer)}")
        
        # Ensure vision tokens exist
        vision_tokens = ['<|vision_start|>', '<|vision_end|>']
        for token in vision_tokens:
            token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
            logger.info(f"Token '{token}' ID: {token_id}")
            if token_id == self.processor.tokenizer.unk_token_id:
                logger.error(f"Token '{token}' not found in tokenizer vocabulary!")
        
        # Add vision tokens if not present
        new_tokens = [t for t in vision_tokens if t not in self.processor.tokenizer.get_vocab()]
        if new_tokens:
            logger.info(f"Adding new tokens to tokenizer: {new_tokens}")
            self.processor.tokenizer.add_tokens(new_tokens)
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
        
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
            
            # Debug: Check trainable parameters
            trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
            logger.info(f"Trainable parameters: {trainable_params}")
            if not trainable_params:
                logger.error("No trainable parameters found! Check LoRA configuration.")
        
        # Enable gradient checkpointing for memory efficiency
        # self.model.gradient_checkpointing_enable()  # Disabled for debugging
        logger.info("Gradient checkpointing disabled for debugging")
        
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
        actual_pixel_values = None
        actual_grid_thw = None
        
        if pixel_values_videos is not None:
            actual_pixel_values = pixel_values_videos
            actual_grid_thw = video_grid_thw
        elif pixel_values is not None:
            actual_pixel_values = pixel_values
            actual_grid_thw = image_grid_thw
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=actual_pixel_values,
            image_grid_thw=actual_grid_thw,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def prepare_inputs(
        self,
        frames: torch.Tensor,  # (B, T, C, H, W)
        prompts: List[str],
        answers: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = frames.shape[0]
        
        videos_list = []
        for i in range(batch_size):
            video = frames[i]
            video = video.permute(0, 2, 3, 1)
            video = video.cpu().numpy()
            if video.max() <= 1.0:
                video = (video * 255).astype('uint8')
            videos_list.append(video)
        
        video_inputs = self.processor.image_processor(
            images=videos_list,
            return_tensors="pt"
        )
        logger.info(f"Video inputs shape: {video_inputs['pixel_values'].shape}")
        
        num_frames = frames.shape[1]
        
        texts = []
        for prompt, answer in zip(prompts, answers or [None] * batch_size):
            system_message = "Bạn là một trợ lý AI chuyên về luật giao thông Việt Nam."
            vision_tokens = "<|vision_start|><|vision_end|>" * num_frames
            user_message = f"{vision_tokens}\n{prompt}"
            
            text = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
            
            logger.info(f"Raw text: {text}")
            
            if answer is not None:
                text += answer + "<|im_end|>"
            
            texts.append(text)
        
        text_inputs = self.processor.tokenizer(
            texts,
            padding=True,
            return_tensors="pt"
        )
        
        logger.info(f"Tokenized input_ids: {text_inputs['input_ids']}")
        logger.info(f"Tokenized input_ids shape: {text_inputs['input_ids'].shape}")
        
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        for i, input_ids in enumerate(text_inputs['input_ids']):
            vision_start_count = (input_ids == vision_start_id).sum().item()
            vision_end_count = (input_ids == vision_end_id).sum().item()
            logger.info(f"Sample {i}: vision_start tokens: {vision_start_count}, vision_end tokens: {vision_end_count}")
        
        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "pixel_values_videos": video_inputs["pixel_values"],
            "video_grid_thw": video_inputs["image_grid_thw"]
        }
        
        if answers is not None and answers[0] is not None:
            labels = inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
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
        
        generated_texts = self.processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_texts
    
    def save_pretrained(self, output_dir: str):
        logger.info(f"Saving model to {output_dir}")
        
        if self.use_lora:
            self.model.save_pretrained(output_dir)
        else:
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
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        model = cls(
            model_name=model_path,
            use_lora=False,
            device_map=device_map,
            torch_dtype=torch_dtype
        )
        
        return model

def extract_answer_letter(generated_text: str, choices: List[str]) -> str:
    text = generated_text.strip().upper()
    
    for letter in ['A', 'B', 'C', 'D']:
        if text.startswith(letter):
            return letter
    
    for i, choice in enumerate(choices):
        letter = chr(65 + i)
        choice_text = choice.split('.', 1)[-1].strip()
        if choice_text.lower() in generated_text.lower():
            return letter
    
    logger.warning(f"Could not extract answer from: {generated_text}")
    return 'A'