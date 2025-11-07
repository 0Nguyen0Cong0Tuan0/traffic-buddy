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
        
        # Convert frames to list of numpy arrays for processor
        videos_list = []
        for i in range(batch_size):
            video = frames[i]  # (T, C, H, W)
            video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
            video = video.cpu().numpy()
            
            # Ensure values are in [0, 255] range
            if video.max() <= 1.0:
                video = (video * 255).astype('uint8')
            else:
                video = video.astype('uint8')
                
            # Convert to list of frames
            video_frames = [video[j] for j in range(video.shape[0])]
            videos_list.append(video_frames)
        
        messages_list = []
        for idx, (prompt, answer) in enumerate(zip(prompts, answers or [None] * batch_size)):
            messages = [
                {
                    "role": "system", 
                    "content": "Bạn là một trợ lý AI chuyên về luật giao thông Việt Nam."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": videos_list[idx],  # Use the actual video for this sample
                            "fps": 1.0  # Add fps parameter
                        },
                        {
                            "type": "text", 
                            "text": prompt
                        }
                    ]
                }
            ]
            
            # Add assistant response if answer is provided (for training)
            if answer is not None:
                messages.append({
                    "role": "assistant", 
                    "content": answer
                })
            
            messages_list.append(messages)
        
        # Apply chat template to get formatted text
        texts = [
            self.processor.apply_chat_template(
                msg, 
                tokenize=False, 
                add_generation_prompt=(answers is None or answers[0] is None)
            ) 
            for msg in messages_list
        ]
        
        logger.info(f"Formatted text example (first 500 chars): {texts[0][:500]}")
        
        # Process inputs with the processor
        # Important: Pass videos in the correct format
        inputs = self.processor(
            text=texts,
            videos=videos_list,
            padding=True,
            return_tensors="pt"
        )
        
        logger.info(f"Processor output keys: {inputs.keys()}")
        logger.info(f"Input_ids shape: {inputs['input_ids'].shape}")
        
        if 'pixel_values_videos' in inputs:
            logger.info(f"Pixel values videos shape: {inputs['pixel_values_videos'].shape}")
        if 'video_grid_thw' in inputs:
            logger.info(f"Video grid thw: {inputs['video_grid_thw']}")
        
        # Verify vision tokens are present
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        video_pad_id = self.processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')
        
        for i, input_ids in enumerate(inputs['input_ids']):
            vision_start_count = (input_ids == vision_start_id).sum().item()
            vision_end_count = (input_ids == vision_end_id).sum().item()
            video_pad_count = (input_ids == video_pad_id).sum().item()
            logger.info(
                f"Sample {i}: vision_start={vision_start_count}, "
                f"vision_end={vision_end_count}, video_pad={video_pad_count}"
            )
            
            if vision_start_count == 0:
                logger.error(f"WARNING: No vision tokens found in sample {i}!")
                logger.error(f"Input IDs: {input_ids[:50]}")
                logger.error(f"Decoded text: {self.processor.tokenizer.decode(input_ids[:100])}")
        
        # Create labels for training
        if answers is not None and answers[0] is not None:
            labels = inputs["input_ids"].clone()
            
            # Mask padding tokens
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            # Find assistant response start and mask everything before it
            for i in range(batch_size):
                # Find the last <|im_start|> which marks the assistant's turn
                im_start_id = self.processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
                assistant_positions = (inputs["input_ids"][i] == im_start_id).nonzero(as_tuple=True)[0]
                
                if len(assistant_positions) > 0:
                    # Mask everything up to and including the last <|im_start|>
                    last_im_start = assistant_positions[-1].item()
                    labels[i, :last_im_start + 1] = -100
                    
                    # Also need to find "assistant\n" text after <|im_start|>
                    # and mask it too
                    assistant_text_ids = self.processor.tokenizer.encode(
                        "assistant\n", 
                        add_special_tokens=False
                    )
                    # Mask a few more tokens to skip "assistant\n"
                    labels[i, :last_im_start + len(assistant_text_ids) + 1] = -100
            
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