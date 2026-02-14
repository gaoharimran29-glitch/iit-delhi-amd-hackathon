## q agent fine tuning

import os

# Create writable cache directory
os.makedirs("./hf_cache", exist_ok=True)

# Set HuggingFace cache path
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_DATASETS_CACHE"] = "./hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache/transformers"

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    max_seq_length = 1024,
    load_in_4bit = True, # Efficiency on MI300X
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
)

# 3. Formatting Function for Q-Agent
def formatting_prompts_func(examples):
    instructions = [f"Generate a tricky MCQ on {t}" for t in examples["topic"]]
    outputs = []
    for i in range(len(examples["topic"])):
        res = {
            "topic": examples["topic"][i],
            "question": examples["question"][i],
            "choices": examples["choices"][i],
            "answer": examples["answer"][i],
            "explanation": examples["explanation"][i]
        }
        outputs.append(str(res))
    
    texts = []
    for ins, out in zip(instructions, outputs):
        texts.append(f"### Instruction:\n{ins}\n\n### Response:\n{out}")
    return { "text" : texts, }

dataset = load_dataset("json", data_files="q_agent_dataset.json", split="train" , cache_dir="./hf_cache")
dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,  # 🔥 ADD THIS LINE
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 50,
        learning_rate = 2e-4,
        bf16 = True,
        logging_steps = 1,
        output_dir = "q_agent_lora",
        save_strategy="no",   # 🔥 prevents checkpoint save
        report_to="none", 
    ),
    packing="False"
)


trainer.train()
model.save_pretrained("q_agent_unsloth")
tokenizer.save_pretrained("q_agent_unsloth")
print("✅ Q-Agent Fine-tuned!")

## checker 

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "q_agent_unsloth",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = """### Instruction:
Ask clarifying questions for this user query:
"I want to build an AI startup"

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
