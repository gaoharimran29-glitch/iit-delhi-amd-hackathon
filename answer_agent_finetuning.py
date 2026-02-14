import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

MODEL_ID = "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots/your_snapshot"
DATASET_PATH = "a_agent_dataset.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_a_agent(examples):
    inputs = [
        f"Solve this MCQ:\n{examples['input'][i]}\nReturn result in strict JSON format:"
        for i in range(len(examples["input"]))
    ]

    outputs = [
        f'{{"answer": "{examples["output"][i]}", "reasoning": "Logical deduction based on given conditions."}}'
        for i in range(len(examples["output"]))
    ]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(preprocess_a_agent, batched=True, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./a_agent_adapter",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,   # ✅ Safe for 200 samples
    learning_rate=2e-4,
    bf16=True,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
model.save_pretrained("./a_agent_adapter")
print("✅ A-Agent adapter saved")
