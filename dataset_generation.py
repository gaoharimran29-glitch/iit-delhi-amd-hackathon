# generate dataset 
import os
import json
import torch
import re
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- 1. SETTINGS & PATHS ---
MODEL_PATH = "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

WRITABLE_DIR = os.path.join(os.getcwd(), "safe_work")
os.makedirs(WRITABLE_DIR, exist_ok=True)
os.environ['HF_HOME'] = WRITABLE_DIR
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# --- 2. LOAD MODEL ---
print("⏳ Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

TOPICS = [
    "Syllogisms",
    "Seating Arrangements (Circular and Linear)",
    "Blood Relations and Family Tree",
    "Mixed Series (Alphanumeric)"
]

# --- Helpers ---
def extract_first_json(text: str):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    json_str = match.group(0)
    json_str = re.sub(r"```json|```", "", json_str).strip()
    return json_str

def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

def validate_q_item(item):
    if not all(k in item for k in ["topic", "question", "choices", "answer", "explanation"]):
        return False
    if not isinstance(item["choices"], list) or len(item["choices"]) != 4:
        return False
    if item["answer"] not in ["A", "B", "C", "D"]:
        return False
    return True

# --- 3. GENERATION ---
def generate_datasets(num_questions=100):
    start_total = time.time()
    q_data, a_data = [], []

    total_bar = tqdm(total=num_questions, desc="Total Attempts", position=0)
    valid_bar = tqdm(total=num_questions, desc="Valid Q-Agent Samples", position=1)

    for i in range(num_questions):
        topic = TOPICS[i % len(TOPICS)]

        q_prompt = f"""
You are an expert competitive-exam question setter for an AI-vs-AI tournament.

Return ONLY a single valid JSON object. Do not add any text before or after JSON.

JSON schema:
{{
  "topic": "{topic}",
  "question": "Question text",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "A/B/C/D",
  "explanation": "Concise logical explanation"
}}

Rules:
- Seating arrangement questions must be purely logical (no numeric/permutation counting).
- Exactly one correct answer.
- English only.
- Topic + question + choices + answer must be concise (≤150 tokens total).
- Explanation under 120 words.
"""

        total_bar.update(1)

        q_out = pipe(
            q_prompt,
            max_new_tokens=300,
            do_sample=False,          # more reliable JSON
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )

        full_text = q_out[0]["generated_text"]
        json_blob = extract_first_json(full_text)
        if json_blob is None:
            continue

        item = try_parse_json(json_blob)
        if item is None or not validate_q_item(item):
            continue

        q_data.append(item)
        valid_bar.update(1)

        # --- A-Agent ---
        a_prompt = f"""
You are an expert logical reasoning solver.

Return ONLY a single valid JSON object. Do not add any text before or after JSON.

JSON schema:
{{
  "answer": "A/B/C/D",
  "reasoning": "Short logical reasoning"
}}

Question: {item['question']}
Choices: {item['choices']}
"""

        a_out = pipe(
            a_prompt,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

        a_full_text = a_out[0]["generated_text"]
        a_json_blob = extract_first_json(a_full_text)
        if a_json_blob is None:
            continue

        ans_json = try_parse_json(a_json_blob)
        if not ans_json or "answer" not in ans_json:
            continue

        if ans_json.get("answer", "").upper() == item["answer"].upper():
            a_data.append({
                "instruction": f"Solve this {item['topic']} MCQ.",
                "input": f"Question: {item['question']}\nChoices: {item['choices']}",
                "output": ans_json["answer"]
            })

    total_bar.close()
    valid_bar.close()

    with open("q_agent_dataset.json", "w") as f:
        json.dump(q_data, f, indent=2)
    with open("a_agent_dataset.json", "w") as f:
        json.dump(a_data, f, indent=2)

    print("\n📊 Final Report:")
    print(f"Q-Agent samples: {len(q_data)} ({len(q_data)/num_questions*100:.2f}%)")
    print(f"A-Agent samples: {len(a_data)} ({len(a_data)/num_questions*100:.2f}%)")
    print(f"⏱️ Total time: {time.time() - start_total:.2f} seconds")

if __name__ == "__main__":
    generate_datasets(num_questions=200)

