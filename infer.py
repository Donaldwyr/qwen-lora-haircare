import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "./deepseek-lora-output"

# 训练时 Trainer 会把 LoRA 权重保存在 checkpoint-* 子目录里，这里自动找最新的一个；
# 如果根目录本身就有 adapter 文件（adapter_config.json 等），则直接用根目录。
if os.path.isdir(LORA_PATH):
    checkpoints = sorted(
        [d for d in os.listdir(LORA_PATH) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[-1])
    )
    if checkpoints:
        LORA_PATH = os.path.join(LORA_PATH, checkpoints[-1])
        print(f"使用 LoRA checkpoint: {LORA_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.to(device)
model.eval()


def generate_response(user_input):
    # 用 Qwen 的 chat template 构建推理提示词
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成的部分，去掉输入的 prompt
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


if __name__ == "__main__":
    print("模型已加载，开始交互式推理（输入 quit / exit 退出）。")
    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        response = generate_response(user_input)
        print("模型：", response)
