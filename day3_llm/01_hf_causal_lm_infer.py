"""
Day 3 / Project 1:
Run a tiny HuggingFace causal language model.

核心目标：
1. tokenizer(prompt)
2. 打印 input_ids / attention_mask shape
3. model(**inputs)
4. 打印 logits shape
5. 用 labels 计算 causal LM loss
6. model.generate(...)
7. decode 输出
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "sshleifer/tiny-gpt2"


def main():
    # 1. 选择设备：有 CUDA 就用 CUDA，否则 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # 2. tokenizer：负责 文本 <-> token id
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT2 类模型通常没有 pad_token。为了 generate 不报警，临时用 eos_token 当 pad_token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. model：AutoModelForCausalLM = Transformer backbone + LM head
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    prompt = "Scene: front vehicle is slowing down. Decision:"

    # 4. tokenizer(prompt)：把字符串变成 input_ids / attention_mask
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("\n=== Tokenized inputs ===")
    print("input_ids:", inputs["input_ids"])
    print("attention_mask:", inputs["attention_mask"])
    print("input_ids shape:", inputs["input_ids"].shape)          # [B, T]
    print("attention_mask shape:", inputs["attention_mask"].shape)  # [B, T]

    # 5. forward：一次前向计算，输出 logits
    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    print("\n=== Forward outputs ===")
    print("logits shape:", logits.shape)  # [B, T, vocab_size]
    print("vocab_size:", tokenizer.vocab_size)

    # 6. 看最后一个位置的 logits
    # 生成下一个 token 时，真正用的是最后一个位置的 logits
    next_token_logits = logits[:, -1, :]  # [B, vocab_size]
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    print("\n=== Next token by greedy argmax ===")
    print("next_token_id:", next_token_id)
    print("next_token:", tokenizer.decode(next_token_id))

    # 7. labels：训练 causal LM 时用
    # 对 AutoModelForCausalLM 来说，labels 通常就是 input_ids 的拷贝。
    # 模型内部会做 shift：第 i 个位置预测第 i+1 个 token。
    with torch.inference_mode():
        outputs_with_loss = model(**inputs, labels=inputs["input_ids"])

    print("\n=== Causal LM loss demo ===")
    print("loss:", outputs_with_loss.loss.item())

    # 8. generate：自回归生成。它会循环调用模型，每次生成一个新 token。
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,  # greedy decoding，方便复现实验
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n=== Generate ===")
    print("generated_ids shape:", generated_ids.shape)  # [B, T + new_tokens]
    print("generated_text:")
    print(generated_text)


if __name__ == "__main__":
    main()
