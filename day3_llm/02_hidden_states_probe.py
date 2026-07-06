"""
Day 3 / Project 2:
Probe hidden states from a HuggingFace causal language model.

核心目标：
1. output_hidden_states=True
2. 打印 hidden_states 层数
3. 打印最后一层 hidden state shape
4. 取最后一个 token 的 hidden state
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "sshleifer/tiny-gpt2"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    prompt = "Scene: front vehicle is slowing down. Decision:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    print("\n=== Basic shapes ===")
    print("input_ids shape:", inputs["input_ids"].shape)  # [B, T]
    print("logits shape:", outputs.logits.shape)          # [B, T, vocab_size]

    # hidden_states 是一个 tuple：
    # hidden_states[0] 通常是 embedding 输出
    # hidden_states[1] 是第 1 层 Transformer block 输出
    # ...
    # hidden_states[-1] 是最后一层输出
    print("\n=== Hidden states ===")
    print("len(outputs.hidden_states):", len(outputs.hidden_states))

    for i, h in enumerate(outputs.hidden_states):
        print(f"hidden_states[{i}] shape:", h.shape)  # [B, T, hidden_dim]

    last_layer_hidden = outputs.hidden_states[-1]
    print("\nlast_layer_hidden shape:", last_layer_hidden.shape)

    # 取最后一个 token 的 hidden state
    last_token_hidden = last_layer_hidden[:, -1, :]  # [B, hidden_dim]
    print("last_token_hidden shape:", last_token_hidden.shape)
    print("last_token_hidden:", last_token_hidden)

    # 辅助理解：最后一个 token 是什么？
    last_token_id = inputs["input_ids"][0, -1].item()
    print("\nlast token id:", last_token_id)
    print("last token text:", repr(tokenizer.decode([last_token_id])))


if __name__ == "__main__":
    main()
