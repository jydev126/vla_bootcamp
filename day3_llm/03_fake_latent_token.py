"""
Day 3 / Project 3:
Add fake latent tokens and extract their hidden states.

核心目标：
1. 给 tokenizer 添加 <latent_1> ... <latent_4>
2. resize_token_embeddings
3. 找到 latent token ids
4. 找到 latent token 在 sequence 里的位置
5. 取这些位置的 hidden states
6. 打印 [B, num_latent, hidden_dim]
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "sshleifer/tiny-gpt2"
LATENT_TOKENS = ["<latent_1>", "<latent_2>", "<latent_3>", "<latent_4>"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 1. 添加特殊 token
    special_tokens = {
        "additional_special_tokens": LATENT_TOKENS
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    print("num_added_tokens:", num_added)
    print("len(tokenizer):", len(tokenizer))

    # 2. tokenizer 词表变大后，模型 embedding table 也必须 resize
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    # 3. 构造带 latent token 的 prompt
    # 注意：latent token 之间加空格，方便观察 tokenization
    prompt = (
        "Scene: front vehicle is slowing down.\n"
        "Reasoning tokens: <latent_1> <latent_2> <latent_3> <latent_4>\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]  # [B, T]

    print("\n=== Prompt ===")
    print(prompt)

    print("\n=== Input ids ===")
    print("input_ids shape:", input_ids.shape)
    print("input_ids:", input_ids)

    # 4. 找每个 latent token 的 token id
    latent_token_ids = tokenizer.convert_tokens_to_ids(LATENT_TOKENS)
    print("\n=== Latent token ids ===")
    for token, token_id in zip(LATENT_TOKENS, latent_token_ids):
        print(f"{token}: {token_id}")

    # 5. 打印逐 token 解码结果，帮助你理解 position
    print("\n=== Tokens by position ===")
    for pos, token_id in enumerate(input_ids[0].tolist()):
        token_text = tokenizer.decode([token_id])
        marker = "<-- latent" if token_id in latent_token_ids else ""
        print(f"pos={pos:02d}, id={token_id:05d}, text={repr(token_text)} {marker}")

    # 6. 找 latent token 在 sequence 中的位置
    latent_positions = []
    for pos, token_id in enumerate(input_ids[0].tolist()):
        if token_id in latent_token_ids:
            latent_positions.append(pos)

    print("\nlatent_positions:", latent_positions)
    print("num_latent_found:", len(latent_positions))

    assert len(latent_positions) == len(LATENT_TOKENS), (
        "没有找到全部 latent tokens。请检查 prompt 里是否写错 token，"
        "以及 tokenizer 是否正确 add_special_tokens。"
    )

    # 7. 取 hidden states
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden = outputs.hidden_states[-1]  # [B, T, hidden_dim]

    # positions 是 Python list，可以直接用于索引 T 这个维度
    latent_hidden = hidden[:, latent_positions, :]  # [B, num_latent, hidden_dim]

    print("\n=== Hidden states ===")
    print("last layer hidden shape:", hidden.shape)
    print("latent_hidden shape:", latent_hidden.shape)

    # 8. 取每个 latent token hidden state 的范数，确认它们不是同一个向量
    print("\n=== Latent hidden vector norms ===")
    norms = torch.norm(latent_hidden, dim=-1)  # [B, num_latent]
    print(norms)

    print("\n结论：")
    print("latent token 本身只是特殊 token id。")
    print("真正有用的是这些位置经过 Transformer 后得到的 hidden state。")


if __name__ == "__main__":
    main()
