"""
Day 3 / Project 4:
Roughly compare prefill/decode behavior with generate timing.

注意：
这个实验不是严格 benchmark。
tiny-gpt2 太小，CPU/GPU 调度、首次加载、缓存都会影响时间。
今天目的只是建立直觉：

长输入 mainly affects prefill.
长输出 mainly affects decode.
显式长 CoT 如果逐 token 生成，会增加 decode 步数。
latent token 作为输入时，主要进入 prefill 阶段。
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "sshleifer/tiny-gpt2"
LATENT_TOKENS = ["<latent_1>", "<latent_2>", "<latent_3>", "<latent_4>"]


def sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_generate(model, tokenizer, prompt, max_new_tokens, device, title):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_token_count = inputs["input_ids"].shape[1]

    sync_if_cuda()
    start = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    sync_if_cuda()
    end = time.perf_counter()

    elapsed = end - start
    total_token_count = generated_ids.shape[1]
    new_token_count = total_token_count - input_token_count

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    print(f"\n=== {title} ===")
    print("input_token_count:", input_token_count)
    print("max_new_tokens:", max_new_tokens)
    print("actual_new_tokens:", new_token_count)
    print("elapsed_seconds:", f"{elapsed:.6f}")
    print("generated_text:")
    print(generated_text)

    return {
        "title": title,
        "input_token_count": input_token_count,
        "max_new_tokens": max_new_tokens,
        "actual_new_tokens": new_token_count,
        "elapsed_seconds": elapsed,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 添加 latent token
    tokenizer.add_special_tokens({
        "additional_special_tokens": LATENT_TOKENS
    })
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()

    # warmup：第一次 generate 常有额外开销，所以先跑一次不计入比较
    warmup_prompt = "Warmup:"
    _ = timed_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=warmup_prompt,
        max_new_tokens=5,
        device=device,
        title="warmup, ignore this result",
    )

    # A：短 prompt + 长 decode
    prompt_a = (
        "Scene: front vehicle is slowing down.\n"
        "Think step by step and explain the driving decision in detail.\n"
        "Reasoning:"
    )

    # B：较长 prompt + latent tokens + 短 decode
    prompt_b = (
        "Scene: front vehicle is slowing down.\n"
        "Reasoning tokens: <latent_1> <latent_2> <latent_3> <latent_4>\n"
        "Answer:"
    )

    sync_if_cuda()
    result_a_start = time.perf_counter()
    result_a = timed_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_a,
        max_new_tokens=80,
        device=device,
        title="A: short prompt + long generated CoT",
    )
    sync_if_cuda()
    result_a_end = time.perf_counter()
    result_a["total_elapsed_seconds"] = result_a_end - result_a_start

    sync_if_cuda()
    result_b_start = time.perf_counter()
    result_b = timed_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_b,
        max_new_tokens=20,
        device=device,
        title="B: latent tokens in prompt + short answer",
    )
    sync_if_cuda()
    result_b_end = time.perf_counter()
    result_b["total_elapsed_seconds"] = result_b_end - result_b_start

    print("\n=== Summary ===")
    print(result_a)
    print(result_b)

    print("\n解释：")
    print("A 的输入短，但 max_new_tokens 长，所以 decode 步数多。")
    print("B 的输入里包含 latent tokens，它们在 prefill 阶段作为输入被处理；输出更短，所以 decode 步数少。")
    print("这个实验只用于建立直觉，不代表真实 OneVL 性能结论。")


if __name__ == "__main__":
    main()
