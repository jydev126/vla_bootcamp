# day2_transformer/01_char_tokenizer.py

"""
目标：
1. 理解 tokenizer 是什么
2. 理解 input_ids 是什么
3. 实现 encode / decode

核心结论：
tokenizer 不是模型。
tokenizer 只是把文本里的离散符号变成整数 id。
模型真正看到的是 input_ids,不是字符串。
"""

text = "the car slows down and the ego vehicle should brake"

# 1. 找出文本中出现过的所有字符
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 2. 字符 -> 整数 id
stoi = {ch: i for i, ch in enumerate(chars)}

# 3. 整数 id -> 字符
itos = {i: ch for ch, i in stoi.items()}


def encode(s: str) -> list[int]:
    """
    把字符串转换成 token id 列表。

    例如：
    "the" -> [id_t, id_h, id_e]
    """
    return [stoi[ch] for ch in s]


def decode(ids: list[int]) -> str:
    """
    把 token id 列表还原成字符串。

    例如：
    [id_t, id_h, id_e] -> "the"
    """
    return "".join([itos[i] for i in ids])


if __name__ == "__main__":
    ids = encode(text)
    recovered_text = decode(ids)

    print("原始文本:")
    print(text)
    print()

    print("字符表 chars:")
    print(chars)
    print()

    print("词表大小 vocab_size:")
    print(vocab_size)
    print()

    print("stoi:")
    print(stoi)
    print()

    print("itos:")
    print(itos)
    print()

    print("encode 后的 token ids:")
    print(ids)
    print()

    print("decode 回来的文本:")
    print(recovered_text)
    print()

    print("decode(encode(text)) == text ?")
    print(recovered_text == text)