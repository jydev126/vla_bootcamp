
# Tokenizer

```text
tokenizer 不是神经网络模型。
tokenizer 的作用是把文本转换成整数 id。

原始文本不能直接输入模型。
模型只能处理数字张量。

input_ids 就是 tokenizer 输出的整数序列。
例如字符级 tokenizer 中，每个字符对应一个整数 id。

encode 是文本到 id。
decode 是 id 到文本。
```