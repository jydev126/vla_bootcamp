# Bigram LM

```text
Bigram LM 是最简单的 next-token prediction 模型。
它只根据当前 token 预测下一个 token。

idx 是输入 token ids，shape 是 [B, T]。
B 是 batch size。
T 是 sequence length，也叫 block size。

logits 是模型输出的未归一化分数。
在语言模型里，logits 的 shape 是 [B, T, vocab_size]。

[B, T, vocab_size] 的意思是：
每个 batch，
每个 token 位置，
都输出一个对整个词表的预测分布。

cross entropy 用来衡量：
模型预测的 token 分布和真实下一个 token id 之间的差距。

训练 next-token prediction 时：
输入是 data[0:T]
目标是 data[1:T+1]

也就是：
当前 token 预测下一个 token。
```