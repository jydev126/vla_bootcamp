# Day 3 上课台词：HuggingFace / LLM / hidden_states / latent token / prefill-decode

## 开场

今天是 Day 3。

前两天我们已经做了两件事。

Day 1，我们掌握了 PyTorch 最小闭环：

Tensor 是数据容器。

loss 是模型错了多少。

backward 是把错误往回传，计算梯度。

optimizer.step 是根据梯度更新参数。

Dataset / DataLoader 是把数据喂给模型的工程入口。

Day 2，我们进入了语言模型本身：

文本会先变成 token。

token 会变成 embedding。

Transformer 通过 attention 让每个 token 看前面的上下文。

causal LM 的训练目标，是根据左边已有 token，预测下一个 token。

所以到今天，我们不再自己手写 mini GPT。

今天要做的事情是：

用 HuggingFace 跑一个真实的 Causal LM。

你要从今天开始熟悉真实大模型代码长什么样。

今天的关键词是：

AutoTokenizer。

AutoModelForCausalLM。

input_ids。

attention_mask。

labels。

logits。

generate。

hidden_states。

KV cache。

prefill。

decode。

今天结束以后，你必须能把一条链路讲清楚：

一段文字 prompt，先经过 tokenizer，变成 input_ids。

input_ids 进入 AutoModelForCausalLM。

模型 forward 之后输出 logits 和 hidden_states。

logits 用来预测下一个 token。

hidden_states 是每个 token 在模型内部的向量表示。

generate 不是一次 forward，而是循环多次 forward，逐 token 生成。

最后，我们还会模拟 latent token。

因为 OneVL 里面最关键的抽象就是：

不是让模型直接输出所有推理文字，而是在序列里面放一些 latent token，然后取这些 token 位置的 hidden state，送给辅助 decoder，去做轨迹、世界状态、规划相关任务。

这就是今天的主线。

---

## 第一部分：为什么今天要学 HuggingFace

我们先说一件事。

前两天你手写的 mini GPT，本质是为了理解原理。

但是工程里面没人每次都从零写 tokenizer、attention、Transformer block、LM head。

真实项目里，我们通常直接加载一个预训练模型。

比如：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

这两行代码今天会反复出现。

你看到它不要觉得它很简单。

它背后其实代表了真实 LLM 工程的入口。

AutoTokenizer 的意思是：

根据模型名字，自动加载这个模型配套的 tokenizer。

AutoModelForCausalLM 的意思是：

根据模型名字，自动加载一个 causal language model。

这里的 causal language model，就是 Day 2 讲过的 next token prediction 模型。

也就是：

给定左边的 token，预测右边下一个 token。

所以今天的第一句话你要记住：

HuggingFace 不是新理论，它是把 Day 1 的 PyTorch 和 Day 2 的 Causal LM 封装成了工程接口。

你今天不是重新学一个陌生系统。

你是在把前两天的东西，接到真实 LLM API 上。

---

## 第二部分：先看最小代码

我们先看第一份代码。

文件叫：

```text
day3_llm/01_hf_causal_lm_infer.py
```

代码如下：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

prompt = "The car is slowing down because"

inputs = tokenizer(prompt, return_tensors="pt")

print("input_ids:", inputs["input_ids"])
print("input_ids shape:", inputs["input_ids"].shape)
print("attention_mask:", inputs["attention_mask"])
print("attention_mask shape:", inputs["attention_mask"].shape)

with torch.no_grad():
    outputs = model(**inputs)

print("logits shape:", outputs.logits.shape)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)

print("generated_ids shape:", generated_ids.shape)

text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(text)
```

我们现在一行一行讲。

第一行：

```python
import torch
```

这个不用多说。

HuggingFace 的模型底层还是 PyTorch 模型。

所以你 Day 1 学的 `nn.Module`、`model.eval()`、`torch.no_grad()`，今天都还会出现。

第二行：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

这里有两个东西。

第一个是 AutoTokenizer。

它负责把文本变成 token id。

第二个是 AutoModelForCausalLM。

它负责加载一个带语言模型头的 causal LM。

注意这个名字里有两个关键词：

第一个关键词是 Causal。

意思是模型只能看左边，不能偷看右边。

第二个关键词是 LM。

也就是 Language Model。

所以 AutoModelForCausalLM 可以理解成：

自动加载一个“用于下一个 token 预测”的语言模型。

然后：

```python
MODEL_NAME = "sshleifer/tiny-gpt2"
```

这里我们用 tiny-gpt2。

为什么不用 Qwen、Llama、DeepSeek？

因为今天不是比效果。

今天要看 shape、看接口、看 hidden_states。

tiny-gpt2 很小，下载快，跑得快，适合教学。

然后：

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

这句代码的意思是：

从 HuggingFace 加载这个模型对应的 tokenizer。

注意，tokenizer 不是通用的。

不同模型的 tokenizer 可能不同。

GPT-2 有 GPT-2 的 tokenizer。

Qwen 有 Qwen 的 tokenizer。

Llama 有 Llama 的 tokenizer。

同一句话，不同 tokenizer 切出来的 token 可能不一样。

所以模型和 tokenizer 必须配套。

再看：

```python
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
```

这句代码就是加载模型本体。

这个 model 是一个 PyTorch 模型。

你可以把它理解成 Day 2 里面我们手写的 GPT，只不过它是别人已经写好、训练好的版本。

然后：

```python
model.eval()
```

这句来自 Day 1。

eval 的意思是进入推理模式。

它会关闭 dropout 这类训练时才用的随机行为。

今天我们只是推理，所以要 `model.eval()`。

然后：

```python
prompt = "The car is slowing down because"
```

这是一段输入文本。

注意，此时它还是普通字符串。

模型不能直接吃字符串。

模型只能吃数字。

所以要 tokenizer。

```python
inputs = tokenizer(prompt, return_tensors="pt")
```

这句非常重要。

`tokenizer(prompt)` 做的是：

把字符串拆成 token。

再把 token 转成 token id。

`return_tensors="pt"` 的意思是：

返回 PyTorch Tensor。

所以 inputs 不是一个字符串，而是一个字典。

大概长这样：

```python
{
    "input_ids": tensor([[...]]),
    "attention_mask": tensor([[...]])
}
```

这里你第一次可能会不适应。

为什么 inputs 是字典？

因为模型 forward 不只需要 input_ids，有时候还需要 attention_mask、position_ids、labels 等等。

所以 HuggingFace 通常把这些输入打包成一个 dict。

然后我们打印：

```python
print("input_ids:", inputs["input_ids"])
print("input_ids shape:", inputs["input_ids"].shape)
```

你会看到 input_ids 是一个二维 Tensor。

shape 是：

```text
[B, T]
```

B 是 batch size。

T 是 sequence length，也就是 token 数量。

如果你只输入一句话，B 通常是 1。

T 取决于 tokenizer 把这句话切成了多少个 token。

所以你要记住：

```text
input_ids 不是文字。
input_ids 是 token 的整数编号。
shape 是 [B, T]。
```

然后：

```python
print("attention_mask:", inputs["attention_mask"])
print("attention_mask shape:", inputs["attention_mask"].shape)
```

attention_mask 也是 `[B, T]`。

它表示哪些位置是真 token，哪些位置是 padding。

如果某个位置是 1，说明是真 token。

如果某个位置是 0，说明是 padding，不应该参与 attention。

今天我们只输入一句短文本，没有 padding，所以大概率全是 1。

但是你必须知道 attention_mask 的作用。

因为后面做 batch 输入时，不同句子长度不同，就一定会出现 padding。

模型要知道哪些 token 是补出来的，哪些 token 是原始文本。

然后：

```python
with torch.no_grad():
    outputs = model(**inputs)
```

这里有两个点要讲。

第一个点：

`torch.no_grad()` 是 Day 1 的内容。

推理时不需要计算梯度。

不计算梯度可以省显存、省时间。

所以推理代码一般放在 `torch.no_grad()` 里面。

第二个点：

```python
model(**inputs)
```

这里的两个星号 `**` 是 Python 语法。

它的意思是把字典展开成函数参数。

如果 inputs 是：

```python
{
    "input_ids": input_ids,
    "attention_mask": attention_mask
}
```

那么：

```python
model(**inputs)
```

等价于：

```python
model(input_ids=input_ids, attention_mask=attention_mask)
```

这个地方很多人第一次看会懵。

但其实它只是字典展开。

然后模型输出 outputs。

我们打印：

```python
print("logits shape:", outputs.logits.shape)
```

logits 是什么？

logits 是模型对每个位置的下一个 token 预测分数。

它的 shape 是：

```text
[B, T, vocab_size]
```

这里要慢一点讲。

input_ids 是 `[B, T]`。

模型看到每一个 token 位置，都会输出一个 vocab_size 维的向量。

这个向量表示：

词表里的每一个 token，都有一个分数。

分数越高，模型越认为它适合作为下一个 token。

所以 logits 是三维的：

batch 维度。

sequence 维度。

vocab 维度。

也就是：

```text
[B, T, vocab_size]
```

如果我们只关心“接下来要生成哪个 token”，应该看哪个位置？

答案是最后一个位置。

也就是：

```python
next_token_logits = outputs.logits[:, -1, :]
```

为什么是最后一个位置？

因为 causal LM 的每个位置都在预测“下一个 token”。

第 0 个位置预测第 1 个 token。

第 1 个位置预测第 2 个 token。

最后一个位置预测 prompt 后面的下一个 token。

所以你一定要记住：

```text
logits[:, -1, :] 才是下一个 token 的预测分布。
```

接下来：

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
```

这就是生成。

这里要重点讲 forward 和 generate 的区别。

`model(**inputs)` 只算一次 forward。

它不会自动生成一长段文字。

它只输出当前输入位置上的 logits。

而 `model.generate()` 是一个生成流程。

它内部会反复调用 forward。

每次生成一个新 token。

然后把新 token 拼到原序列后面。

再继续生成下一个 token。

所以：

```text
forward = 算一次。
generate = 循环生成。
```

`max_new_tokens=20` 的意思是最多新生成 20 个 token。

`do_sample=False` 的意思是不随机采样，通常就是贪心解码。

最后：

```python
text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(text)
```

这里是把 token id 转回字符串。

也就是：

```text
token id -> token -> text
```

所以第一份代码的链路就是：

```text
prompt
-> tokenizer
-> input_ids / attention_mask
-> model forward
-> logits
-> generate
-> decode text
```

这一段你必须背下来。

---

## 第三部分：input_ids、attention_mask、labels、logits

现在我们集中解释四个最容易混的概念。

第一个，input_ids。

input_ids 是模型输入。

它是 token 的编号。

比如一句话：

```text
The car is slowing down
```

会被 tokenizer 切成若干 token。

每个 token 对应一个整数 id。

最后得到一个 Tensor：

```text
input_ids: [B, T]
```

第二个，attention_mask。

attention_mask 也是输入。

它告诉模型哪些 token 是有效的，哪些 token 是 padding。

shape 也是：

```text
attention_mask: [B, T]
```

如果你只输入一句话，一般全是 1。

如果一个 batch 里面句子长短不一样，就需要 padding。

padding 的位置 attention_mask 是 0。

第三个，labels。

labels 今天的推理代码里暂时没用，但你必须知道它是什么。

labels 是训练时用的目标答案。

在 causal LM 训练里，input_ids 和 labels 经常几乎一样。

比如输入是：

```text
I love autonomous driving
```

模型训练目标是：

看到 I，预测 love。

看到 I love，预测 autonomous。

看到 I love autonomous，预测 driving。

所以在 HuggingFace 的 Causal LM 里，你经常会看到：

```python
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
```

这里 labels=input_ids 不是写错了。

因为 causal LM 会在模型内部做 shift。

它会让当前位置预测下一个位置。

所以你今天要知道：

```text
input_ids 是输入 token。
labels 是训练目标 token。
logits 是模型预测出来的分数。
loss 是 logits 和 labels 算出来的错误。
```

第四个，logits。

logits 是模型输出。

shape 是：

```text
[B, T, vocab_size]
```

你不要把 logits 当成生成结果。

logits 只是分数。

还要经过 argmax 或 sample，才能得到具体的 token id。

generate 里面就做了这件事。

---

## 第四部分：hidden_states 是什么

现在进入今天最重要的第二个脚本。

文件：

```text
day3_llm/02_hidden_states_probe.py
```

代码如下：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

prompt = "The front vehicle is slowing down."

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

print("num hidden_states:", len(outputs.hidden_states))

for i, h in enumerate(outputs.hidden_states):
    print(i, h.shape)

last_layer_hidden = outputs.hidden_states[-1]
last_token_hidden = last_layer_hidden[:, -1, :]

print("last_layer_hidden shape:", last_layer_hidden.shape)
print("last_token_hidden shape:", last_token_hidden.shape)
```

我们重点讲这句：

```python
outputs = model(
    **inputs,
    output_hidden_states=True,
    return_dict=True
)
```

默认情况下，模型可能只返回 logits。

如果你想看每一层 Transformer 的 hidden state，就要加：

```python
output_hidden_states=True
```

`return_dict=True` 的意思是让输出变成类似字典的结构，可以用：

```python
outputs.logits
outputs.hidden_states
```

来访问。

现在问题来了：

hidden_states 是什么？

我们回到 Day 2。

一个 token 进模型之前，只是一个 token id。

token id 会查 embedding table，变成一个向量。

然后这个向量经过第 1 层 Transformer。

再经过第 2 层 Transformer。

再经过第 3 层 Transformer。

每过一层，这个 token 的向量表示都会变化。

这些中间向量，就叫 hidden state。

所以 hidden state 的本质是：

```text
每个 token 在模型内部的语义向量表示。
```

它不是文字。

它也不是 token id。

它是一个浮点向量。

shape 通常是：

```text
[B, T, hidden_dim]
```

B 是 batch size。

T 是 token 数量。

hidden_dim 是模型隐藏层维度。

比如 tiny-gpt2 的 hidden_dim 很小。

大模型里面 hidden_dim 可能是 4096、5120、8192。

然后：

```python
print("num hidden_states:", len(outputs.hidden_states))
```

这里打印的是 hidden_states 有多少层。

注意，很多模型返回的 hidden_states 数量，可能是层数加 1。

为什么？

因为它可能包括 embedding 输出，以及每一层 Transformer 的输出。

所以如果模型有 2 层，你可能看到 3 个 hidden state。

这不是 bug。

这是因为第 0 个 hidden state 可能是 embedding 后的表示。

最后一个：

```python
last_layer_hidden = outputs.hidden_states[-1]
```

就是最后一层的 hidden state。

shape 是：

```text
[B, T, hidden_dim]
```

然后：

```python
last_token_hidden = last_layer_hidden[:, -1, :]
```

这句是取最后一个 token 的 hidden state。

shape 是：

```text
[B, hidden_dim]
```

这里你要建立一个非常重要的连接：

普通 LLM 是怎么生成 token 的？

它会把最后一层 hidden state 送进 LM head。

LM head 输出 logits。

logits 决定下一个 token。

也就是：

```text
hidden state -> LM head -> logits -> next token
```

但是在 VLA 或 OneVL 里面，我们不一定只用 hidden state 生成文字。

我们可以拿某些特殊位置的 hidden state，送到别的 head 里面。

比如：

```text
hidden state -> trajectory decoder -> 轨迹
hidden state -> world decoder -> 世界状态
hidden state -> action head -> 控制动作
```

所以今天的 hidden state 是 OneVL 的地基。

你后面看 OneVL 代码时，如果看到：

```text
取 latent token 位置 hidden state
送入 auxiliary decoder
```

你就知道它在干什么了。

它不是在拿 token 字符串。

它是在拿 Transformer 处理后的内部向量。

---

## 第五部分：为什么最后一个 token 很重要

我们再停一下。

你可能会问：

为什么老是取最后一个 token？

为什么不是第一个？

这要回到 causal LM 的结构。

causal LM 的每个 token 只能看自己和左边的 token。

所以越靠右的位置，看过的上下文越多。

最后一个 token 的 hidden state，已经融合了整个 prompt 的信息。

比如 prompt 是：

```text
The front vehicle is slowing down.
```

最后一个 token 的 hidden state 理论上可以看到：

The。

front。

vehicle。

is。

slowing。

down。

所以它适合用来预测下一个 token。

也适合当作某种整体上下文表示。

但是 OneVL 里的 latent token 会更进一步。

它不是只拿最后一个 token。

它会在 prompt 里面插入一些特殊 token。

比如：

```text
<latent_1><latent_2><latent_3><latent_4>
```

然后取这些位置的 hidden state。

这些 latent token 位置的 hidden state，可以被训练成某种中间思考状态。

这就是下一部分。

---

## 第六部分：模拟 latent token

现在看第三份代码。

文件：

```text
day3_llm/03_fake_latent_token.py
```

代码如下：

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

special_tokens = {
    "additional_special_tokens": [
        "<latent_1>",
        "<latent_2>",
        "<latent_3>",
        "<latent_4>",
    ]
}

num_added = tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

print("num_added:", num_added)
print("vocab size:", len(tokenizer))

model.eval()

prompt = (
    "Scene: front vehicle is slowing down.\n"
    "Reasoning tokens: <latent_1><latent_2><latent_3><latent_4>\n"
    "Answer:"
)

inputs = tokenizer(prompt, return_tensors="pt")

input_ids = inputs["input_ids"]
print("input_ids shape:", input_ids.shape)
print("tokens:")
print(tokenizer.convert_ids_to_tokens(input_ids[0]))

latent_token_ids = tokenizer.convert_tokens_to_ids([
    "<latent_1>",
    "<latent_2>",
    "<latent_3>",
    "<latent_4>",
])

print("latent_token_ids:", latent_token_ids)

latent_positions = []

for latent_id in latent_token_ids:
    pos = (input_ids[0] == latent_id).nonzero(as_tuple=True)[0]
    latent_positions.append(pos.item())

latent_positions = torch.tensor(latent_positions)

print("latent_positions:", latent_positions)

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True
    )

last_hidden = outputs.hidden_states[-1]

latent_hidden = last_hidden[:, latent_positions, :]

print("last_hidden shape:", last_hidden.shape)
print("latent_hidden shape:", latent_hidden.shape)
```

这一段非常重要。

我们慢慢讲。

首先：

```python
special_tokens = {
    "additional_special_tokens": [
        "<latent_1>",
        "<latent_2>",
        "<latent_3>",
        "<latent_4>",
    ]
}
```

这里定义了四个特殊 token。

它们不是普通英文单词。

它们是我们人为加进去的占位符。

然后：

```python
num_added = tokenizer.add_special_tokens(special_tokens)
```

这句是把这些 token 加进 tokenizer 的词表。

也就是说，原来 tokenizer 不认识 `<latent_1>`。

加完以后，它认识了。

它可以把 `<latent_1>` 当成一个完整 token，而不是拆成 `<`、`latent`、`_`、`1`、`>`。

接下来：

```python
model.resize_token_embeddings(len(tokenizer))
```

这句很关键。

为什么加完 tokenizer，还要 resize model embedding？

因为 tokenizer 词表变大了。

原来模型 embedding table 的大小，等于原始词表大小。

现在你新增了 4 个 token，词表长度增加了 4。

如果不 resize，模型的 embedding table 里面没有这 4 个新 token 的向量。

所以要调用：

```python
model.resize_token_embeddings(len(tokenizer))
```

它会扩展 embedding table，让模型能处理新 token。

这里有一个现实问题：

新增 token 的 embedding 是随机初始化的。

也就是说，刚加进去的 `<latent_1>` 本身没有任何特殊能力。

它只是一个新 token。

如果不训练，它没有学会承载推理信息。

所以你今天一定要明白：

```text
latent token 本身没有魔法。
```

它真正有用，是因为训练时我们强迫模型在这些 token 位置的 hidden state 里面存有用信息。

继续看 prompt：

```python
prompt = (
    "Scene: front vehicle is slowing down.\n"
    "Reasoning tokens: <latent_1><latent_2><latent_3><latent_4>\n"
    "Answer:"
)
```

这里我们构造了一个自动驾驶场景。

前车正在减速。

然后我们放了四个 latent token。

最后接 Answer。

这个形式已经有点像 OneVL 了。

真实 OneVL 里面，输入可能有图像 token、语言 token、latent token。

然后模型处理整个序列。

我们现在只是用文本模拟。

接下来：

```python
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
```

还是把 prompt 变成 input_ids。

然后：

```python
print(tokenizer.convert_ids_to_tokens(input_ids[0]))
```

这句很重要。

它可以让你看到 tokenizer 到底怎么切 token。

你要检查 `<latent_1>` 是不是被当成一个 token。

如果它被拆开了，就不对。

然后：

```python
latent_token_ids = tokenizer.convert_tokens_to_ids([
    "<latent_1>",
    "<latent_2>",
    "<latent_3>",
    "<latent_4>",
])
```

这句是把 latent token 字符串转成 token id。

例如：

```text
<latent_1> -> 50257
<latent_2> -> 50258
```

具体数字不重要。

重要的是每个 special token 都有自己的 id。

然后我们要找这些 token 在序列里的位置。

```python
latent_positions = []

for latent_id in latent_token_ids:
    pos = (input_ids[0] == latent_id).nonzero(as_tuple=True)[0]
    latent_positions.append(pos.item())
```

这段逻辑是：

在 input_ids 里面找等于 latent_id 的位置。

比如 input_ids 可能是：

```text
[100, 200, 300, 50257, 50258, 50259, 50260, 400]
```

那么 latent token 的位置就是：

```text
[3, 4, 5, 6]
```

这一步非常重要。

因为你后面要从 hidden state 里面取这些位置。

继续：

```python
outputs = model(
    **inputs,
    output_hidden_states=True,
    return_dict=True
)
```

我们拿到所有 hidden state。

然后：

```python
last_hidden = outputs.hidden_states[-1]
```

最后一层 hidden state，shape 是：

```text
[B, T, hidden_dim]
```

最后：

```python
latent_hidden = last_hidden[:, latent_positions, :]
```

这句就是今天最核心的代码。

它的意思是：

从所有 token 的 hidden state 里面，只取 latent token 位置的 hidden state。

最后得到：

```text
latent_hidden shape = [B, num_latent, hidden_dim]
```

比如：

```text
[1, 4, 2]
```

或者真实大模型里：

```text
[1, 4, 4096]
```

这就是 latent token 的内部表示。

我们今天用 tiny-gpt2，只是模拟这个机制。

但抽象已经完全一样。

你必须记住：

```text
<latent_1> 这个字符串不是重点。
<latent_1> 对应位置的 hidden state 才是重点。
```

OneVL 后面真正用的，不是 `<latent_1>` 这几个字符。

而是：

```python
latent_hidden = hidden_states[:, latent_positions, :]
```

然后送到 auxiliary decoder。

---

## 第七部分：latent token 和 CoT 的关系

现在我们把 latent token 和 CoT 对比一下。

普通 CoT 是什么？

就是让模型输出一段显式推理文字。

比如：

```text
前车减速，所以自车应该减速。
如果距离小于安全距离，需要制动。
最终动作是减速。
```

这个叫显式 chain-of-thought。

它的问题是：

第一，生成很慢。

因为每个文字 token 都要 decode。

第二，它不一定适合控制任务。

因为控制任务最终要的是轨迹、动作、速度规划，不一定需要自然语言解释。

第三，文字推理可能很长，但有效信息很少。

latent token 的想法是：

不把所有中间推理都展开成文字。

而是在序列里放几个特殊 token。

让模型把中间状态压到这些 token 的 hidden state 里。

也就是：

```text
显式 CoT：用文字承载推理。
latent token：用隐藏向量承载推理。
```

这就是你后面理解 OneVL 的关键。

OneVL 不是简单地问大模型：

“请输出轨迹。”

而是可能在模型内部设计某些 latent 位置。

这些位置的 hidden state 被拿出来，接辅助任务。

比如：

```text
世界状态预测
轨迹预测
动作预测
风险判断
```

所以你今天做 fake latent token，不是玩具。

它是为了让你理解 OneVL 的核心抽象。

---

## 第八部分：generate 到底做了什么

现在我们回到 generate。

你前面已经看到：

```python
model.generate(...)
```

它能输出文字。

但是我们要知道它里面大概做了什么。

假设 prompt 是：

```text
The car is slowing down because
```

第一步：

模型 forward 整个 prompt。

得到最后一个位置的 logits。

第二步：

从 logits 里面选一个 token。

比如选出：

```text
it
```

第三步：

把 it 拼到输入后面。

现在输入变成：

```text
The car is slowing down because it
```

第四步：

再 forward。

再选下一个 token。

比如：

```text
needs
```

第五步：

继续拼接：

```text
The car is slowing down because it needs
```

然后继续。

这就是自回归生成。

所以 generate 不是一次把一整段话吐出来。

generate 是一个循环。

每轮生成一个 token。

这也是为什么大模型输出长文本会慢。

因为它不能并行生成未来 token。

第 10 个 token 依赖第 9 个 token。

第 9 个 token 依赖第 8 个 token。

所以 decode 阶段是串行的。

你要记住：

```text
训练时可以并行。
生成时必须自回归串行。
```

训练时为什么可以并行？

因为训练数据已经完整存在。

模型可以一次性看到整段输入，通过 causal mask 保证每个位置只看左边。

但是生成时，未来 token 还不存在。

所以只能一个一个生成。

---

## 第九部分：prefill 和 decode

现在进入今天第四个脚本。

文件：

```text
day3_llm/04_prefill_decode_timing.py
```

代码如下：

```python
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "sshleifer/tiny-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

short_prompt = "Scene: front vehicle is slowing down. Think step by step:"

long_prompt = (
    "Scene: front vehicle is slowing down.\n"
    "Latent reasoning: "
    + " ".join(["<latent>"] * 64)
    + "\nAnswer:"
)

def run_generate(prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")

    t0 = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    t1 = time.perf_counter()

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return t1 - t0, text, inputs["input_ids"].shape, output_ids.shape

t_short, text_short, in_shape_short, out_shape_short = run_generate(
    short_prompt,
    max_new_tokens=80
)

t_long, text_long, in_shape_long, out_shape_long = run_generate(
    long_prompt,
    max_new_tokens=10
)

print("A short prompt + long decode")
print("input shape:", in_shape_short)
print("output shape:", out_shape_short)
print("time:", t_short)
print(text_short)

print("=" * 80)

print("B long prompt + short decode")
print("input shape:", in_shape_long)
print("output shape:", out_shape_long)
print("time:", t_long)
print(text_long)
```

这个脚本不是严格 benchmark。

它是为了让你感受两个阶段。

第一种情况：

```text
短 prompt + 生成很长答案
```

第二种情况：

```text
长 prompt + 只生成很短答案
```

这里涉及两个概念：

prefill。

decode。

prefill 是什么？

prefill 是处理 prompt 的阶段。

比如输入：

```text
Scene: front vehicle is slowing down.
```

模型一次性处理完整 prompt。

它会为 prompt 里的每个 token 计算 hidden state。

同时计算 attention 里面的 key 和 value。

这些 key/value 会放进 KV cache。

所以 prefill 可以理解成：

```text
把已有上下文一次性吃进去，并建立缓存。
```

decode 是什么？

decode 是生成新 token 的阶段。

生成的时候，每次只生成一个 token。

每生成一个 token，就要再根据这个 token 继续生成下一个。

所以 decode 是串行的。

你可以这样理解：

```text
prefill：读题。
decode：一个字一个字作答。
```

读题的时候，你可以一次性读完整段题目。

但是作答的时候，后面的字依赖前面的字。

所以只能一个一个写。

然后是 KV cache。

KV cache 是什么？

在 Transformer attention 里，每个 token 会产生 Q、K、V。

生成第一个新 token 时，prompt 里的所有历史 token 都已经有 K、V。

生成第二个新 token 时，如果没有缓存，模型要重新计算前面所有 token 的 K、V。

这很浪费。

所以 KV cache 会保存历史 token 的 key/value。

后面 decode 时，只需要计算新 token 的 Q、K、V，然后和历史 K/V 做 attention。

所以 KV cache 的作用是：

```text
避免每生成一个 token 都重新计算完整历史。
```

但是注意：

KV cache 不能让 decode 变成完全并行。

它只是减少重复计算。

decode 仍然要一个 token 一个 token 生成。

所以你要记住：

```text
KV cache 加速 decode，但不改变自回归生成的串行本质。
```

---

## 第十部分：为什么 latent token 和 prefill/decode 有关系

现在把 latent token 和 prefill/decode 连接起来。

普通 CoT 是：

```text
模型在 decode 阶段生成一大段推理文字。
```

比如生成 200 个 reasoning token。

这意味着 decode 要跑 200 步。

decode 是串行的，所以慢。

latent token 的思路是：

把一些 latent token 放在输入里面。

比如：

```text
<latent_1><latent_2><latent_3><latent_4>
```

这些 token 在 prefill 阶段就被一次性处理。

然后我们取它们的 hidden state。

也就是说：

```text
显式 CoT：decode 生成文字推理。
latent token：prefill 中形成隐藏推理表示。
```

当然，这里要说清楚。

不是说你随便放 latent token 就一定有推理能力。

必须经过训练。

训练时要让这些 latent hidden states 对 auxiliary loss 有用。

比如：

```text
轨迹预测 loss
世界模型 loss
动作预测 loss
风险预测 loss
```

通过这些 loss，模型才会学会：

在 latent token 的 hidden state 里面编码有用信息。

否则它只是几个随机新增的 token。

这一点非常重要。

今天我们只是模拟机制。

OneVL 真正的工作，是训练这些位置的 hidden state 变得有意义。

---

## 第十一部分：为什么这一步直接对应 OneVL

现在我们明确连接 OneVL。

你后面看 OneVL，大概率会看到类似结构：

输入包括：

```text
image tokens
language tokens
latent tokens
```

image tokens 来自视觉 encoder。

language tokens 来自文本 tokenizer。

latent tokens 是人为插入的特殊位置。

Transformer 统一处理这些 token。

最后取 latent tokens 对应位置的 hidden state。

然后：

```text
latent hidden -> auxiliary decoder -> trajectory / world / planning
```

你今天的 fake latent token 脚本，其实已经把这个抽象做出来了。

只不过今天没有图像。

今天没有训练。

今天没有 auxiliary decoder。

今天只有：

```text
special token
position
hidden state extraction
```

这一步先掌握，后面 Day 4、Day 5、Day 6 才能接上。

Day 4 你会看 VLM。

你会看到 image token 怎么进入语言模型。

Day 5 你会做 VLA toy。

你会把 hidden state 接到 trajectory output。

Day 6 你会看 OneVL repo。

你会找 infer pipeline 和 latent config。

Day 7 你会看 auxiliary loss、world loss、ablation、failure case。

所以 Day 3 的定位就是：

```text
把真实 LLM 的输入、输出、hidden state、生成流程摸清楚。
```

---

## 第十二部分：今天必须掌握的五个回答

现在我们做验收。

第一个问题：

input_ids、attention_mask、logits 分别是什么？

你要这样回答：

input_ids 是 tokenizer 输出的 token 整数编号。

shape 是 `[B, T]`。

B 是 batch size。

T 是 token 数量。

attention_mask 表示哪些位置是真 token，哪些位置是 padding。

shape 也是 `[B, T]`。

logits 是模型对每个位置的下一个 token 的预测分数。

shape 是 `[B, T, vocab_size]`。

第二个问题：

forward 和 generate 有什么区别？

你要这样回答：

forward 是一次模型前向计算。

输入 input_ids，输出 logits、hidden_states、loss 等。

generate 是生成流程。

它内部会循环调用 forward。

每次根据最后一个位置的 logits 选出下一个 token，再把这个 token 拼回输入序列，继续生成。

简单说：

```text
forward 是算一次。
generate 是循环生成。
```

第三个问题：

hidden_states 怎么取？

你要这样回答：

调用模型时传：

```python
output_hidden_states=True
return_dict=True
```

然后：

```python
hidden_states = outputs.hidden_states
last_hidden = hidden_states[-1]
```

如果要取最后一个 token：

```python
last_token_hidden = last_hidden[:, -1, :]
```

如果要取 latent token：

```python
latent_hidden = last_hidden[:, latent_positions, :]
```

第四个问题：

latent token position 怎么找？

你要这样回答：

先把 latent token 转成 token id：

```python
latent_id = tokenizer.convert_tokens_to_ids("<latent_1>")
```

然后在 input_ids 里面找等于这个 id 的位置：

```python
pos = (input_ids[0] == latent_id).nonzero(as_tuple=True)[0]
```

多个 latent token 就循环找。

第五个问题：

prefill 和 decode 为什么速度不同？

你要这样回答：

prefill 是一次性处理 prompt，可以并行计算输入 token 的 hidden state 和 KV cache。

decode 是生成新 token，每次只能生成一个，因为下一个 token 依赖上一个 token。

KV cache 可以保存历史 token 的 K/V，避免重复计算历史上下文。

但 KV cache 不能改变 decode 的自回归串行本质。

---

## 第十三部分：今天写代码时最容易犯的错

第一，忘记 `return_tensors="pt"`。

如果不加它，tokenizer 返回的是 Python list，不是 PyTorch Tensor。

模型不一定能直接吃。

所以写：

```python
inputs = tokenizer(prompt, return_tensors="pt")
```

第二，忘记 `model.eval()`。

推理时要 eval。

虽然 tiny-gpt2 问题不大，但工程习惯必须养成。

第三，忘记 `torch.no_grad()`。

推理时不用梯度。

否则浪费显存。

第四，新增 special token 后，忘记：

```python
model.resize_token_embeddings(len(tokenizer))
```

这是很常见的错误。

tokenizer 认识新 token，不代表 model embedding 认识新 token。

第五，以为 `<latent_1>` 字符串本身有意义。

这是错的。

latent token 的意义来自训练后的 hidden state，不来自字符串。

第六，把 logits 当成最终文字。

logits 只是分数。

decode 之后才是文字。

第七，以为 generate 是一次 forward。

这是错的。

generate 是循环生成。

---

## 第十四部分：今天的最终代码任务安排

今天你按这个顺序做。

第一步，跑通：

```text
01_hf_causal_lm_infer.py
```

你要看到：

```text
input_ids shape
attention_mask shape
logits shape
generated text
```

你要能解释：

```text
input_ids: [B, T]
logits: [B, T, vocab_size]
generate 是自回归生成
```

第二步，跑通：

```text
02_hidden_states_probe.py
```

你要看到：

```text
len(outputs.hidden_states)
每层 hidden state shape
最后一层 hidden state shape
最后一个 token hidden state shape
```

你要能解释：

```text
hidden state 是每个 token 在每一层之后的内部向量表示。
```

第三步，跑通：

```text
03_fake_latent_token.py
```

你要看到：

```text
latent_token_ids
latent_positions
latent_hidden shape
```

你要能解释：

```text
latent token 本身只是占位符。
真正有信息的是它经过 Transformer 后的 hidden state。
```

第四步，跑通：

```text
04_prefill_decode_timing.py
```

你要看到：

```text
短 prompt + 长 decode 的耗时
长 prompt + 短 decode 的耗时
```

你不用纠结 tiny-gpt2 的时间是否稳定。

你只要理解：

```text
prompt 处理是 prefill。
新 token 生成是 decode。
decode 是逐 token 串行。
```

---

## 第十五部分：今天最后总结

今天我们完成了从手写 mini GPT 到真实 HuggingFace LLM 的过渡。

你现在应该知道：

AutoTokenizer 负责把文本变成 token id。

AutoModelForCausalLM 负责加载 causal LM。

input_ids 是 token 编号，shape 是 `[B, T]`。

attention_mask 表示哪些 token 有效，shape 是 `[B, T]`。

logits 是每个位置预测下一个 token 的分数，shape 是 `[B, T, vocab_size]`。

forward 是一次前向计算。

generate 是循环生成。

hidden_states 是模型内部每层每个 token 的向量表示。

最后一层 hidden state 可以接 LM head，也可以接别的任务 head。

latent token 是特殊 token。

它本身没有魔法。

真正重要的是 latent token 位置经过 Transformer 后的 hidden state。

OneVL 里面最关键的抽象，就是取 latent token hidden state，送入 auxiliary decoder，做轨迹、世界状态、动作或规划相关任务。

prefill 是一次性处理 prompt。

decode 是逐 token 生成。

KV cache 保存历史 K/V，减少重复计算，但 decode 仍然是自回归串行。

所以今天的最终闭环就是：

```text
prompt
-> tokenizer
-> input_ids / attention_mask
-> model forward
-> logits / hidden_states
-> 找 latent token 位置
-> 取 latent hidden state
-> 理解 generate 的 prefill/decode
```

如果你能把这条链路讲清楚，Day 3 就过了。

不要被 HuggingFace 的封装吓住。

它看起来像很多 API，其实本质还是 Day 1 和 Day 2 的东西：

Tensor。

forward。

shape。

Transformer。

next-token prediction。

今天你真正要掌握的不是 API 名字，而是每个 Tensor 在模型里代表什么。

尤其是这句话：

```text
latent token 不是天然有意义的文字。
它只是序列里的一个位置。
真正有意义的是这个位置经过 Transformer 后的 hidden state。
训练会迫使这个 hidden state 承载推理、世界状态或轨迹规划信息。
```

这句话，是你后面理解 OneVL 的钥匙。
