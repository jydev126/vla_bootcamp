# Day 5: VLM projector alignment 与 instruction tuning

Day4 解决结构问题:

```text
image -> visual tokens
text -> text tokens
visual tokens + text tokens -> hidden
```

Day5 解决训练问题:

```text
projector 怎么对齐?
为什么要 freeze?
instruction tuning 和普通分类有什么区别?
```

## 运行顺序

```bash
.venv/bin/python day5_vlm_alignment/01_projector_alignment.py
.venv/bin/python day5_vlm_alignment/02_instruction_tuning.py
```

## 两个脚本分别掌握什么

```text
01_projector_alignment.py
冻结 TinyViTEncoder。
训练 projector + token_pooler + label_anchors。
目标: 让图像 embedding 靠近正确 reason anchor。

02_instruction_tuning.py
冻结 TinyViTEncoder。
训练 projector + tiny_llm + answer_head。
目标: 让模型在不同 prompt 下都能在 ANSWER 位置形成正确 hidden state。
```

## Stage 1: alignment

toy 版 alignment 路径:

```text
image
-> frozen vision encoder
-> trainable projector
-> trainable token_pooler
-> pooled_visual [B, 96]
-> cosine similarity with trainable label_anchors
-> logits [B, 4]
```

这里的分类只是训练手段。真正想让你看到的是:

```text
projector 不只是改 shape, 它要学习把视觉特征挪到语义空间。
```

## Stage 2: instruction tuning

toy 版 instruction tuning 路径:

```text
image -> visual_tokens
prompt -> text_tokens
concat([visual_tokens, text_tokens])
-> tiny_llm
-> ANSWER hidden
-> answer_head
```

训练时 prompt 会随机从下面三种里选:

```text
BOS what affects ego planning ANSWER
BOS choose driving reason now ANSWER
BOS look image and decide ANSWER
```

所以它不只是“看图分类”，而是在模拟:

```text
同一张图, 用户可以用不同方式提问, 模型仍然要回答同一个视觉语义。
```

## 和后面 VLA 的关系

今天的输出头是:

```text
hidden -> answer_head -> reason
```

后面做 VLA 时，只要换成:

```text
hidden -> action_head -> trajectory / action
```

主干思想仍然是:

```text
视觉信息和语言/状态信息先融合成 hidden state, 再由任务 head 读出来。
```
