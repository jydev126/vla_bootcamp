# Day 4: 从 GPT token 到 image token

目标: 像 Day2 mini GPT 一样，用可运行的小代码理解 VLM 主干结构。

今天不调用现成 VLM API。今天只回答一个问题:

```text
图片怎么变成 LLM 能接住的一串 token?
```

## 运行顺序

建议使用项目虚拟环境:

```bash
.venv/bin/python day4_vlm/01_patchify_image.py
.venv/bin/python day4_vlm/02_tiny_vit_encoder.py --batch-size 2
.venv/bin/python day4_vlm/03_minigpt4_bridge.py --batch-size 2
.venv/bin/python day4_vlm/04_train_tiny_vlm.py
```

## 四个脚本分别掌握什么

```text
01_patchify_image.py
image [B, 3, 64, 64] -> patches [B, 64, 192]
重点: patch 就是图像版 token; unpatchify 验证切块不丢信息。

02_tiny_vit_encoder.py
patches -> patch_tokens -> visual_features
重点: Linear 像 token embedding, position_embedding 像 GPT 位置编码。

03_minigpt4_bridge.py
visual_features -> projector -> visual_tokens
text ids -> text_tokens
concat([visual_tokens, text_tokens]) -> tiny LLM hidden
重点: projector 把 vision_dim=64 对齐到 llm_dim=96。

04_train_tiny_vlm.py
image + prompt -> ANSWER hidden -> reason logits
重点: 真正训练 tiny VLM, 看训练前/训练后预测变化。
```

## 重点理解: Linear + position 为什么是加法

在 `02_tiny_vit_encoder.py` 里:

```python
patch_tokens = self.patch_projection(patches)
patch_tokens = patch_tokens + self.position_embedding
```

可以把第一行理解成:

```text
patch 内容 -> 内容特征向量
```

第二行是在做:

```text
内容特征向量 + 位置特征向量 -> 带空间身份的 patch token
```

`position_embedding` 是可训练参数，形状是 `[1, 64, 64]`。它相当于给 64 个 patch 位置各准备一个可学习的特征偏移量。第 0 个 patch 会加第 0 个位置向量，第 63 个 patch 会加第 63 个位置向量。

所以数理上可以写成:

```text
x_i = Linear(patch_i)
z_i = x_i + p_i
```

其中:

```text
x_i: 第 i 个 patch 里面有什么
p_i: 第 i 个 patch 在哪里
z_i: 第 i 个位置上的这个 patch
```

这里用加法不是因为只有加法能融合信息，而是因为加法简单、稳定、不会改变 hidden size，并且后面的 attention 会继续学习更复杂的内容-位置交互。

## 主线 shape

```text
image
-> pixel_values:    [B, 3, 64, 64]
-> patches:         [B, 64, 192]
-> patch_tokens:    [B, 64, 64]
-> visual_features: [B, 64, 64]
-> visual_tokens:   [B, 64, 96]

prompt ids
-> text_tokens:     [B, 6, 96]

concat
-> vlm_inputs:      [B, 70, 96]
-> hidden:          [B, 70, 96]
-> answer_hidden:   [B, 96]
-> logits:          [B, 4]
```

## 和 MiniGPT-4 的对应关系

- `TinyViTEncoder`: toy vision encoder
- `VisionProjector`: toy projection layer
- `TinyTextBackbone`: toy LLM backbone
- `answer_head`: 为了教学把生成任务简化成 reason 分类

MiniGPT-4 的真实模型更大、更复杂，但关键骨架一样:

```text
vision encoder -> projector -> LLM hidden space
```

## 你应该看到什么

`04_train_tiny_vlm.py` 会先打印训练前预测，再训练，再打印训练后预测。

训练前通常是随机猜。训练后应该能明显学会部分规则，例如:

```text
前车近并且相对速度为负 -> brake_due_to_close_lead
左转提示 -> lane_change_left
右转提示 -> lane_change_right
```

这个脚本的目的不是追求 SOTA，而是让你确认:

```text
visual tokens + text tokens -> hidden state -> answer head
```

这条链路真的能被训练起来。
