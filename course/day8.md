# Day 8 上课台词：把视觉 token 接回 action

## 开场

今天是 Day 8。

今天是新版路线里非常关键的一天。

因为前几天学的东西终于要合在一起了。

我们先回顾一下。

Day4，我们学了 VLM 的结构：

```text
image -> patches -> visual features -> projector -> visual tokens
text -> text tokens
visual + text -> hidden state
```

Day5，我们学了 projector alignment 和 instruction tuning。

也就是视觉特征怎么对齐到语言模型 hidden space，以及模型怎么根据图像和指令回答。

Day6，我们学了 ACT-style continuous action。

也就是：

```text
observation hidden -> action queries -> continuous action chunk
```

Day7，我们学了 OpenVLA-style action token。

也就是：

```text
continuous action -> discrete action ids -> causal LM
```

今天我们回到 VLA 主线。

今天只做一件事：

把 Day4 的 visual tokens，和 Day6 的 trajectory head 接起来。

换句话说，今天我们要让模型不只是回答“场景里发生了什么”，而是直接输出未来轨迹。

今天的主线是：

```text
image -> visual tokens
state tokens -> text tokens
visual + text + latent -> fusion hidden
latent hidden -> trajectory
```

今天脚本只有一个：

```bash
python day8_vla_bridge/01_visual_tokens_to_waypoints.py
```

你今天要掌握到什么程度？

第一，知道 VLM hidden state 为什么可以接 action head。

第二，知道 latent tokens 在这里扮演什么角色。

第三，知道 visual tokens、text tokens、latent tokens 拼起来后的 shape。

第四，知道 trajectory loss 如何训练 latent 和 fusion transformer。

---

## 第一部分：为什么需要 VLA bridge

到目前为止，我们有两个世界。

第一个世界是 VLM。

它擅长：

```text
图像 + 文本 -> hidden state -> answer
```

第二个世界是 action model。

它擅长：

```text
状态 -> future waypoints
```

VLA 要做的事情，就是把这两个世界连起来。

如果一个模型只能说：

```text
前车很近，应该减速。
```

那它还只是解释模型。

自动驾驶系统最后需要的是：

```text
未来轨迹坐标
```

比如：

```text
[(0.5, 0.0), (0.9, 0.0), ..., (2.0, 0.0)]
```

所以我们需要一个 bridge：

```text
VLM hidden state -> action output
```

今天的 bridge 做法是：

在 visual tokens 和 state text tokens 后面，加一组 latent tokens。

让它们一起进入 fusion transformer。

最后取 latent tokens 位置的 hidden state，送进 trajectory head。

---

## 第二部分：Dataset 里有哪些输入

脚本里的 Dataset 叫：

```python
PixelActionDataset
```

每个样本返回四样东西：

```text
image
ids
mask
future trajectory
```

image 是 BEV 图像 tensor。

shape 是：

```text
[3, 64, 64]
```

DataLoader 后是：

```text
[B, 3, 64, 64]
```

ids 是 state tokens 编码后的 token ids。

比如：

```text
BOS speed_4 dist_2 rel_1 cmd_brake hist0_x_... EOS
```

shape 是：

```text
[B, T]
```

mask 标记哪些位置是真 token，哪些是 padding。

future trajectory 是监督目标。

shape 是：

```text
[B, 12]
```

也就是 6 个 waypoint flatten 后的 12 个数。

所以今天的样本已经同时包含三类信息：

```text
视觉输入
状态/语言 token 输入
动作监督
```

这就是 VLA 数据的最小形态。

---

## 第三部分：VisualWaypointVLA 的模块

模型叫：

```python
VisualWaypointVLA
```

它里面有：

```text
vision_encoder
projector
text_embedding
latents
position embedding
fusion transformer
trajectory head
```

我们逐个解释。

vision_encoder 来自 Day4。

它做：

```text
image -> visual_features
```

projector 也来自 Day4。

它做：

```text
visual_features -> visual_tokens
```

text_embedding 做：

```text
state token ids -> text token embeddings
```

latents 是今天最关键的新东西。

它是：

```python
self.latents = nn.Parameter(torch.randn(1, num_latents, 96) * 0.02)
```

也就是说，它不是输入样本里的 token id。

它是模型自己学的一组向量。

如果 num_latents 是 4，那么它 shape 是：

```text
[1, 4, 96]
```

forward 时 expand 到 batch：

```text
[B, 4, 96]
```

这些 latent tokens 是什么？

你可以把它们理解成 planning scratchpad。

也就是专门留给模型汇聚视觉和状态信息、最后用于规划的槽位。

---

## 第四部分：为什么需要 latent tokens

你可能会问：为什么不直接取最后一个 text token hidden 去预测轨迹？

可以。

但 latent tokens 有几个好处。

第一，它们是专门为规划任务准备的位置。

state tokens 表示输入状态。

visual tokens 表示图像 patch。

latent tokens 不直接表示某个输入字段，它们可以学习成为“规划表示槽位”。

第二，它们可以有多个。

多个 latent 可以分工。

比如一个 latent 关注前车，一个关注导航命令，一个关注历史轨迹，一个关注未来轨迹平滑性。

当然 toy 模型里不会真的这么清晰，但这个直觉很重要。

第三，它们和 OneVL 的 latent token 思想直接对应。

Day9 会把 latent 进一步拆成 language latent 和 visual/world latent。

所以今天这组 latent 是桥梁。

---

## 第五部分：forward 里的 shape

我们看 forward 主线：

```python
vf, patches = self.vision_encoder(image)
visual = self.projector(vf)
text = self.text_embedding(ids)
lat = self.latents.expand(image.shape[0], -1, -1)
tokens = torch.cat([visual, text, lat], dim=1)
hidden = self.fusion(tokens + self.pos(pos))
latent_h = hidden[:, -lat.shape[1]:]
traj = self.head(latent_h.mean(dim=1))
```

逐个画 shape。

image：

```text
[B, 3, 64, 64]
```

patches：

```text
[B, 64, 192]
```

visual features 经过 projector 后：

```text
visual: [B, 64, 96]
```

text embedding：

```text
text: [B, T, 96]
```

latent tokens：

```text
lat: [B, L, 96]
```

拼接后：

```text
tokens: [B, 64 + T + L, 96]
```

fusion transformer 输出：

```text
hidden: [B, 64 + T + L, 96]
```

取最后 L 个位置：

```text
latent_h: [B, L, 96]
```

mean pooling：

```text
[B, 96]
```

trajectory head：

```text
[B, 12]
```

这就是今天最重要的 shape 链。

你以后看 OneVL 时，也要这样画。

不要被模块名吓到。

先找 tokens，找 hidden，找 latent positions，找 head。

---

## 第六部分：trajectory loss 怎么训练 latent

训练时 loss 是 MSELoss：

```python
loss = mse(pred_traj, gt_traj)
```

你可能会问：latent tokens 没有自己的标签，它们怎么学？

答案是通过 trajectory loss 间接学。

计算图是连通的：

```text
trajectory loss
-> trajectory head
-> latent_h
-> fusion transformer
-> visual tokens / text tokens / latent parameters
-> projector / vision encoder / text embedding
```

只要中间没有 detach，梯度就会传回去。

所以 latent tokens 会被训练成对轨迹预测有用的表示。

这和 Day3 说的特殊 token hidden 是同一个思想。

不是 token 字符串重要。

是它经过模型之后的 hidden state 重要。

---

## 第七部分：今天和 Day7 action token 的关系

今天走的是 continuous action head 路线。

也就是：

```text
latent hidden -> trajectory head -> continuous waypoints
```

Day7 走的是 action token 路线：

```text
trajectory -> action ids -> causal LM -> action token logits
```

两条路线都可以做 VLA。

今天更接近 OneVL 里 hidden state 接 planning head 的抽象。

Day7 更接近 OpenVLA 把 action 纳入 token generation 的抽象。

你现在不需要判断谁更好。

你需要能看懂两种路线。

---

## 收尾

今天我们把视觉 token 接回 action 了。

你现在应该能完整讲出：

```text
image -> TinyViTEncoder -> visual features -> projector -> visual tokens
state tokens -> text embeddings
visual tokens + text tokens + latent tokens -> fusion transformer
latent hidden -> trajectory head -> future waypoints
```

今天最重要的一句话是：

```text
VLA 不是只让 VLM 描述场景，而是把多模态 hidden state 接到 action 输出。
```

明天 Day9，我们会在这个基础上继续接近 OneVL。

我们会把 latent 拆成 language latent 和 visual/world latent。

然后用 language auxiliary 和 world auxiliary 去塑造这些 latent hidden states。
