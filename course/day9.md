# Day 9 上课台词：OneVL-style dual auxiliary

## 开场

今天是 Day 9。

昨天 Day8，我们已经把 VLM hidden state 接到了 trajectory head。

昨天的结构是：

```text
visual tokens + state/text tokens + latent tokens
-> fusion transformer
-> latent hidden
-> trajectory head
-> future waypoints
```

这已经是一个最小 VLA 了。

但是 OneVL 还不止这一步。

OneVL 里很关键的一点是：latent hidden 不只用 trajectory loss 训练。

它还会被辅助任务塑造。

今天我们讲 dual auxiliary。

dual 是两个。

auxiliary 是辅助。

也就是两个辅助任务。

在我们的 toy 版本里，两个辅助任务是：

```text
language auxiliary: reason classification
world auxiliary: future lead distance prediction
```

对应两类 latent：

```text
LANG_LATENT_*
VIS_LATENT_*
```

今天的结构是：

```text
language latent -> language auxiliary decoder
visual/world latent -> world auxiliary decoder
all latent -> trajectory head
```

今天脚本是：

```bash
python day9_onevl_aux/01_dual_aux_staged.py
```

今天你要掌握到什么程度？

第一，知道为什么要把 latent 分成 language latent 和 visual/world latent。

第二，知道 auxiliary loss 不是为了推理时多输出一个东西，而是为了训练 hidden representation。

第三，知道 staged training 的三个阶段分别在训练什么。

第四，能把这个 toy 结构映射到 OneVL。

---

## 第一部分：为什么 trajectory loss 可能不够

我们先问一个问题。

昨天 Day8 已经能用 latent hidden 预测轨迹了。

那为什么还要 auxiliary？

因为只用 trajectory loss，监督信号比较单一。

模型只知道最终轨迹对不对。

但它未必学到清晰的中间结构。

比如一个场景里前车很近，相对速度为负。

正确轨迹应该减速。

trajectory loss 会告诉模型：你预测的点太远了，要短一点。

但它不会显式告诉模型：

```text
这是因为前车太近。
这是因为相对速度为负。
这是 brake_due_to_close_lead 场景。
```

language auxiliary 就是补这个语义监督。

再比如，前车当前距离是 10 米，相对速度是 -3 m/s。

未来 1 秒后可能只有 7 米。

trajectory loss 间接需要这个信息。

但 world auxiliary 可以显式要求模型预测未来前车距离。

这会逼 latent 学动态。

所以 auxiliary 的作用是：

```text
给 latent hidden state 增加额外训练约束。
```

---

## 第二部分：为什么分 LANG_LATENT 和 VIS_LATENT

Day8 里，我们只有一组 latent。

Day9 里拆成两组：

```text
LANG_LATENT_0 ... LANG_LATENT_3
VIS_LATENT_0 ... VIS_LATENT_3
```

language latent 偏语义。

比如：

```text
keep
brake_due_to_close_lead
lane_change_left
lane_change_right
```

visual/world latent 偏未来动态。

比如：

```text
未来前车距离
未来视觉状态
世界变化
```

真实 OneVL 里，language auxiliary 可能是重建语言推理，world auxiliary 可能是预测未来视觉 token。

我们的 toy 版本更简单。

但抽象一样：

```text
不同 latent 接不同辅助任务，让它们学习不同类型的信息。
```

---

## 第三部分：latent_tokens 函数做了什么

脚本里有：

```python
def latent_tokens(sample, lang_latents, vis_latents):
```

它先取 state tokens，但去掉最后 EOS。

然后加 language latent：

```text
LANG_LATENT_0
LANG_LATENT_1
LANG_LATENT_2
LANG_LATENT_3
```

再加 visual latent：

```text
VIS_LATENT_0
VIS_LATENT_1
VIS_LATENT_2
VIS_LATENT_3
```

最后再加 EOS。

所以完整序列类似：

```text
BOS speed_4 dist_2 rel_1 cmd_brake hist0_x_* ...
LANG_LATENT_0 LANG_LATENT_1 LANG_LATENT_2 LANG_LATENT_3
VIS_LATENT_0 VIS_LATENT_1 VIS_LATENT_2 VIS_LATENT_3
EOS
```

Dataset 会找两类位置：

```python
lang_pos = [i for i, t in enumerate(tokens) if t.startswith("LANG_LATENT_")]
vis_pos = [i for i, t in enumerate(tokens) if t.startswith("VIS_LATENT_")]
```

这一步你要和 Day3 连接起来。

Day3 我们讲过特殊 token hidden。

Day8 我们取最后 latent hidden。

今天是更精细地取两组 latent hidden。

核心动作还是：

```text
找到特殊 token 的位置，取 hidden state。
```

---

## 第四部分：forward 里的三路输出

模型先做 embedding 和 backbone：

```python
hidden = self.backbone(self.token(ids) + self.pos(pos), ...)
```

得到：

```text
hidden: [B, T, C]
```

然后取 language latent hidden：

```python
lang_h = hidden[batch, lang_pos]
```

shape 是：

```text
[B, L_lang, C]
```

再取 visual latent hidden：

```python
vis_h = hidden[batch, vis_pos]
```

shape 是：

```text
[B, L_vis, C]
```

trajectory head 用两类 latent：

```python
traj = self.traj_head(torch.cat([lang_h, vis_h], dim=1).mean(dim=1))
```

也就是所有 latent 一起服务主任务。

language auxiliary 只用 lang_h：

```python
reason = self.language_aux(lang_h.mean(dim=1))
```

world auxiliary 只用 vis_h：

```python
world = self.world_aux(vis_h.mean(dim=1))
```

所以输出有三个：

```text
trajectory: [B, 12]
reason_logits: [B, num_reasons]
world_pred: [B, 1]
```

---

## 第五部分：三个 loss 分别塑造什么

训练时有三个 loss。

第一个：

```python
traj_loss = F.mse_loss(traj, traj_y)
```

这是主任务。

它训练模型输出正确未来轨迹。

第二个：

```python
lang_loss = F.cross_entropy(reason, reason_y)
```

这是 language auxiliary。

它训练 language latent 能预测 reason。

reason 是：

```text
keep
brake_due_to_close_lead
lane_change_left
lane_change_right
```

它让 language latent 更有语义结构。

第三个：

```python
world_loss = F.mse_loss(world, world_y)
```

这是 world auxiliary。

world_y 是 future_world_target。

在 toy 里就是未来前车距离。

它让 visual/world latent 学未来动态。

所以三个 loss 的分工是：

```text
trajectory loss: 学会规划
language loss: 学会语义原因
world loss: 学会未来动态
```

---

## 第六部分：auxiliary 不是为了推理时输出解释

这里要特别停一下。

很多人会误解 auxiliary decoder。

他们会以为：既然训练了 language_aux，那推理时是不是要输出 reason？

不一定。

auxiliary decoder 的核心作用是在训练时提供额外梯度。

它通过 loss 影响 backbone 和 latent hidden state。

推理时可以只保留：

```text
backbone + latent hidden + trajectory head
```

把 language_aux 和 world_aux 拿掉。

这不代表 auxiliary 没用。

它已经在训练时把约束写进 latent representation 了。

这点和老师训练学生很像。

训练时要求你写解题步骤，是为了塑造你的思路。

考试时不一定每一步都写出来。

辅助任务也是这样。

---

## 第七部分：staged training 三个阶段

脚本里有三个 stage。

Stage 0：trajectory only。

loss 是：

```python
loss = traj_loss
```

先让模型学会主任务。

也就是先能规划。

Stage 1：auxiliary decoders。

代码里会冻结：

```text
token embedding
position embedding
backbone
trajectory head
```

只训练 auxiliary heads。

loss 是：

```python
loss = lang_loss + world_loss
```

这一阶段的意思是：

先让辅助 decoder 学会从已有 latent hidden 里读出语义和世界信息。

Stage 2：joint fine-tune。

所有模块重新打开。

loss 是：

```python
loss = traj_loss + 0.2 * (lang_loss + world_loss)
```

主任务仍然最重要。

辅助任务作为额外约束。

这就是 staged training。

它不是随便分阶段。

它是在控制学习顺序：

先主任务，后辅助读出，再联合微调。

---

## 第八部分：今天和 OneVL 的对应关系

把 toy 映射到 OneVL：

```text
LANG_LATENT_* -> language latent tokens
VIS_LATENT_* -> visual/world latent tokens
language_aux -> language auxiliary decoder
world_aux -> visual/world auxiliary decoder
traj_head -> planning / trajectory head
stage0/1/2 -> staged training
```

真实 OneVL 的 language auxiliary 可能不是四分类。

可能是更复杂的语言 decoder。

真实 world auxiliary 也可能不是预测一个前车距离。

可能是预测未来视觉 token 或世界状态。

但你先不要被真实复杂度吓到。

它们的抽象就是今天这个 toy：

```text
不同 latent hidden 接不同 decoder，用不同 loss 塑造 latent。
```

---

## 收尾

今天我们完成了 OneVL-style dual auxiliary 的 toy 版本。

你现在应该能讲清楚：

```text
LANG_LATENT -> language_aux -> reason classification
VIS_LATENT -> world_aux -> future world target
LANG + VIS latent -> trajectory head
```

三个训练阶段是：

```text
stage0: trajectory only
stage1: train auxiliary decoders
stage2: joint fine-tune
```

今天最重要的一句话是：

```text
auxiliary decoder 的核心作用是训练时塑造 latent hidden state，而不是推理时多输出一个解释。
```

明天 Day10，我们回到真实 OneVL repo。

你要带着今天这些概念去找：

latent token 在哪里插入。

hidden state 在哪里取。

trajectory head 在哪里。

language/world auxiliary decoder 在哪里。

staged training 脚本在哪里。
