# Day 6 上课台词：从 action 表示到 ACT

## 开场

今天是 Day 6。

前两天我们一直在讲 VLM。

Day4 讲的是：图像怎么变成 visual tokens，再接到 LLM hidden space。

Day5 讲的是：projector alignment 和 instruction tuning 到底训练什么。

这两天解决的是 VLA 里的 V 和 L。

V 是 vision。

L 是 language。

但 VLA 还有一个最关键的 A。

A 是 action。

今天我们专门讲 action。

你现在可能会有一个模糊感觉：模型看图，理解语言，然后输出动作。

但是“动作”到底是什么？

这件事必须讲清楚。

如果 action 不清楚，后面 OpenVLA、ACT、OneVL 你都会看得很飘。

在聊天机器人里，输出是文字 token。

在自动驾驶或机器人里，输出不是一句话。

输出可能是：

```text
未来轨迹 waypoints
方向盘角度 steering
油门 throttle
刹车 brake
机械臂关节位置
末端执行器位姿
夹爪开合
```

今天我们用 toy driving，把 action 简化成最直观的一种：

```text
未来 6 个 waypoints
```

每个 waypoint 是 `[x, y]`。

所以一个 action chunk 是：

```text
[6, 2]
```

今天参考 ACT 和 ALOHA。

ACT 的一个核心思想是：不要一次只预测一个动作，而是一次预测一段动作，也就是 action chunk。

今天三个脚本：

```bash
python day6_act/01_state_to_waypoints.py
python day6_act/02_token_observation.py
python day6_act/03_tiny_act.py
```

今天你要掌握到什么程度？

第一，知道 action 可以是连续数值，不一定是 token。

第二，知道未来 waypoints 为什么可以作为自动驾驶 action。

第三，知道 action chunk 为什么比单步 action 更稳定。

第四，知道 ACT-style action queries 是什么。

---

## 第一部分：先解释 observation 和 action

在监督学习里，我们常说输入 x，输出 y。

在机器人和自动驾驶里，更常用两个词：

```text
observation
action
```

observation 是系统看到和知道的东西。

比如：

```text
图像
车速
前车距离
导航命令
历史轨迹
机器人关节状态
```

action 是系统要做的事情。

比如：

```text
未来轨迹
控制量
机器人未来动作序列
```

今天 toy driving 的 observation 是结构化状态：

```text
ego_speed
lead_distance
rel_speed
command
history trajectory
```

今天 toy driving 的 action 是：

```text
future waypoints
```

这就是最基本的 imitation learning 形式：

```text
observation -> action
```

模型从数据里学会，在某个观测下应该输出什么动作。

---

## 第二部分：01_state_to_waypoints.py，最朴素的 action head

第一个脚本是：

```text
day6_act/01_state_to_waypoints.py
```

它做的是最朴素的版本：

```text
features -> MLP -> waypoints
```

features 函数会把样本变成 13 维向量。

我们算一下。

ego_speed 一个数。

lead_distance 一个数。

rel_speed 一个数。

command 有四类，one-hot 是 4 个数。

history 有 3 个点，每个点 x、y 两个数，所以是 6 个数。

总共：

```text
1 + 1 + 1 + 4 + 6 = 13
```

输出是 future_to_tensor。

future 里有 6 个 waypoint，每个 `[x, y]`。

flatten 后是：

```text
12 个数
```

所以 MLP 是：

```text
[B, 13] -> [B, 12]
```

这一步看起来简单，但非常重要。

它告诉你 action learning 本质上仍然是 Day1 的监督学习。

```python
pred = model(x)
loss = mse(pred, y)
loss.backward()
optimizer.step()
```

复杂 VLA 只是把 `x` 从 13 维 features 变成了图像 token、文本 token、latent token。

把 `model` 从 MLP 变成了 VLM backbone。

但主任务仍然是：

```text
预测未来动作，使它接近 ground truth action。
```

---

## 第三部分：为什么 action chunk 很重要

现在讲 action chunk。

如果只预测一个点，比如 t+1 的动作，模型每一步都要重新预测。

这会带来两个问题。

第一，误差累积。

如果第一步错了一点，下一步 observation 可能也变了，然后后面越来越偏。

第二，动作不平滑。

逐步预测可能每一步都抖一下。

ACT 的思路是：一次预测未来一段动作。

比如机器人一次预测未来 100ms 或 1s 的动作序列。

在我们的 toy driving 里，就是一次预测 6 个未来 waypoint。

所以 action chunk 是：

```text
[[x1, y1], [x2, y2], ..., [x6, y6]]
```

shape 是：

```text
[B, 6, 2]
```

这个结构比 `[B, 12]` 更有语义。

`[B, 12]` 只是 flatten。

`[B, 6, 2]` 明确告诉你：有 6 个时间步，每个时间步 2 个坐标。

这就是第三个脚本 TinyACT 要做的事情。

---

## 第四部分：02_token_observation.py，为什么 observation 要变成 token

第二个脚本是：

```text
day6_act/02_token_observation.py
```

它把结构化 observation 变成 token sequence。

你可能会问：为什么第一步 MLP 已经能做了，还要把状态变成 token？

因为后面我们要和 LLM/VLM 接起来。

LLM 的接口是 token sequence。

VLM 的接口也是 visual tokens + text tokens。

如果 observation 一直是一个 13 维向量，它和 LLM/VLM 的接口不统一。

所以我们把状态离散化成 token：

```text
BOS
speed_4
dist_2
rel_1
cmd_brake
hist0_x_3
hist0_y_2
...
EOS
```

这和 Day2 的 tokenizer 类似。

Day2 是把自然语言变成 token。

今天是把结构化驾驶状态变成 token。

模型主线是：

```text
input_ids [B, T]
-> embedding [B, T, C]
-> GRU hidden [B, T, C]
-> last_valid_hidden [B, C]
-> head [B, 12]
```

为什么要 last_valid_hidden？

因为 batch 里可能 padding。

如果你直接取 `hidden[:, -1, :]`，最后一个位置可能是 PAD。

所以代码用 mask 算真实长度，然后取每条样本最后一个有效 token。

这和 Day3 的 attention_mask 是同一个思想。

---

## 第五部分：03_tiny_act.py，ACT-style action queries

第三个脚本是：

```text
day6_act/03_tiny_act.py
```

它更接近 ACT。

它不再把整个轨迹 flatten 后一次输出。

它使用 action queries。

模型结构是：

```text
observation tokens -> encoder -> observation memory
action queries -> decoder attends to memory -> action hidden
action hidden -> head -> action chunk
```

我们看关键参数：

```python
self.action_queries = nn.Parameter(torch.randn(1, NUM_WAYPOINTS, hidden_dim) * 0.02)
```

这里 `NUM_WAYPOINTS = 6`。

所以 action_queries 是：

```text
[1, 6, C]
```

expand 到 batch 后：

```text
[B, 6, C]
```

这 6 个 query 不是输入数据给的。

它们是模型学习出来的查询槽位。

每个 query 负责问：

第 1 个未来 waypoint 应该是什么？

第 2 个未来 waypoint 应该是什么？

一直到第 6 个。

observation tokens 先经过 encoder，得到：

```text
memory: [B, T, C]
```

然后 decoder 让 action queries 去 attend memory。

输出：

```text
action_hidden: [B, 6, C]
```

最后线性 head：

```text
[B, 6, C] -> [B, 6, 2]
```

这就是 action chunk。

---

## 第六部分：ACT-style 和普通 MLP head 的区别

普通 MLP head 是：

```text
一个 summary hidden -> 12 个数
```

ACT-style 是：

```text
6 个 action queries -> 6 个 action hidden -> 6 个 waypoint
```

它的好处是结构更明确。

每个未来时间步有自己的 query 和 hidden state。

这让模型更容易学到时间维度上的分工。

你可以把 action queries 理解成“动作输出槽位”。

这和后面 latent token 很像。

latent token 是推理槽位。

action query 是动作槽位。

两者都不是普通输入 token。

它们是模型为了输出或中间推理而保留的可学习位置。

---

## 第七部分：今天和 OpenVLA / OneVL 的关系

今天我们讲的是 continuous action head。

它和 OpenVLA 的 action token 路线不同。

今天是：

```text
hidden -> continuous waypoints
```

明天 Day7 是：

```text
continuous waypoints -> discrete action token ids
context -> causal LM -> action token ids
```

Day8 会回到今天这条 continuous head 路线，把 VLM hidden state 接到 trajectory head。

所以 Day6 是 action 的地基。

没有今天，你后面会不知道 trajectory head 输出的 `[B, 12]` 或 `[B, 6, 2]` 到底是什么意思。

---

## 收尾

今天我们把 VLA 里的 action 讲清楚了。

你应该能解释：

```text
observation 是模型看到的状态。
action 是模型要输出的行为。
在 toy driving 里，action 是未来 6 个 waypoints。
action chunk 是一次预测一段未来动作。
```

三个脚本的关系是：

```text
01: structured features -> MLP -> [B, 12]
02: state tokens -> sequence hidden -> [B, 12]
03: state tokens -> encoder memory, action queries -> [B, 6, 2]
```

今天最重要的一句话是：

```text
ACT-style action queries 给模型提供了专门的动作输出槽位，一次预测整个 action chunk。
```

明天 Day7，我们讲另一种 action 表示：

不是直接回归连续动作，而是把连续动作离散成 token。

这就是 OpenVLA-style action token。
