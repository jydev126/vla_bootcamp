# Day 10 上课台词：OneVL repo map 与实验清单

## 开场

今天是 Day 10。

这是新版 Day4 到 Day10 的收束课。

前面几天我们不是随便写 toy。

每一天都在为读 OneVL 做准备。

我们先把这条路线完整回顾一遍。

Day4，我们讲了 VLM 的结构：

```text
image -> patches -> visual features -> projector -> visual tokens
```

Day5，我们讲了 MiniGPT-4 风格训练：

```text
projector alignment
instruction tuning
```

Day6，我们讲了 ACT：

```text
action chunk
action queries
continuous action head
```

Day7，我们讲了 OpenVLA：

```text
continuous action -> action token ids -> causal LM
```

Day8，我们把 VLM 和 action 接起来：

```text
visual tokens + state tokens + latent -> hidden -> trajectory
```

Day9，我们讲了 OneVL-style auxiliary：

```text
language latent -> language auxiliary
visual/world latent -> world auxiliary
all latent -> trajectory
```

今天要做的是：把这些 toy 概念映射到真实 OneVL repo。

今天不是让你立刻完整复现 OneVL benchmark。

这不现实。

真实 repo 会有环境、数据、权重、评测脚本、显存、路径配置等很多问题。

今天的目标更具体：

你要知道读 OneVL 时该按什么顺序读。

你要知道每个 toy 模块在真实代码里应该找什么。

你要知道最小 ablation 怎么设计。

这就是 Day10。

---

## 第一部分：为什么读 repo 不能从头翻到尾

很多人读大 repo 的方式是：

打开文件夹，从第一个文件看到最后一个文件。

这样很容易迷路。

OneVL 这种 repo 不是小脚本。

里面可能有：

```text
model construction
inference script
training script
dataset
processor
tokenizer
visual encoder
projector
latent token
planning head
auxiliary decoder
evaluation
```

如果你没有地图，很快就会不知道自己在看什么。

所以今天我们用 toy 到 OneVL 的映射表做地图。

读 repo 不是看所有文件。

读 repo 是带着问题找路径。

---

## 第二部分：toy 到 OneVL 映射

先看第一组映射。

`day4_vlm/03_minigpt4_bridge.py` 对应 OneVL 里的：

```text
visual feature projector
```

也就是把视觉特征投影到语言模型 hidden space 的模块。

你读 OneVL 时，如果看到 `projector`、`mm_projector`、`visual_projector`、`vision_proj` 之类的名字，要马上想到 Day4。

第二个映射：

`day8_vla_bridge/01_visual_tokens_to_waypoints.py` 对应：

```text
VLM hidden state -> trajectory head
```

也就是多模态 hidden state 如何接规划输出。

你要找真实 OneVL 里 trajectory 是从哪里来的。

它是从 generate 文本解析出来的吗？

还是从 hidden state 接 head 输出的？

第三个映射：

```text
LANG_LATENT_* -> language latent tokens
VIS_LATENT_* -> visual/world latent tokens
```

你要在真实 repo 里找 latent token 的定义、数量、插入位置。

第四个映射：

```text
language_aux -> language auxiliary decoder
world_aux -> visual/world auxiliary decoder
```

你要找它们接的是哪一类 hidden state，用的是什么 loss，推理时是否保留。

第五个映射：

```text
staged training -> OneVL three-stage training
```

你要找训练脚本里 stage 是怎么切换的。

哪些模块 freeze。

哪些 loss 打开。

loss weight 是多少。

---

## 第三部分：推荐阅读顺序

Day10 README 给的顺序是：

```text
1. README / project page
2. inference script
3. model construction
4. latent token insertion
5. hidden state extraction
6. trajectory head
7. auxiliary decoders
8. staged training scripts
```

我们逐个讲。

第一，README / project page。

先读项目想解决什么。

不要一上来钻到某个模型文件里。

你要先知道 OneVL 的贡献是什么。

它是不是强调 latent reasoning？

是不是强调 language auxiliary？

是不是强调 world auxiliary？

是不是强调三阶段训练？

第二，inference script。

推理脚本告诉你真实输入输出是什么。

你要找：

```text
图像或视频怎么读
prompt 或 state 怎么构造
model 怎么 load
processor/tokenizer 怎么用
输出 trajectory 从哪里来
```

第三，model construction。

模型构造告诉你 backbone、projector、latent、head 怎么接。

你要画结构图。

第四，latent token insertion。

这是重点。

搜索：

```text
latent
num_latent
num_latent_vis
special_tokens
```

你要知道 latent token 是文本 special token，还是可学习 parameter，还是两者结合。

第五，hidden state extraction。

搜索：

```text
hidden_states
last_hidden_state
latent_positions
gather
index_select
```

你要找到：

```text
hidden: [B, T, C]
latent_h: [B, L, C]
```

第六，trajectory head。

你要找 trajectory 输出 shape。

是 `[B, N, 2]`？

还是 flatten 的 `[B, 12]`？

还是更复杂的控制量？

第七，auxiliary decoders。

你要看 language auxiliary 和 world auxiliary 分别接什么 hidden，预测什么目标。

第八，staged training scripts。

你要看训练阶段怎么切换。

这一步决定你是否真的理解论文里的方法。

---

## 第四部分：读代码时不要只搜关键词，要追数据流

你可以搜这些关键词：

```text
projector
vision_tower
visual_encoder
latent
num_latent
num_latent_vis
hidden_states
last_hidden_state
trajectory
planning
aux
language
world
decoder
stage
freeze
requires_grad
```

但是搜到关键词只是开始。

你要继续问五个问题。

第一，这个变量的 shape 是什么？

第二，它从哪里来？

第三，它传到哪里去？

第四，它参与哪个 loss？

第五，推理时还在不在？

举个例子。

如果你搜到 `hidden_states[-1]`，不要只说“这里取最后一层 hidden”。

你要继续找：

这个 hidden 是 `[B, T, C]` 吗？

T 里包含 image tokens 吗？

latent positions 怎么找？

取出的 latent_h 是 `[B, L, C]` 吗？

它送到 trajectory head，还是 auxiliary decoder？

这才叫读懂。

---

## 第五部分：最小 ablation 怎么设计

Day10 README 给了最小 ablation：

```text
1. trajectory only
2. + language auxiliary
3. + world auxiliary
4. + language + world auxiliary
5. + staged training
```

我们逐个解释。

第一个，trajectory only。

这是 baseline。

没有 baseline，后面所有结论都没有意义。

你必须知道，只用主任务能做到什么程度。

第二个，加 language auxiliary。

观察它是否改善语义相关失败。

比如：

```text
前车太近但没刹车
变道方向错误
导航命令理解错
```

第三个，加 world auxiliary。

观察它是否改善动态相关失败。

比如：

```text
前车未来距离估计错误
动态障碍物处理不好
遮挡后风险判断差
```

第四个，同时加 language + world auxiliary。

看它们是否互补。

第五个，加 staged training。

比较 staged training 是否比直接 joint training 更稳定。

注意，ablation 一次只改一个因素。

如果你同时改模型大小、latent 数量、loss weight、训练阶段，就不知道提升来自哪里。

---

## 第六部分：实验记录不要只写 loss

OneVL 这种任务不能只看一个 loss。

你至少要记录：

```text
trajectory metric
language aux accuracy / loss
world aux loss
inference latency
failure cases
```

尤其是 failure cases。

平均指标会掩盖很多问题。

比如两个模型平均误差差不多。

一个经常在前车刹车场景失败。

另一个经常在变道场景失败。

这两个模型的改进方向完全不同。

所以 failure case 要分类：

```text
front vehicle brake
lane change wrong direction
curve
occlusion
multi-agent interaction
navigation command mismatch
```

每个 case 保存：

```text
输入图像
输入 prompt / state
GT trajectory
pred trajectory
模型解释或辅助输出
错误类型
猜测原因
下一步改动
```

这样你才是在做实验。

否则只是跑命令。

---

## 第七部分：今天的验收问题

今天结束，你要能回答这些问题。

第一，OneVL 里 visual projector 对应我们哪份 toy 代码？

答案是：

```text
day4_vlm/03_minigpt4_bridge.py
```

第二，VLM hidden state 接 trajectory head 对应哪份 toy 代码？

答案是：

```text
day8_vla_bridge/01_visual_tokens_to_waypoints.py
```

第三，language latent 和 visual latent 对应哪份 toy 代码？

答案是：

```text
day9_onevl_aux/01_dual_aux_staged.py
```

第四，auxiliary decoder 推理时一定要保留吗？

答案是不一定。

它主要在训练阶段塑造 latent hidden state。

第五，最小 ablation 怎么排？

答案是：

```text
trajectory only
+ language aux
+ world aux
+ both aux
+ staged training
```

---

## 收尾

今天我们完成了新版 Day4 到 Day10 的闭环。

你现在看 OneVL，不应该再觉得它是一团黑盒。

你可以把它拆成：

```text
视觉特征
projector
文本 / 状态 token
latent token
hidden state extraction
trajectory head
language auxiliary
world auxiliary
staged training
```

你现在还不是精通 OneVL。

但你已经有能力读主要代码路径。

你能画 shape。

你能找 loss。

你能设计 ablation。

最后记住一句话：

```text
读 VLA repo，不要只问模型输出什么；要问 hidden state 被谁塑造，又被谁使用。
```

这句话就是 Day10 的核心。
