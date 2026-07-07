# Day 5: 用真实 LLM 搭一个小型 VLM

Day4 用 toy 模型讲清楚了结构和训练目标。Day5 往真实路线迈一步:

```text
image
-> frozen CLIP vision encoder
-> trainable prefix projector
-> visual prefix tokens

prompt + caption
-> frozen DistilGPT2 token embeddings

concat([visual prefix tokens, text tokens])
-> frozen DistilGPT2
-> caption LM loss
```

这里 LLM 不是随机小模型，而是 `distilgpt2`。视觉塔也不是 toy ViT，而是 `openai/clip-vit-base-patch32`。训练数据来自公开的 Flickr30k Hugging Face 数据集 `nlphuji/flickr30k`。

## 自动驾驶 VLM 和普通 VLM 的区别

普通 VLM 常见目标是:

```text
看图 -> 描述图里有什么
看图 + 普通问题 -> 回答对象/颜色/数量
```

自动驾驶更麻烦，因为它不是只要“识别”，而是要支持决策:

```text
看图/多帧/多视角 -> 找关键 road users
估计风险和交互 -> 解释为什么影响 ego planning
结合导航/规则/历史 -> 输出高层决策或未来轨迹
```

所以真正的 driving VLM / VLA 还会遇到这些问题:

```text
1. 时序: 单帧看不出速度、让行、切入趋势。
2. 空间: 前后左右多相机、BEV、3D 距离比普通 caption 更重要。
3. 长尾安全: 少见但危险的场景比平均 caption 分数更重要。
4. 低延迟: 车上不能靠很长的自由生成 chain-of-thought 再控制。
5. 输出接口: 最后不只生成文字，还要能接 action token / trajectory head。
6. 评估: 不能只看 BLEU/CIDEr，要看规划选择、风险识别、闭环表现。
```

Day5 现在分成两套代码:

```text
教学版:
  01_clip_prefix_captioner.py
  02_driving_qa_tuning.py
  03_planning_answer_ranking.py

真实版:
  04_real_driving_vqa_jsonl_tuning.py
```

脚本按这个顺序加深:

```text
01_clip_prefix_captioner.py
普通 image captioning，确认真实 CLIP + 真实 LLM + projector 能训练。

02_driving_qa_tuning.py
driving-scene visual instruction tuning，把 caption 任务改成 road users / risk / planning response QA。

03_planning_answer_ranking.py
候选驾驶答案 ranking，用 answer loss 判断模型是否偏向正确 planning response。

04_real_driving_vqa_jsonl_tuning.py
面向 DriveLM / NuScenes-QA / LingoQA / BDD-X 等公开驾驶数据的真实 JSONL 入口。
```

## 为什么算经典 VLM 路线

这属于 ClipCap / BLIP-2 / MiniGPT-4 一类思路的轻量版:

```text
冻结预训练视觉模型
冻结预训练语言模型
只训练中间 bridge / projector
把图像特征翻译成 LLM hidden space 里的若干 prefix tokens
```

Day4 的 `VisionProjector` 是:

```text
visual_features [B, 64, 64] -> visual_tokens [B, 64, 96]
```

Day5 的 projector 是:

```text
CLIP image embedding [B, 512] -> visual_prefix [B, prefix_length, 768]
```

区别是这里的 768 维就是 DistilGPT2 真实 hidden size。

## 运行

先安装依赖:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

小规模训练:

```bash
.venv/bin/python day5_real_vlm/01_clip_prefix_captioner.py \
  --streaming \
  --train-samples 512 \
  --val-samples 64 \
  --epochs 2 \
  --batch-size 8
```

如果电脑内存比较宽裕，可以去掉 `--streaming`，让 `datasets` 本地缓存后再 shuffle。

训练结束默认只保存 bridge 权重:

```text
common/outputs/day5_real_vlm/prefix_projector.pt
```

复用这个 bridge:

```bash
.venv/bin/python day5_real_vlm/01_clip_prefix_captioner.py \
  --streaming \
  --train-samples 64 \
  --val-samples 16 \
  --epochs 0 \
  --load-path common/outputs/day5_real_vlm/prefix_projector.pt
```

## 你应该观察什么

脚本会打印三类信号:

```text
1. shape trace
   pixel_values -> visual_prefix -> text_embeds -> llm_inputs_embeds

2. generation
   训练前/训练后对同一批图片生成 caption

3. caption-rank probe
   固定几张图片和几条 caption，看模型是否给正确 image-caption pair 更低 loss
```

`caption-rank probe` 比单纯看 loss 更直观: 如果换图后正确 caption 的 loss 也跟着换，说明模型确实在用视觉 prefix，而不是只靠语言先验。

## 02: Driving QA

02 仍然用公开 Flickr30k，但只筛选含 road / street / car / bus / person 等词的真实图片，并从公开 caption 派生轻量 QA:

```text
Question: What road users are visible?
Answer: vehicles and pedestrians

Question: What should the driving assistant pay attention to?
Answer: people near the roadway

Question: What is a reasonable high level driving response?
Answer: slow down and watch for pedestrians
```

运行:

```bash
.venv/bin/python day5_real_vlm/02_driving_qa_tuning.py \
  --streaming \
  --train-samples 256 \
  --val-samples 48 \
  --epochs 2 \
  --batch-size 8
```

训练结束默认保存:

```text
common/outputs/day5_real_vlm/driving_qa_projector.pt
```

这个脚本的重点不是声称已经解决自动驾驶，而是让你看到普通 caption VLM 如何变成视觉指令微调:

```text
image + question -> answer
```

## 03: Planning Answer Ranking

03 加一个更像驾驶系统的观察方式。它不让模型随便生成一句话，而是在候选驾驶响应之间比较 loss:

```text
continue carefully
keep a safe distance from vehicles
slow down and watch for pedestrians
slow down and give the cyclist space
```

运行:

```bash
.venv/bin/python day5_real_vlm/03_planning_answer_ranking.py \
  --streaming \
  --samples 12 \
  --load-path common/outputs/day5_real_vlm/driving_qa_projector.pt
```

如果正确候选的 loss 更低，说明视觉 prefix 真的在影响 LLM 的答案偏好。如果所有图片都偏向同一个答案，说明模型更多在靠语言先验。

## 04: 真实公开驾驶数据入口

默认使用 Flickr30k 是为了小、公开、容易跑。真正面向自动驾驶时，数据层应该换成:

```text
DriveLM / NuScenes-QA / LingoQA / BDD-X
```

这些数据集通常不会像小 demo 一样一条命令就下载完并直接给你 PIL 图片。更实际的做法是先把公开数据预处理成 JSONL:

```json
{"image":"CAM_FRONT/sample.jpg","question":"What should the ego vehicle pay attention to?","answer":"a pedestrian crossing ahead","task":"planning"}
```

多视角样本:

```json
{"images":{"front":"CAM_FRONT/xxx.jpg","front_left":"CAM_FRONT_LEFT/xxx.jpg","front_right":"CAM_FRONT_RIGHT/xxx.jpg"},"question":"What is the safest response?","answer":"slow down","candidates":["continue","slow down","stop"],"task":"planning"}
```

真实版运行:

```bash
.venv/bin/python day5_real_vlm/04_real_driving_vqa_jsonl_tuning.py \
  --train-jsonl data/driving_vqa/train.jsonl \
  --val-jsonl data/driving_vqa/val.jsonl \
  --image-root data/driving_vqa/images \
  --epochs 2 \
  --batch-size 8
```

然后保留类似模型骨架:

```text
frozen or lightly tuned vision encoder
-> projector / Q-Former / resampler
-> LLM hidden space
-> QA loss + planning/action loss + auxiliary perception/world losses
```

也就是说，Day5 先吃透 VLM 的“图像进入 LLM 并接受语言监督”；后面再把监督从 caption/QA 升级成轨迹、动作 token、风险预测和闭环评估。

## 参数量

默认设置下，CLIP 和 DistilGPT2 都被冻结。总参数量大约是两亿多，但真正训练的只有几百万个 projector 参数，所以适合教学机或 Apple Silicon / 单卡 GPU 上做小规模实验。
