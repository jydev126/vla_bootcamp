# Day 4-10 新路线总览

这份路线替代旧 Day 4-7。旧材料已经删除，新的目录按概念递进重建。

| Day | 目录 | 核心问题 | 主要出处 |
| --- | --- | --- | --- |
| 4 | `day4_vlm` | LLM 如何接收 image token | MiniGPT-4 |
| 4 | `day4_vlm` | projector alignment / instruction tuning 训练什么 | MiniGPT-4 |
| 6 | `day6_act` | action 是什么，为什么要 action chunk | ACT / ALOHA |
| 7 | `day7_openvla_action` | 连续 action 如何变成 token | OpenVLA |
| 8 | `day8_vla_bridge` | VLM hidden state 如何接 action head | MiniGPT-4 + ACT |
| 9 | `day9_onevl_aux` | OneVL 的 language/world dual auxiliary 如何塑造 latent | OneVL |
| 10 | `day10_onevl_study` | 如何读 OneVL repo 并设计 ablation | OneVL |

主线：

```text
Day2 mini GPT:
text -> token ids -> embeddings -> Transformer -> logits

Day4 VLM:
image -> patches -> visual features -> projector -> visual tokens
text -> text tokens
visual + text -> tiny LLM hidden

Day6-8 VLA:
visual/text hidden -> action head 或 action tokens -> trajectory

Day9-10 OneVL:
latent hidden -> trajectory + language aux + world aux
```
