# Day 7：OpenVLA-style action token

主要出处：

- OpenVLA 项目页 / videos：https://openvla.github.io/
- OpenVLA 代码：https://github.com/openvla/openvla

顺序：

```bash
python day7_openvla_action/01_action_tokenizer.py
python day7_openvla_action/02_tiny_action_lm.py
```

递进关系：

```text
01 连续 trajectory -> 离散 action ids -> 近似恢复
02 context tokens -> causal transformer -> action token ids
```
