# Day 6：从 action 表示到 ACT

主要出处：

- ALOHA 项目页 / videos：https://tonyzhaozh.github.io/aloha/
- ACT 代码：https://github.com/tonyzhaozh/act

顺序：

```bash
python day6_act/01_state_to_waypoints.py
python day6_act/02_token_observation.py
python day6_act/03_tiny_act.py
```

递进关系：

```text
01 action 是未来 waypoints
02 observation 变成 token sequence
03 ACT-style action queries 一次预测 action chunk
```
