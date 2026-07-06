# Day 8：把视觉 token 接回 action

顺序：

```bash
python day8_vla_bridge/01_visual_tokens_to_waypoints.py
```

今天只做一件事：把 Day4 的 visual tokens 和 Day6 的 action head 接起来。

```text
image -> visual tokens
state tokens -> text tokens
visual + text + latent -> hidden state
latent hidden -> trajectory
```
