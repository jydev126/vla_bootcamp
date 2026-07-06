# Day 9：OneVL-style dual auxiliary

主要出处：

- OneVL 项目页：https://xiaomi-embodied-intelligence.github.io/OneVL/
- OneVL 代码：https://github.com/xiaomi-research/OneVL

顺序：

```bash
python day9_onevl_aux/01_dual_aux_staged.py
```

递进关系：

```text
language latent -> language auxiliary decoder
visual latent -> world auxiliary decoder
all latent -> trajectory head
stage0 trajectory only
stage1 auxiliary decoders
stage2 joint fine-tune
```
