# Day 10：OneVL repo map 与实验清单

主要出处：

- OneVL 项目页：https://xiaomi-embodied-intelligence.github.io/OneVL/
- OneVL 代码：https://github.com/xiaomi-research/OneVL

## Toy 到 OneVL 映射

| toy | OneVL |
| --- | --- |
| `day4_vlm/03_minigpt4_bridge.py` | visual feature projector |
| `day8_vla_bridge/01_visual_tokens_to_waypoints.py` | VLM hidden state -> trajectory head |
| `LANG_LATENT_*` | language latent tokens |
| `VIS_LATENT_*` | visual/world latent tokens |
| `language_aux` | language auxiliary decoder |
| `world_aux` | visual/world auxiliary decoder |
| staged training | OneVL three-stage training |

## 阅读顺序

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

## 最小 ablation

```text
1. trajectory only
2. + language auxiliary
3. + world auxiliary
4. + language + world auxiliary
5. + staged training
```
