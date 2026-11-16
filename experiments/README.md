# experiments/ — MI300X 参数搜索与 MoE 归因的工作区

这些脚本原本放在仓库**之外**（计划 §2.3 要求"不要污染 Fleet 的代码"）。
在实例即将销毁时被**移进仓库**，以免随镜像一起丢失。**这是一个有意的偏离**：
移入的是纯新增目录，**没有改动任何 Fleet 的 kernel 或 python 源码**。

## 谁是谁

| 文件 | 作用 |
|---|---|
| `setup_mi300x_env.sh` | **一键环境引导**（6 处修复 + 依赖 + 编译 + 冒烟）。镜像重建后先跑这个 |
| `param_search.py` | Part A 参数搜索 driver。清缓存、设环境、跑 demo、解析、追加 TSV。崩溃与超时也记一行，不丢样本 |
| `summarize.py` | 合并跨 stage 的同一配置，给出噪声下限；`vs best` 只在同一 batch size 内比较 |
| `smoke_test.sh` | 单配置冒烟（计划的 §1.6 闸门） |
| `correctness_check.sh` | PyTorch vs MPK 输出比对（有效性闸门） |
| `find_ck_commit.py` | 二分定位 composable_kernel 的匹配提交（方法见移植指南 §2.1） |
| `moe_routing_stats.py` | **B1/B2**：每专家 token 直方图与行利用率（PyTorch hook，不依赖 Fleet 跑通） |
| `moe_roofline.py` | MoE 访存下限与算术强度 |
| `moe_smoke.sh` / `b4_moe.sh` | MoE 冒烟与 B4 对照 |
| `results/results.tsv` | Part A 的 41 行原始结果（每次运行一行，仅追加） |
| `results/moe_routing_bs{1,32}.json` | B1/B2 的路由统计原始结果 |
| `results/A-part.md` | Part A 结论（英文版工作稿；中文版在 `docs/results/`） |
| `MI300X-ADAPTATION-NOTES.md` | 环境事实、阻塞清单、CK 定位论证、指标陷阱（英文工作稿） |
| `attention_share.md` | attention 占比及其对参数搜索的稀释效应 |
| `logs/` | 每次运行的完整 stdout+stderr（占位；大文件未入库） |

> 中文交付文档在 **`docs/results/`**：执行情况、Part B 归因、Part B 修复补遗。
> 移植步骤在 **`docs/MI300X-移植指南.md`**。

## 用法

```bash
bash experiments/setup_mi300x_env.sh          # 一次性环境引导

export AMDGPU_TARGETS=gfx942
export MIRAGE_HOME=$(pwd) MIRAGE_ROOT=$(pwd)
export PATH=/root/venvs/fleet/bin:$PATH

cd experiments
python param_search.py --stage A2 --dry-run   # 先看要跑什么
python param_search.py --stage A0,A1 --repeat 2
python summarize.py
```

环境开关：
- `FLEET_CK_FMHA`（默认 `0`）—— 见移植指南 §4.3。设为 `1` 需要先解决 gfx942 的
  decode attention 移植。
- `DEBUG_MOE_MASK=1` —— 打开 MoE 路径的逐层插桩（默认完全惰性）。
- `FLEET_ROOT` / `FLEET_PYTHON` / `FLEET_RESULTS` / `FLEET_LOGS` —— 路径覆盖。

## 解读数据时必须记住的两条

1. **`ms_per_token` 不是 decode-only TPOT**，而是 prefill+decode 合起来的每次迭代平均时间
   （`demo.py:1316-1326`）。同一 batch size 内可比，**跨 batch size 不可比**。
2. 所有 Part A 数据都是 `USE_CK_FMHA=0`。attention 约占 bs=32 迭代的 **66%**，
   会把 gang 的相对收益稀释约 3 倍。详见 `attention_share.md`。
