# MI300X 上复现与测试 Fleet：执行计划

日期：2026-11-16
**读者：接手的 Claude 窗口。本文是自包含的**——不需要先读其他文档就能执行。
**背景：想研究面向Chiplet GPU(以MI300X为例)的MegaKernel Generation。目前已经找到Fleet作为Baseline，对其进行复现和分析。**
本文只做两件事：**A. 跑通 Fleet 并用静态参数搜索测出优化空间**；**B. 复现 Fleet 的 MoE 路径并做瓶颈归因**。

上游材料（需要细节时才读）：
- `plan/moe-plan-v2.md` —— MoE 研究主线的完整论证
- `plan/fleet-analysis-and-research-opportunities.md` —— Fleet 的代码级分析（哪些是硬编码、哪些是桩函数）
- `index.md` —— 所有文档的定位与状态
代码仓库:
- `projects/chiplet-oriented-mka/fleet-chiplet-megakernel` —— 已从官方原仓库Fork至我的GitHub仓库；仓库链接：https://github.com/wqd008/fleet-chiplet-megakernel.git
本地GPU卡：
- MI300X，已经包含pytorch和ROCm，请根据fleet的需求安装依赖
---

## 0. 需求说明

### 0.1 我们要回答的两个问题

**问题 A**：Fleet 的芯粒调度参数是**手工输入的常量**（`GANG_TILE_N=64`、`GANG_M_TILES=bs//16`、`GANG_WGM=0`）。**只做静态参数搜索，和对参数进行设计空间探索，他们之间的性能差距有多少？**

> 这一问的动机来自对 Fleet 代码的分析：它的 cost model `select_strategy()` 是一个**永不返回动态结果的桩函数**（只做整除性检查后 `return "gang"`），几何全部写死，而且它**从未调优过 `GANG_WGM`**（源码里恒为 0）。也就是说，Fleet 把 muGraph 的搜索器换成了手写常数。**我们要测的就是这个替代损失了多少。**这决定了我们做后续优化还有什么空间。

**问题 B**：Fleet 的 MoE 路径（`USE_GANG_MOE=1`）在真机上表现如何？瓶颈在哪？

> 动机：MoE kernel有其稀疏和动态性，而Fleet原文报告的实验结果主要针对于稠密的小attention应用，并且其固化的参数似乎对于MoE来说，优化效果大打折扣。例如，`MPerBlock` 是**编译期常量 16**，而每个专家实际收到的 token 数是运行期才知道的数据相关量。Mixtral 规模下 B=8 时每专家平均只有 2 个 token——如果这个错配是真的，就有约 8× 的浪费。**这一问要确认它在真机上到底是否成立。**另外，不要只局限于这里提到的一些MoE相关问题，请你自己仔细去分析，若我们将Fleet作为MegaKernel Generation的Baseline，在MoE Benchmark上还有什么问题。

### 0.2 三个交付物

1. **一份修改后的代码仓库**: 原Fleet是针对MI350的，请你先在MI300X上测试是否能够跑通，如果不能，则修改相应代码完成适配。
2. **一个参数搜索驱动脚本**（小工具，不改 Fleet 的 kernel）：给定参数组合，清缓存、设环境变量、跑 demo、解析结果、记录到 TSV。
3. **两份数据表 + 一份归因结论**：A 的搜索表与结论；B 的 MoE 基线表与瓶颈判断。

### 0.3 硬约束

- 机器镜像：ROCm7.2.4, ubuntu24.04, PyTorch 2.10.0， AMD Developer官方镜像, 包含一块MI300X 192GB。
- **每次改参数都要重新编译 megakernel，单次 3–10 分钟**（README 数据）。所以搜索必须**有界、分阶段、可中断**。
- 不改 Fleet 的 kernel 实现，只修需要从MI350适配到MI300X的一些代码，不改影响其性能的kernel代码。参数搜索只走**环境变量**这一条既有通道。

---

## 1. 环境与前置检查（**先做这几项，其中第 3 项会直接决定成败**）

### 1.1 拿代码

本地已有一份 checkout：`projects/fleet-chiplet-megakernel`（含 `python/`、`src/`、`demo/`、`README.md`）。

```bash
cd fleet-chiplet-megakernel
git rev-parse HEAD > /tmp/fleet_head.txt
export MIRAGE_HOME=$(pwd)
pip install -e . -v
```

### 1.2 ⚠ 必须显式设置编译目标架构（**不设会失败**）

`python/mirage/mpk/persistent_kernel.py:350` 写的是：

```python
amdgpu_target = os.environ.get("AMDGPU_TARGETS", "gfx950")   # ← 默认值是 MI350
...
f"--offload-arch={amdgpu_target}"
```

**默认值 `gfx950` 是 MI350 的架构。MI300X 是 `gfx942`。** 不覆盖这个变量，编译产物将无法在你的卡上运行。**每一步运行前都要 export：**

```bash
export AMDGPU_TARGETS=gfx942
```

### 1.3 可用的 gang 路径（**好消息，已核对源码**）

`persistent_kernel.py:425-427`：

```python
if self.is_rocm:
    # MI300 (gfx94x): use 94 as target_cc for MPK task selection
    self.target_cc = 94
```

**任何 ROCm 构建下 `target_cc` 恒为 94**（不管是 gfx942 还是 gfx950），而所有 gang 层的门禁都是 `assert self.target_cc == 94`。**所以 gang 路径在 MI300X 上会正常启用，不需要改代码。** 这是本计划能成立的前提。

### 1.4 基线环境摸底（5 分钟，必须做）

```bash
python3 -c "
import torch; print('torch', torch.__version__, 'hip', torch.version.hip)
p = torch.cuda.get_device_properties(0)
print('name', p.name, '| CUs', p.multi_processor_count, '| mem GB', p.total_memory/2**30)
"
rocminfo | grep -E "gfx|Marketing Name" | head
```

记录：**实际 CU 数、显存、gfx 型号**。MI300X 应为 304 CU / 192 GB / gfx942；MI350 是 248 CU。**CU 数不同意味着每 XCD 的 worker 数不同（MI300X 约 38，MI350 约 31），这会直接影响"每 XCD 有多少 worker 分同一份权重"的算术，必须在报告里写明。**

### 1.5 模型获取（注意镜像环境）

demo 默认用 `huggingface_hub` 下载。在官方环境里 HF 可能不通。两个办法，任选：

- 用魔搭：`pip install modelscope && modelscope download --model Qwen/Qwen3-8B --local_dir <path>`，然后传 `--model-path <path>`；
- 或预先通过其他途径把模型落到本地盘，用 `--model-path`。
- 或采用镜像地址的办法，阿里/清华镜像均可。

**先确认能用哪个，不要卡在下载上。**

### 1.6 单次冒烟测试（在开始任何搜索前）

```bash
export AMDGPU_TARGETS=gfx942
rm -rf permanent_output_dir
HIP_VISIBLE_DEVICES=0 USE_GANG=1 USE_CK_FMHA=1 USE_FUSED_SILU=1 GANG_M_TILES_GATEUP=1 \
  python3 demo/qwen3/demo.py --use-mirage \
  --model Qwen/Qwen3-8B \
  --max-num-batched-tokens 1 --max-num-batched-requests 1 \
  --max-seq-length 1200 --max-new-tokens 1024 --ignore-eos \
  --prompt "Explain the key differences between RISC and CISC processor architectures, including their instruction set design philosophies, pipeline implementations, memory access patterns, register file organization, microcode usage, and how modern processors have evolved to blur the traditional boundaries between these two approaches in terms of performance, power efficiency, and overall transistor utilization."
```

**成功标志**：输出里有一行

```
Decode: 1141 tokens in 1141 iterations ≈ XXXX ms total (Y.YYY ms/token)
```

**`Y.YYY` 就是全部后续实验的唯一指标（decode-only TPOT, ms/token）。**

**失败时的排查顺序**：① `AMDGPU_TARGETS` 是否设成 gfx942；② 是否残留了上一轮的 `permanent_output_dir`（**必须每次删**）；③ `mpi4py` 是否装了（`world_size=1` 时可缺省）；④ 显存是否够。

> ⚠ **不要跳过这一步去直接搜参数。** 单次编译 3–10 分钟，如果冒烟测试都跑不通，后面的搜索会浪费几小时。

---

## 2. 参数搜索模块设计

### 2.1 搜什么（已从源码枚举，这是可搜的完整集合）

Fleet 的参数全部通过**环境变量 + CLI** 进入，**没有运行期可调项**——也就是说每个配置点都要重编译。可搜项：

| 参数 | 来源 | 默认 | 取值范围与约束 |
|---|---|---|---|
| `GANG_TILE_N` | `demo/qwen3/demo.py:464` | 64 | 必须满足 `(output_size/8) % tile_n == 0`。Qwen3-8B：`output_size/8 ∈ {512(qkv), 3072(gate_up), 512(down)}` ⇒ tile_n 必须是 512 的因数。**候选：32 / 64 / 128 / 256**（512 会只剩 1 个 tile，无用） |
| `GANG_M_TILES` | `demo.py:471` | `max(1, bs//16)` | 必须整除 batch。**候选：1 / 2 / 4**（bs=32 时） |
| `GANG_WGM` | `demo.py:465` | **0** | HipKittens 窗口高度。**`0` = 全 M-major；有效范围 `1..m_tiles`。这是从未被调过的那个量。** 候选：0 / 1 / 2 / 4 |
| `USE_GANG_M_SPLIT` | `demo.py:462` | 0 | 0/1，改走 `gang_linear_msplit_layer` |
| `USE_GANG_N_TILE` | `demo.py:461` | 0 | 0/1. **注意这也是个"从未调过"的量。** 源码注释说 decode 时它**不应**带来权重复用收益——跑它是一次**阴性对照检验**：如果它也变快，说明我们测到的是别的东西 |
| `GANG_M_TILES_GATEUP` | `demo.py:473` | 与 `GANG_M_TILES` 相同 | 给 gate_up 单独设 M 切分。**已有先例说明"不同 op 需要不同几何"** |
| `GANG_K_SPLITS` | `demo.py:463` | 4 | 仅 `USE_GANG_SPLITK` 路径 |
| `USE_CK_FMHA` / `USE_FUSED_SILU` | `demo.py` | — | 0/1，控制 attention 与 silu 融合 |

### 2.2 搜索策略（**有界、分阶段、可中断**）

**不要做笛卡尔积**——上面几个维度的组合远超可行预算。按下面的阶梯走，**每一阶段都可能提前结束**：

```
阶段 A0  可行性（3 个配置，~30 分钟）
  跑 GANG_TILE_N ∈ {64, 128, 256}，其余取默认
  目的：探出哪些取值能编译通过、不崩溃
  ← 如果只有 64 能跑，整个 A 部分提前降级为"参数空间的实际自由度很小"

阶段 A1  基线（2 个配置，~20 分钟）
  (1) 不用 gang：USE_GANG=0（= Mirage/MPK 基线）
  (2) gang 默认：GANG_TILE_N=64, GANG_M_TILES=bs//16, GANG_WGM=0
  目的：复现 README 的加速比，确立本机的对照锚点

阶段 A2  主扫描：tile_n × wgm（最多 12 个配置，~2 小时）
  固定 bs=32, m_tiles=2
  GANG_TILE_N ∈ {64, 128, 256} × GANG_WGM ∈ {0, 1, 2, 4}
  ← 这是最有希望的一个面：tile_n 决定每 XCD 的 tile 数（必须 ≥ worker 数才能喂满），
     wgm 决定 L2 复用窗口

阶段 A3  第二面：m_tiles（最多 4 个配置，~40 分钟）
  取 A2 最优的 tile_n 与 wgm，扫 GANG_M_TILES ∈ {1, 2, 4}
  （bs=32 时三者都整除）

阶段 A4  batch 延展（6 个配置，~1.5 小时）
  bs ∈ {1, 32, 64} × {gang 默认, A2/A3 最优}
  目的：看最优参数是否随 batch 移动。**如果移动了，这就是"需要搜索而非常数"的直接证据**

阶段 A5  阳性/阴性对照（2 个配置，~20 分钟）
  USE_GANG_N_TILE=1（预期不应有收益）
  GANG_M_TILES_GATEUP 与 GANG_M_TILES 取不同值（预期可能有收益）
```

**总计约 25–30 个配置，6–8 小时**（含编译）。每个配置**重复 2 次**取中位（README 自己说热状态会造成 5–15% 波动）。

### 2.3 驱动脚本规格

写一个 `param_search.py`（放在 Fleet 仓库根目录之外的地方，**不要污染 Fleet 的代码**）。要求：

```python
CONFIG = {
    "env":   {"USE_GANG": "1", "GANG_TILE_N": "128", "GANG_WGM": "2", ...},
    "args":  ["--max-num-batched-tokens", "32", "--max-new-tokens", "1024", ...],
    "tag":   "tile128_wgm2_bs32",
}

def run_one(cfg) -> dict:
    # 1. shutil.rmtree("permanent_output_dir", ignore_errors=True)   ← 必须
    # 2. env = {**os.environ, "AMDGPU_TARGETS": "gfx942", **cfg["env"]}
    # 3. subprocess.run(cmd, cwd=FLEET_ROOT, env=env, timeout=3600, capture_output=True)
    # 4. 正则解析 "Decode: N tokens in N iterations ≈ X ms total (Y ms/token)"
    # 5. 返回 {tag, ms_per_token, compile_ok, seconds, stdout_tail, exit_code}
    # 6. 每次 append 到 results.tsv —— 崩溃/超时也要写一行，不能丢样本
```

**必须记录的三类字段**：性能（`ms_per_token`）、成本（该配置的墙钟秒数，含编译）、成败（`exit_code` / 是否命中 assert / stdout 尾部 20 行）。**编译失败的配置要保留在表里**——"哪些参数不可用"本身就是结论。

**健壮性要求**：单配置超时 60 分钟即中止并记一行；连续 3 次编译失败则停止整个阶段并报告。**不要让它无人值守跑一整夜**——每阶段结束看一下结果，因为后面阶段的取值依赖前面的。

---

## 3. Part B：Fleet 的 MoE 复现与瓶颈归因

### 3.1 入口

代码里有 `demo/qwen3/demo_30B_A3B.py`（Qwen3-30B-A3B）。它的开关（已核对源码）：

```python
USE_GANG     = is_rocm and int(os.environ.get("USE_GANG", "0")) == 1
USE_GANG_MOE = USE_GANG and int(os.environ.get("USE_GANG_MOE", "0")) == 1   # ← MoE gang 的总开关
GANG_TILE_N  = int(os.environ.get("GANG_TILE_N", "64"))
GANG_WGM     = int(os.environ.get("GANG_WGM", "0"))
GANG_M_TILES = max(1, args.max_num_batched_tokens // 16) if USE_GANG else 1
```

**注意 `USE_GANG_MOE` 默认 0**。要测 MoE gang，必须显式打开。

### 3.2 模型规模核对

**Qwen3-30B-A3B（bf16）约 60 GB，MI300X 的 192 GB 放得下**，单卡可跑。但**它是"极小专家"**：专家权重 3.15 MB/个，per-XCD 切片仅 0.39 MB——**远小于 4 MB L2**。所以：

> **它跑得好，不能推出"MoE gang 成立"。** 必须至少再加一个**中等专家**的模型作对照。

先检查候选模型能否过 Fleet 的硬门禁（在 `gang_moe_w13_linear_layer` 里）：

```python
assert output_size % tile_n == 0            # tile_n 默认 64
assert reduction_size % 256 == 0            # W13 的 K 维
```

- Qwen3-30B-A3B：`2*inter = 2*768 = 1536`，`1536 % 64 == 0` ✓；`reduction_size = hidden = 2048`，`2048 % 256 == 0` ✓
- Mixtral-8x7B：`2*14336 = 28672`，`28672 % 64 == 0` ✓；`reduction_size = 4096`，`4096 % 256 == 0` ✓（**待核实 Mixtral 的确切 hidden/inter 配置**）

**先跑 Qwen3-30B-A3B，再尝试 Mixtral 级。** 若 Mixtral 因显存或门禁跑不动，**明确报告这一点**，它是"Fleet 的 MoE 只在极小专家上验证过"的证据。

### 3.3 归因要测的量（按信息量排序）

| # | 测什么 | 怎么测 | 回答什么问题 |
|---|---|---|---|
| **B1** | 每专家实际 token 数直方图 | 在 routing 之后 dump `moe_routing_indices`（host 侧），或加一个轻量计数 kernel。跑不同 batch 各一次 | **P1 错配有多大？** 均值 vs 方差 vs 最大值。最热专家是多少 token？ |
| **B2** | M-tile 行利用率 | 有效 token 行数 / 处理的 16 行。可由 B1 推算，或设备端计数 | **P1 的直接量化**：浪费了多少计算与访存 |
| **B3** | 每 XCD 的处理耗时分布 | **代码已内建**：`MPK_ENABLE_DEVICE_TASK_TIMING` 宏打开后，`gang_moe_linear_mi300.cuh` 里有 `s_memrealtime()` 计时与 printf | **P6**：固定的 `(num_experts+7)//8` 轮询是否造成 XCD 间尾延迟 |
| **B4** | gang-MoE vs 非 gang MoE | `USE_GANG_MOE=1` vs `USE_GANG=1 USE_GANG_MOE=0` | **MoE gang 到底有没有净收益？** 这是最基础的一问，README 没有回答 |
| **B5** | 对照组：Qwen3-30B-A3B vs Mixtral | 同一套配置换模型 | 收益是否只在极小专家上存在 |

**B1/B2 是本轮最重要的两个量。** 如果 B1 显示每专家 token 数普遍 ≪ 16（Mixtral 规模下 B=8 时均值只有 2），那么"几何与数据相关基数错配"这个判断就在真机上坐实了，后面所有设计才有立足点。

### 3.4 B 部分的一个前置诚实性说明

在开始 B 之前先记下这条，避免后面误读：

> **MI300X ≠ MI350。** MI350 是 248 CU / 5.3 TB/s；MI300X 是 304 CU / 约 3.7 TB/s。**绝对时间不可与 README 直接比。** 可比的是：**① 相对加速比（gang vs 非 gang）；② 加速比随 batch 的变化趋势；③ 每 XCD 的缓存容量与带宽特征。**

---

## 4. 结果交付格式

### 4.1 表格一：Part A 参数搜索结果（TSV）

```
tag                bs  tile_n  m_tiles  wgm  msplit  ntile  ms_per_token  seconds  exit_code  note
baseline_mpk       32  -       -        -    0       0      12.5          420      0          USE_GANG=0
gang_default       32  64      2        0    0       0      11.2          460      0          README 对照
tile128_wgm2_bs32  32  128     2        2    0       0      ...           ...      0
...
```

### 4.2 表格二：Part B MoE 归因

```
model              use_gang_moe  bs  ms_per_token  每专家 token(均值/中位/最大)  行利用率  尾延迟比(最慢 XCD / 最快 XCD)
Qwen3-30B-A3B      0             32  ...
Qwen3-30B-A3B      1             32  ...
Mixtral-8x7B       0             32  ...
Mixtral-8x7B       1             32  ...
```

### 4.3 结论要回答的四个问题

1. **静态参数搜索的空间有多大？** 最好配置比 Fleet 默认快多少？**这个差距在重复测量的波动（README 说 5–15%）之内还是之外？**
2. **最优参数是否随 batch / 模型移动？** 移动了 ⇒ 常数不成立，需要搜索；不移动 ⇒ Fleet 的手选常数是合理的，第一部分的价值要降级。
3. **MoE gang 有没有净收益？** 在 Qwen3-30B-A3B 上、在 Mixtral 级模型上分别是什么结果？
4. **每专家 token 数与 `MPerBlock=16` 的错配有多大？** 行利用率是多少？这决定"几何档位"这条路线是否值得做。

### 4.4 必须要保留的负面结果

- **任何编译失败/崩溃的配置**：它们的参数取值要写清楚，这是"参数空间实际自由度"的证据；
- **`USE_GANG_N_TILE=1` 的结果**：如果它也变快，说明测到的不是权重复用，**必须重新解释 A 部分的所有结论**；
- **如果某个阶段的收益全在波动范围内**：直接写"未观测到超出噪声的收益"，不要挑一个好看的数报出来。

---

## 5. 执行顺序总表

```
准备                                                         预计耗时
 0. 摸底：CU 数 / 显存 / gfx 型号 / 模型下载通路                10 min
 1. 冒烟测试通过（含 AMDGPU_TARGETS=gfx942）                   15 min
 2. 写 param_search.py 驱动                                    60 min

Part A
 3. A0 可行性：tile_n ∈ {64,128,256}                           30 min
 4. A1 基线：MPK vs gang 默认                                  20 min
 5. A2 主扫描：tile_n × wgm（12 配置）                          2 h
 6. A3 m_tiles 扫描（4 配置）                                   40 min
 7. A4 batch 延展（6 配置）                                     1.5 h
 8. A5 对照（2 配置）                                           20 min

Part B
 9. Qwen3-30B-A3B 跑通 + B4（gang vs 非 gang）                  1 h
10. B1/B2 路由与行利用率统计                                    1 h
11. B3 每 XCD 耗时分布                                          40 min
12. 换 Mixtral 级模型（能跑则跑）                               2 h

13. 汇总两张表 + 回答 §4.3 的四个问题                            1 h
```

**总计约 12–14 小时机时。** 建议分两次会话：准备 + A0/A1 一次（确认通路），A2–B 一次。

---

## 6. 明确不在本轮范围内

- **不做**跨 XCD 的 L2 分片带宽微基准、跨 XCD 消息延迟微基准——那些在 `plan/moe-plan-v2.md` 的 §3 里，是独立的下一步。
- **不改** Fleet 的 kernel、不做任何 IR 设计。
- **不下**"MoE 的芯粒优化该怎么做"的结论——本轮只产出**数据与归因**。

---

## 7. 给接手窗口的三条提醒

1. **`AMDGPU_TARGETS=gfx942` 是硬要求**（§1.2）。忘了设会浪费一整个编译周期。
2. **每次改参数必须删 `permanent_output_dir/`**。Fleet 的 megakernel 是按模式编译的，不删会用旧产物。
3. **A0 阶段的结果决定了后面还要不要做**。如果可用参数组合极少，立即报告，不要把预算耗在无意义的扫描上。
