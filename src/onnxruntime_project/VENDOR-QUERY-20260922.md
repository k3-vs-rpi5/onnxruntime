# 致 SpacemiT：ONNX Runtime 2.0.6 provider 的三项可复现问题（K3 / Bianbu 4.0.6）

**环境**：K3 开发板（X100 cpu0-7 + A100 cpu8-15），Bianbu 4.0.6，kernel 6.18.3，ONNX Runtime
`2.0.6`（deb：runtime `076d9e83…`、provider `d8fa3633…`；tarball：`93bb7560…`/`dcc95030…`），
`onnxruntime_perf_test -e spacemit -x 8 -S 1 -c 1 -I -i "SPACEMIT_EP_INTRA_THREAD_NUM|8"`，
`SPACEMIT_EP_DEBUG_PROFILE` 与 `-p`（per-node profile）用于归属判定，`spacemit-tcm-smi` 确认 AI 核占用。
模型：官方 `yolo26n-depth.fp16.onnx`（我们的探针与它同形）。

## 问题 1：能力检查按"连通分量"判定，孤立的受支持算子会被拒

`Resize(mode=linear, coordinate_transformation_mode=half_pixel)` 在文档中列为受支持
（`onnxruntime_ep_ops.md`：Resize 支持 `asymmetric`/`half_pixel` 与 `nearest`/`linear`）。实测：

| 图 | provider 归属 |
|---|---|
| 只有一个 `Resize`（`sizes` 或 `scales` 形态、fp32 或 fp16、192→768） | **CPU EP 100%** |
| 同上，但用你们 `pp_liteseg` 同款通道/尺寸（C=128 1→16、C=19 64→128） | **CPU EP 100%** |
| `Conv` 之后接同一个 `Resize` | **被认领**（整图一个 Spine 子图，0 个 CPU 节点） |
| 同上再包 Q/DQ | **同样被认领** |

也就是说：**同一个算子、同样属性，只因所处碎片大小不同就被接受或被拒**。反证在你们自己的包里——
`pp_liteseg.q.onnx` 的 7 个 `linear + half_pixel` Resize 全部被认领，并出现在
`SPACEMIT_EP_DUMP_SUBGRAPHS` 导出的子图内；而 `yolo26n-depth.fp16.onnx` 里同形的 1 个被拒。

**Q1**：这个"碎片必须足够大/必须挨着已认领算子"的规则是有意的吗？若是，能否给出可判定的条件
（最小规模、连通性要求），以便我们在转换阶段预测，而不用靠试。

## 问题 2：去掉不支持的算子把区域重新连通后，provider 反而慢 16%

`yolo26n-depth` 的头部链条是 `Mul(Pow(Exp(Clip(x)), a), c)`，其中 **`Exp` 不在你们的支持清单里**。
我们按"消除不支持算子可让区域连通"的思路改写成 `Exp → Pow(e, x)`（数值差 **0.03 fp16 ulp**）：

| 指标 | 改写前 | 改写后 |
|---|---|---|
| 每帧 CPU 节点执行次数 | 15 | **8** |
| CPU 占比（profile 内） | 18.4 % | **8.0 %** |
| CPU 侧时间 | 12.46 ms | **6.14 ms** |
| **无插桩单帧时间（-r 20，两次）** | **64.52 / 64.59 ms** | **74.93 / 74.83 ms** |

回退确实按预测减少了，**但整帧慢了 16%**：被重新连通的 provider 区域比它替换掉的 CPU 碎片贵约 16 ms。
同样的算术，两种执行路径差 16 ms。

**Q2**：这是预期行为吗？能否解释该头部区域在 provider 上的执行为何比 CPU 路径更贵（布局转换？该区域
未走 IME 矩阵单元？`Pow` 的实现？）——这是本项最大的单点损失。

**Q3**：`Exp` 是故意不支持的吗？若是，推荐的等价替代是什么（`Pow(e,x)` 可用但如上所述并不划算）？

## 问题 3：能力检查通过了、执行却崩溃（段错误）

为绕开问题 1，我们把 `linear` Resize 改写成可分离插值（`Gather`+`Mul`+`Add`，索引/权重为常量）。
各基本算子单独测试**全部正常**（`Gather` axis 2/3、int32/int64、负轴、广播 `Mul`），但该**组合**
让 provider **段错误**（不是拒绝，是崩溃），在 24→48、48→96、192→768 三种尺寸上均复现：

```
$ SPACEMIT_EP_DEBUG_PROFILE=/tmp/t onnxruntime_perf_test <rewritten.onnx> -e spacemit -r 20 -x 8 -S 1 -c 1 -I \
    -i "SPACEMIT_EP_INTRA_THREAD_NUM|8"
Segmentation fault
```

**Q4**：`Gather` 组合触发的崩溃是否有已知限制/修复计划？能力检查若无法判定，希望至少是**拒绝**而不是崩溃。

## 附：我们已做但无改观的方向（供参考，避免重复建议）

- `onnxslim 0.1.87` 化简该模型：算子数与体积不变，回退 18.7% vs 原始 18.4%（无改观）；
- `Resize` 属性归一化（去掉 `nearest_mode=floor`、`cubic_coeff_a`）、`scales→sizes` 形态归一化：
  数值**逐位一致**，回退不变；
- fp16 的 linear Resize 用 fp32 Cast 包裹：数值 0.50 fp16 ulp，回退不变（并多 1 个 CPU 节点）；
- `-o 0/2/99`、并行执行器、arena/pattern、spinning、env allocators、`SPACEMIT_EP_*` 线程与绑核、
  `INTER×INTRA` 多流、X100 分担：均无改观或更差（X100 跑同类 int8 图慢 97 倍）。

## 复现材料（随附）

- 探针：`Resize` 单算子与 `Conv+Resize` 对照、`Exp→Pow` 改写模型、Gather 改写模型；
- 归属证据：ORT per-node profile（`CPUExecutionProvider` vs `SpaceMITExecutionProvider`）与
  `SPACEMIT_EP_DEBUG_PROFILE` trace；
- 全部原始文件与 sha256 见证据目录
  `/root/k3-evidence/onnxruntime/baseline-20260921/model-alternatives/`（含
  `converter-r1/summary.txt`、`summary-round2.txt`）。

如需要，我们可以把任一探针打包成你们可直接运行的最小复现包。
