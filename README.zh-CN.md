[English](README.md)

# ONNX Runtime 项目（SpacemiT 分支与厂商包）

AI 工作的 ONNX 侧。两个产物必须放在一起看：承载 SpacemiT 构建文件的公开分支，以及
板子上实际运行的闭源提供者包。

## 固定来源

| 字段 | 取值 |
|---|---|
| 上游 | `https://github.com/spacemit-com/onnxruntime.git` |
| ref | `upstream-1.24.2` |
| 提交 | `d76541ad26ca93d2d52e5d2db2aacaff862df5cf` |
| 许可证 | MIT |
| 检出 | `src/onnxruntime` |
| 厂商包 | `spacemit-ort.riscv64.2.0.6.tar.gz`，包提交 `5a5b5914` |

提供者本身没有以源码形式发布：分支里只有工具链文件与一个提供者测试工具，发布包则
提供 `libonnxruntime.so.1.24.2+spacemit.a1` 与已 strip 的
`libspacemit_ep.so.2.0.6`。因此 `sources.lock.yaml` 同时登记分支的文件摘要与密封
包身份，以及我们被允许使用的公开入口。

## 状态

- 分支与包身份已同步；提供者内部按设计保持闭源。
- A100 的开发规则写在工作区契约里，不写在这里。
- 自动化（`src/onnxruntime_project`、`scripts/build.sh`、`scripts/run.sh`）是下一个
  交付项，并已在 `project.yaml` 中标注为待完成。
- 2026-09-20 规划轮：发布包已取回并核对身份（其库与板卡 deb 安装库在同一版本号下并非同一
  二进制，见 `TECHNICAL-ANALYSIS.md` §2），项目目录与测量计划见
  `src/onnxruntime_project/{PLAN.md,TECHNICAL-ANALYSIS.md}`，基于 AI 集群实测整理的排序优化清单
  见 `DEEP-ANALYSIS.md`（VLM 与具身智能优先）。随后 2026-09-21 完成第一轮实测：
  `BASELINE-20260921.md` 给出时延/线程/TCM/覆盖/算子/VLM 各表，与厂商公布的 K3 数据在
  百分之几内吻合；`BASELINE2-20260921.md` 再补数值门（60 个中间张量，在不应变化时逐位一致）、
  冷启动与每新帧成本、以及 X100 干扰阶梯。`METHOD-ALIGNMENT-20260921.md` 汇总厂商 K3 模型库数据
  （本地文档与上游逐字节一致），逐项对比厂商基准方法与我们的方法，并固化验收协议：优化既要对照
  官方数字、也要对照站得住的测试方案，两者不可只引其一。`BASELINE3-20260921.md` 用数字关闭了第一次
  int8 编码器尝试：动态量化路线（`xslim --dynq`）比出厂 fp16 编码器更慢（518.6 对 150.5 ms），纯 fp32
  副本慢 10 倍，`DENSE_ACCURACY_LEVEL` 只能让 dynq 模型变化 ±7 %；剩下的是"带校准的静态 QDQ"路线。
  `BASELINE4-20260921.md` 把厂商自己的模型库仓库（`model-zoo-vision` 2026-09-21 的 HEAD、`model-zoo-vlm`）
  当作验证材料：新增七行视觉基线（含他们的 no-DFL 检测器 9.08 ms 与两个 VLM 级编码器），在板卡上跑通
  他们的 VLM 矩阵脚本（四项功能检查全过；其自带 metrics 门槛失败——流式路径未被计数），并独立确认
  厂商发布的 dynq 编码器同样落在慢速 fp32 路径上（224² 在 4 个 AI 核上 2.10 s）。
  `OPTIMISATION-20260921.md` 测部署形态：批处理让 CNN 每帧提升 1.41-1.47 倍；单会话 8 线程的并发吞吐
  在 ~110 inf/s 饱和（多流只增加延迟，拆核反而 -36 %）；两簇互相干扰 10-25 %——另有一次采集因两个 EP
  进程争抢同一批 8 块 TCM 而作废，并原样保留为 invalid。
  `OPTIMISATION2-20260921.md` 用厂商自有管道基准测"模型之外"的开销（前后处理合计只占一帧的 1-5 %，
  据此按实测关闭 OpenCL 预处理），并用数字了结一个方法问题：厂商矩阵命令的 `-x 1` 对"EP 只认领部分
  节点"的图低估 **2.32 倍**——回退节点清单、TCM 占用采样与 provider 自身 trace 三方互证。
  `METHOD-ALIGNMENT-20260921.md` 第 7 节了结两行一直对不上的官方数字。LLM 解码行：发行包版本（含厂商
  自己的 0.1.1 发布包）、命令、模型身份与运行时 banner 现已与官方截图**逐字段一致**，而数值是预填
  -1.8 %、解码 -10.8 %，仅剩 OS 镜像代次这一个变量。VLM 行：单元格的统计口径官方未定义，候选口径之间
  相差 40 %；只有"视觉编码器自身的推理时间"能对上（-5.2 ~ -7.6 %），故该行只能当数量级参考、不能当
  端到端验收目标——已固化为验收规则 R10。
  `src/onnxruntime` 未做任何修改。

## 工作区登记

```xml
<project name="onnxruntime.git" path="projects/onnxruntime" />
```
