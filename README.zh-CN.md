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

## 工作区登记

```xml
<project name="onnxruntime.git" path="projects/onnxruntime" />
```
