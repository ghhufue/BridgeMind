# DDS v3.1.0 — Windows x64 构建产物

来源：https://github.com/dds-bridge/dds/releases/tag/v3.1.0

固定提交：`fe6a6581a13535779eda5d9141aa704b3f6c8965`。上游源码未修改。

## 文件

| 文件 | 用途 |
| --- | --- |
| `dds.dll` | Windows x64 原生动态库，包含传统 C 接口和上下文接口 |
| `dds.lib` | DLL 导入库，来自 Bazel 的 `dds_shared.if.lib`；不是独立静态库 |
| `calc_dd_table.exe` | 上游完整牌表校验示例 |
| `solve_board.exe` | 上游单局面求解校验示例 |
| `LICENSE` | 上游 Apache-2.0 许可 |
| `manifest.json` | 源码版本、构建工具和文件 SHA-256 |
| `logs/` | 构建、测试、示例及 DLL 加载记录 |

## 编译环境

- Windows x64，Visual Studio Community 2022。
- MSVC 14.44.35207，Windows SDK 10.0.22621.0。
- 项目本地便携 Bazelisk 1.29.0：`D:\code\BridgeMind\.tools\bazelisk.exe`。
- 上游固定的 Bazel 9.1.0，`opt` 配置，8 个并发构建任务。
- 本次实际使用 MSVC，不是 Android NDK 编译器；构建日志记录了编译器路径。

没有修改系统 PATH、系统编码或 Visual Studio 安装。DLL 已在当前机器的 Python 3.13 中通过 `ctypes.WinDLL` 加载，`GetDDSInfo` 确认版本 3.1.0、64 位。部署到其他 Windows 机器时，仍需满足 DLL 的运行库依赖。

## 重建命令

在 `D:\code\BridgeMind\third_party\dds` 中运行 PowerShell：

```powershell
$env:BAZELISK_HOME = 'D:\code\BridgeMind\.tools\bazelisk-cache'
$env:BAZEL_VC = 'D:\Microsoft Visual Studio\2022\Community\VC'
$env:BAZEL_SH = 'D:\Git\bin\bash.exe'

& 'D:\code\BridgeMind\.tools\bazelisk.exe' `
  --output_user_root=D:/code/BridgeMind/.build/bazel `
  --output_base=D:/code/BridgeMind/.build/dds `
  build --config=opt --jobs=8 `
  --cxxopt=/utf-8 --host_cxxopt=/utf-8 --repo_contents_cache= `
  //library/src:dds //jni:dds_shared `
  //examples:calc_dd_table //examples:solve_board
```

原始产物位于 `.build/dds/execroot/_main/bazel-out/x64_windows-opt/bin/`。重建不会自动刷新本目录的副本，需要重新复制产物并更新清单。

## 环境问题与处理

1. 初始缺少 Bazel/Bazelisk：使用项目目录内的便携版本，未做全局安装。
2. 中文 Windows 默认代码页 936 导致 MSVC C4819，且上游开启警告视为错误：用 `/utf-8` 解决，未修改源码或关闭警告检查。
3. GoogleTest 获取时，Bazel 将文件移入仓库内容缓存遇到 `Permission denied`：用 `--repo_contents_cache=` 绕过该缓存，测试成功；保留下载缓存和编译缓存。
4. Bazel 提示 MSVC 英文诊断输出识别失败，禁用了头文件裁剪：属于增量构建效率提示，没有阻止本次构建和测试，没有为此修改 Visual Studio。

## 验证范围

以下 6 个上游测试目标全部通过，共 28 个测试用例：

- `//library/tests:dds_c_api_test`（13）
- `//library/tests/solve_board:solve_board_test`（1）
- `//library/tests/system:concurrency_validation_test`（1）
- `//library/tests/system:context_equivalence_test`（4）
- `//library/tests/system:max_threads_equivalence_test`（4）
- `//library/tests/system:worker_context_reuse_test`（5）

复测使用相同环境变量和启动参数，将 `build` 换为 `test`，保留优化、编码、缓存参数，添加 `--test_output=errors` 并传入上述目标。

此外，`calc_dd_table.exe` 的 3 副样例均为 `OK`；`solve_board.exe` 的 3 副样例、每副两种结果选项均为 `OK`。没有运行整个上游仓库的全部测试，也没有构建 Python 扩展、安卓、WASM 或独立服务。
