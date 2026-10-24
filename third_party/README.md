# 桥牌相关第三方源码

此目录保存上游源码，供 BridgeMind 后续封装、验证和复现使用。上游实现保持原样；独立主线 DDS 已完成 Windows x64 优化编译。本地 OpenSpiel 检出尚未编译；项目虚拟环境另行安装了 OpenSpiel 2.0.2 的 PyPI wheel，用于环境适配和交叉验证。

## 来源与版本

| 组件 | 来源 | 固定提交 |
| --- | --- | --- |
| OpenSpiel | https://github.com/google-deepmind/open_spiel | `5886e0b822bcc469cacb53a82c18ac1e8748d820` |
| OpenSpiel 使用的 DDS 分支 | https://github.com/jblespiau/dds/tree/develop | `091ea94358a4016d4fb6069dea5c452cdc98d0bd` |
| 独立主线 DDS v3.1.0 | https://github.com/dds-bridge/dds/releases/tag/v3.1.0 | `fe6a6581a13535779eda5d9141aa704b3f6c8965` |

OpenSpiel 使用浅克隆与稀疏检出，保留了 Git 元数据。其目录内的 DDS 来自该提交的 `open_spiel/scripts/install.sh` 指定的仓库和分支，不是另行替换的新版 DDS。三个仓库均保留了 Apache-2.0 LICENSE。

## 独立主线 DDS

`dds/` 是主线 DDS v3.1.0 的完整浅克隆，检出到发布标签对应的固定提交（detached HEAD），用于后续开发独立求解模块。

- `dds/README.md`：上游介绍。
- `dds/docs/c++_interface.md`：C++ 求解接口与线程上下文。
- `dds/docs/python_interface.md`：Python 接口。
- `dds/library/src/solver_context/solver_context.hpp`：求解上下文实现。
- `dds/LICENSE`：Apache-2.0 许可。

主线 DDS 已使用 Bazel 9.1.0 和 MSVC 14.44.35207 完成 Windows x64 优化编译，生成 DLL、导入库及两个示例程序。选定的 6 个上游测试目标（28 个用例）全部通过，示例结果校验和 Python 加载 DLL 的版本检查通过。产物见 `../artifacts/dds/windows-x64/`，编译命令、问题处理及验证范围见该目录的 `README.md`。

`dds/` 与 `open_spiel/open_spiel/games/bridge/double_dummy_solver/` 独立保存，OpenSpiel 本身的编译依赖仍为旧版。BridgeMind 已在 `src/bridgemind/` 实现原生适配、独立求解服务、客户端和 OpenSpiel 外部 DDS 叫牌结算适配；详情见项目根目录 README。没有编译 DDS Python 扩展，而是通过 ctypes 调用现有 DLL。求解服务已完成独立进程测试，测试进程运行结束后关闭。

## 阅读顺序

以下路径相对于 `open_spiel/` 仓库根目录：

1. `open_spiel/games/bridge/bridge.h`：状态、动作、阶段和观察接口。
2. `open_spiel/games/bridge/bridge.cc`：叫牌、出牌、观察、DDS 结算。
3. `open_spiel/games/bridge/bridge_scoring.h`、`bridge_scoring.cc`：定约表示和计分。
4. `open_spiel/games/bridge/bridge_test.cc`：规则测试。
5. `open_spiel/python/pybind11/games_bridge.*`、`open_spiel/python/tests/games_bridge_test.py`：桥牌专用 Python 绑定与测试。
6. `open_spiel/python/examples/bridge_supervised_learning.py`：叫牌监督学习示例。

同时检出了：

- `bridge_uncontested_bidding.*`：无竞争叫牌环境。
- `open_spiel/games/tiny_bridge/`：简化桥牌环境与测试。
- `open_spiel/python/bots/*bridge*.py`：Blue Chip Bridge 机器人协议适配与测试。
- `open_spiel/python/examples/bridge_wb5.py`、`bridge_uncontested_bidding_bluechip.py`：外部机器人示例。
- `open_spiel/integration_tests/playthroughs/*bridge*.txt`：对局轨迹样例。
- `open_spiel/python/examples/data/bridge/`：仓库附带的小型样例数据，不是完整训练数据集。
- `open_spiel/games/bridge/double_dummy_solver/`：上游指定的 DDS 源码，包含独立 Git 元数据和许可。
- OpenSpiel 顶层核心 `.h`/`.cc`、构建入口、安装脚本和安装文档：便于追踪依赖。

## 当前边界

这是桥牌相关源码的稀疏检出，不是可独立编译的完整 SDK。桥牌依赖 OpenSpiel 通用核心、Abseil、Python 绑定基础设施等；本次没有运行安装脚本、安装 Python 包、下载完整数据集或执行测试。

OpenSpiel 的 `use_double_dummy_result` 参数默认开启，表示叫牌结束后直接用 DDS 结算。要模拟完整出牌过程，需要关闭此参数。

后续如果需要完整构建，可先在 OpenSpiel 仓库中执行 `git sparse-checkout disable` 展开同一提交的全部上游文件，再依照上游安装文档配置依赖。该操作可能继续联网下载对象；不会自动安装依赖。不要直接运行安装脚本并假定它只会安装桥牌组件。

BridgeMind 自己的适配层应放在此目录之外，便于保持第三方源码与上游一致。
