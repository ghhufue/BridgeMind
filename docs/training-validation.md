# 训练验证记录

日期：2026-10-25，Windows x64。本记录是流程验证，尚未建立棋力提升的证据。

## 环境与检查

- Python 3.13，OpenSpiel 2.0.2 wheel，PyTorch 2.11.0+cu128，NumPy 2.5.4。
- NVIDIA RTX 5070，12 GB；已实际在 CUDA 上执行监督与 PPO 更新。
- 完整 pytest：58 项通过，9.09 秒。包含既有 DDS、协议、并发、计分测试，以及编码、导入、合法掩码、队伍收益、对手冻结、优化器更新、推理和续训测试。
- CPU 上连续训练两轮，与第一轮后保存、恢复再训练一轮，最终 PPO 参数逐位一致；改变学习率后恢复会被拒绝。
- 六个本地集合按完整发牌 SHA256 审计，互不相交；测试牌库未用于本轮效果评估。
- 公开牌表随机划分后的前 32 副，由 DDS v3.1.0 重算的 20 个结果均一致。
- 真实独立 DDS TCP 进程、两个工作线程，生成 16 副新牌表并按 8/4/4 保存；结束后求解器进程已关闭。

## 监督短训练

从 brl 公共监督模型开始，使用 100,000 副牌、1,106,954 条决策，Adam 学习率 1e-4，batch 1024，共 300 步。验证从独立 valid 集用固定随机种子抽取 8,192 条决策，含重复抽样。

| 指标 | 初始 | 第 300 步 |
|---|---:|---:|
| 验证 NLL | 0.207479 | 0.185826 |
| 验证叫牌预测准确率 | 92.4316% | 93.1885% |

这是对 WBridge5 叫牌的拟合指标。模型在同一批 1,024 副验证牌上，与公共初始模型双室对战：**+0.05957 IMP/副，bootstrap 95% 区间 [-0.18164, +0.30078]**。区间包含零，不能据此认定棋力提高。

```powershell
.\.venv\Scripts\python.exe -m bridgemind.training train-sl --run runs/sl-pilot --init data/baselines/brl-sl.pt --steps 300 --eval-every 50 --device cuda
.\.venv\Scripts\python.exe -m bridgemind.training evaluate --model runs/sl-pilot/last.pt --opponent data/baselines/brl-sl.pt --boards data/boards/valid.npz --count 1024 --output runs/sl-pilot/duplicate-validation.json
```

## PPO 短训练

从上述第 300 步监督模型开始，学习率 1e-5，128 副牌/次更新，4 个 epoch，minibatch 1024。先运行 8 次更新，再实际从检查点恢复到第 10 次，合计 1,280 个双室牌局样本。历史快照每 2 次更新保存，锚点为训练开始时的监督模型。

第 10 次模型在同一批 1,024 副验证牌上，相对监督锚点：**-0.05273 IMP/副，bootstrap 95% 区间 [-0.21777, +0.10840]**。57 胜、910 平、57 负；胜负计数相等不意味着 IMP 相等。训练能够更新参数，但目前没有证据证明 PPO 改善棋力。

```powershell
.\.venv\Scripts\python.exe -m bridgemind.training train-ppo --run runs/ppo-pilot --init runs/sl-pilot/last.pt --updates 8 --boards-per-update 128 --eval-boards 256 --eval-every 4 --snapshot-every 2 --device cuda
.\.venv\Scripts\python.exe -m bridgemind.training train-ppo --run runs/ppo-pilot --resume --updates 10 --boards-per-update 128 --eval-boards 256 --eval-every 4 --snapshot-every 2 --device cuda
.\.venv\Scripts\python.exe -m bridgemind.training evaluate --model runs/ppo-pilot/last.pt --opponent runs/sl-pilot/last.pt --boards data/boards/valid.npz --count 1024 --output runs/ppo-pilot/duplicate-vs-sl.json
```

另外，公共模型与自身在 512 副固定验证牌上的贪心双室结果全部为零 IMP，验证了交换位置和结算的对称性。

## 留存内容与范围

`artifacts/training/` 保存机器可读汇总、完整训练指标和逐副评估结果，含牌局 ID、模型及数据哈希。`runs/` 留存本机检查点和冻结对手；`data/` 留存数据与初始模型。这两个目录由 Git 忽略，不随仓库分发。代码中的检查点元数据记录训练时的 Git 提交和 dirty 状态；本轮训练发生在实现提交前，因此 dirty 为 true。

数据划分、训练随机数、优化器及断点恢复均已验证；长期收敛、多种子稳定性、与其他强程序的对战、实际非全知出牌环境下的叫牌效果尚未验证。后续选择检查点应依靠独立验证对战，不能仅依据监督准确率或自对弈训练收益。
