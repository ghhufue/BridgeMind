"""Plot saved SL v2 metrics with matplotlib; no model or evaluation is rerun."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def ema(values, span=9):
    result, running = [], float(values[0])
    for value in values:
        running += 2/(span+1)*(float(value)-running)
        result.append(running)
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/play-sl-v2-20261026-2h")
    parser.add_argument("--output", default="artifacts/cardplay-sl-v2-20261026/curves")
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = run/"metrics.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    train = [r for r in rows if r["event"] == "progress"]
    valid = [r for r in rows if r["event"] == "validation"]
    assert train and valid
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#334155", "text.color": "#17243b",
                         "axes.edgecolor": "#b7c2d0", "grid.color": "#e1e7ef", "figure.facecolor": "#f5f7fb",
                         "axes.facecolor": "white", "legend.frameon": False, "savefig.facecolor": "#f5f7fb"})
    blue, orange, purple = "#2563eb", "#d76b16", "#7856bd"
    tx = np.asarray([r["elapsed_seconds"]/60 for r in train])
    vx = np.asarray([r["elapsed_seconds"]/60 for r in valid])
    vmax = float(max(tx[-1], vx[-1]))

    def decorate(ax, title, ylabel):
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("累计运行时间 / 分钟")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, vmax)
        ax.set_xticks(np.arange(0, 121, 20))
        ax.grid(axis="y", alpha=.85)
        ax.set_axisbelow(True)

    def finish(fig, name):
        fig.savefig(out/(name+".png"), dpi=160)
        vector = out/(name+".svg")
        fig.savefig(vector)
        vector.write_text("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(left=.075, right=.975, top=.86, bottom=.12, hspace=.44, wspace=.22)
    fig.suptitle("SL v2 · 两小时训练曲线", x=.075, y=.965, ha="left", fontsize=23, fontweight="bold")
    fig.text(.075, .917, "75% 真实对局 + 25% 基础课程 · 24 种花色增强 · 学习率 1e-5 · 248,419 步", fontsize=12, color="#53657e")
    ax = axes[0, 0]
    loss = np.asarray([r["train"]["loss"] for r in train])
    ax.plot(tx, loss, color=blue, alpha=.22, linewidth=.9, label="当时单批次")
    ax.plot(tx, ema(loss), color=blue, linewidth=2.2, label="指数平滑（span=9）")
    decorate(ax, "训练总损失", "Loss（越低越好）")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=10)

    ax = axes[0, 1]
    kinds = [("real_canonical", "原始真实样本", blue), ("real_augmented", "花色增强样本", orange), ("curriculum", "基础课程", purple)]
    for key, label, color in kinds:
        y = [r["validation"][key]["nll"] for r in valid]
        assert all(v>0 for v in y)
        ax.plot(vx, y, color=color, linewidth=2, marker="o", markersize=3, label=f"{label}  {y[-1]:.4f}")
    ax.set_yscale("log")
    decorate(ax, "固定验证集：策略损失", "NLL（对数坐标，越低越好）")
    ax.legend(loc="lower left", fontsize=10)

    ax = axes[1, 0]
    for key, label, color in kinds:
        metric = "macro_accuracy" if key == "curriculum" else "accuracy"
        y = [100*r["validation"][key][metric] for r in valid]
        label = "基础课程（宏平均）" if key == "curriculum" else label
        ax.plot(vx, y, color=color, linewidth=2, marker="o", markersize=3, label=f"{label}  {y[-1]:.2f}%")
    decorate(ax, "固定验证集：正确率", "正确率 / %（越高越好）")
    ax.set_ylim(50, 104)
    ax.legend(loc="lower right", fontsize=10)

    ax = axes[1, 1]
    means = [r["validation"]["duplicate_vs_initial"]["mean_imps"] for r in valid]
    intervals = np.asarray([r["validation"]["duplicate_vs_initial"]["bootstrap_95_ci"] for r in valid])
    ax.fill_between(vx, intervals[:, 0], intervals[:, 1], color=blue, alpha=.13, label="逐次 95% bootstrap 区间")
    ax.plot(vx, means, color=blue, linewidth=2, marker="o", markersize=3, label=f"相对初始模型  {means[-1]:+.2f} IMP/副")
    ax.axhline(0, color="#7d899b", linestyle="--", linewidth=1)
    decorate(ax, "128 副选模验证牌：双室对战", "平均 IMP / 副（越高越好）")
    ax.legend(loc="upper left", fontsize=10)
    fig.text(.075, .052, f"来源：metrics.jsonl · {len(train)} 条训练快照、{len(valid)} 轮验证；训练曲线不是区间平均，也未保存每一步。", fontsize=10, color="#64748b")
    fig.text(.075, .026, "右下为反复用于选模的验证牌，区间不校正选模偏差；独立 2,048 副验收结果见训练后验收报告。", fontsize=10, color="#64748b")
    finish(fig, "training-overview")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
    fig.subplots_adjust(left=.065, right=.98, top=.76, bottom=.22, wspace=.3)
    fig.suptitle("SL v2 · 辅助损失与梯度", x=.065, y=.94, ha="left", fontsize=22, fontweight="bold")
    for ax, key, title, color in [(axes[0], "belief_loss", "隐藏牌归属损失", blue),
                                  (axes[1], "value_loss", "价值预测损失", orange),
                                  (axes[2], "gradient_norm", "裁剪前梯度范数", purple)]:
        y = np.asarray([r["train"][key] for r in train])
        ax.plot(tx, y, color=color, alpha=.22, linewidth=.9, label="当时单批次")
        ax.plot(tx, ema(y), color=color, linewidth=2, label="指数平滑（span=9）")
        decorate(ax, title, "范数" if key == "gradient_norm" else "Loss")
        if key == "gradient_norm":
            ax.axhline(1, color="#7d899b", linestyle="--", linewidth=1, label="实际裁剪阈值 = 1")
        if key == "value_loss":
            ax.set_yscale("log")
            ax.set_ylabel("MSE（对数坐标）")
        ax.legend(fontsize=9, loc="upper left" if key == "gradient_norm" else "upper right")
    fig.text(.065, .075, "归属与价值损失仅使用真实样本；总损失 = 混合策略损失 + 0.2 × 归属损失 + 0.1 × 价值损失。", fontsize=10, color="#64748b")
    fig.text(.065, .035, "梯度图记录裁剪前范数，高于 1 不表示裁剪失效。辅助曲线为训练单批次快照，未单独记录验证辅助损失。", fontsize=10, color="#64748b")
    finish(fig, "training-auxiliary")

    flat = []
    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = prefix+key
            if isinstance(item, dict):
                result.update(flatten(item, name+"."))
            else:
                result[name] = json.dumps(item) if isinstance(item, list) else item
        return result
    for row in rows:
        flat.append(flatten(row))
    with (out/"metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in flat for k in row}))
        writer.writeheader()
        writer.writerows(flat)
    (out/"source.json").write_text(json.dumps({"metrics_path": str(log.resolve()), "metrics_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "training_points": len(train), "validation_points": len(valid), "smoothing": "causal EMA, span=9; no validation smoothing",
        "time_axis": "elapsed_seconds / 60; includes evaluation time", "matplotlib_version": matplotlib.__version__}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "training_points": len(train), "validation_points": len(valid)}))


if __name__ == "__main__":
    main()
