import { useState } from "react";
import type { AnalysisResponse } from "../api/analysis";
import type { Card, Game } from "../bridge/types";
import { SEATS } from "../bridge/types";
import { cardLabel, seatName } from "../bridge/engine";

export function ModelDiagnostics({ data, game }: { data: AnalysisResponse; game: Game }) {
  const [showTruth, setShowTruth] = useState(false);
  const [unknownOnly, setUnknownOnly] = useState(true);
  const d = data.diagnostics;
  if (!d) return <p className="diagnostic-unavailable">当前服务只返回出牌建议。连接新版电脑服务后，可查看归属与价值预测。</p>;
  const actualOwner = (card: Card) => SEATS.find((s) => game.initialHands[s].includes(card))!;
  const unknown = d.ownership.filter((row) => row.status === "unknown");
  const hits = unknown.filter((row) => [...SEATS].sort((a, b) => row.probabilities[b]-row.probabilities[a])[0] === actualOwner(row.card)).length;
  const rows = d.ownership.filter((row) => row.status !== "played" && (!unknownOnly || row.status === "unknown"));
  function exportSnapshot() {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = `model-analysis-board-${game.board}-r${game.revision}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <div className="model-diagnostics">
    <div className="diagnostic-metrics">
      <section><span>预测终局本方得分</span><strong>{d.value.predictedScore >= 0 ? "+" : ""}{d.value.predictedScore.toFixed(1)}</strong>
        <small>{d.value.perspective === "NS" ? "南北" : "东西"}视角 · 原始分</small></section>
      <section><span>胜率估计</span><strong className="unavailable">暂不可用</strong><small>当前模型没有胜率头</small></section>
    </div>
    <p className="debug-explanation">价值输出 {d.value.normalized.toFixed(5)} × 7600。它预测终局本方得分，不是胜率，也不是 IMP。{d.winProbabilityReason}</p>
    <div className="debug-heading"><h4>每张手牌的动作概率</h4><span>合法牌概率合计 100%</span></div>
    <div className="debug-actions" aria-label="全部手牌动作概率">{[...d.actionProbabilities].sort((a, b) => b.probability-a.probability).map((row) =>
      <div className={`debug-action ${!row.legal ? "illegal" : ""}`} key={row.card}>
        <b className={/[HD]/.test(row.card[0]) ? "red" : ""}>{cardLabel(row.card)}</b>
        <div><i style={{ width: `${100*row.probability}%` }} /></div>
        <span>{row.legal ? `${(row.probability*100).toFixed(3)}%` : "不可出 · 0%"}</span>
      </div>)}</div>
    <div className="debug-heading"><h4>牌张归属预测</h4><span>{unknown.length} 张未知牌</span></div>
    <div className="debug-toggles">
      <label><input type="checkbox" checked={unknownOnly} onChange={(e) => setUnknownOnly(e.target.checked)} />只看未知牌</label>
      <label><input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} />显示真值对照</label>
    </div>
    {showTruth && <p className="truth-summary">本局未知牌最高概率归属命中：{hits}/{unknown.length}{unknown.length ? `（${(100*hits/unknown.length).toFixed(1)}%）` : ""}。真值仅在客户端对照，不传给模型。</p>}
    <div className="ownership-scroll"><table className="ownership-table"><thead><tr><th>牌</th>{SEATS.map((s) => <th key={s}>{seatName[s]}家</th>)}{showTruth && <th>实际</th>}</tr></thead>
      <tbody>{rows.map((row) => <tr key={row.card}><th className={/[HD]/.test(row.card[0]) ? "red" : ""}>{cardLabel(row.card)}{row.status === "known" && <small>已知</small>}</th>
        {SEATS.map((s) => <td key={s} style={{ background: `rgba(36, 112, 91, ${row.probabilities[s]*.2})` }}>{(100*row.probabilities[s]).toFixed(1)}%</td>)}
        {showTruth && <td>{seatName[actualOwner(row.card)]}家</td>}</tr>)}</tbody></table></div>
    {!rows.length && <p>当前没有需要预测的未知牌。</p>}
    <p className="debug-explanation">未知牌显示原始归属头概率，未做剩余张数约束校正；已知牌由可见信息固定。模型只获得当前控制者的合法可见信息。</p>
    <div className="debug-fingerprint"><span>权重 SHA256 · {d.modelSha256?.slice(0, 16) || "未提供"}</span><button onClick={exportSnapshot}>导出分析 JSON</button></div>
  </div>;
}
