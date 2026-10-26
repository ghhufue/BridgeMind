// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { analysisRequest, biddingRequest, fetchBidding, validateAnalysis, type AnalysisResponse } from "../api/analysis";
import { applyCall, controller, legalCards, newGame, playCard, stateId } from "../bridge/engine";
import { SEATS, type Call, type Game } from "../bridge/types";
import { ModelDiagnostics } from "./ModelDiagnostics";
import { HomeScreen } from "./HomeScreen";
import { MobileTable } from "./MobileTable";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
function game() { return (["1NT", "P", "P", "P"] as Call[]).reduce((g, c) => applyCall(g, c), newGame(1, 46)); }
function response(g: Game): AnalysisResponse {
  const request = analysisRequest(g, "ai");
  const known = "hand" in request ? [...request.hand.cards, ...(request.dummy?.cards || [])] : [];
  const played = [...g.tricks.flatMap((t) => t.cards), ...g.currentTrick].map((e) => e.card);
  const legal = legalCards(g);
  return { protocol: "bridgemind.analysis.v1", stateId: stateId(g), source: "ai", bestCards: [legal[0]],
    diagnostics: { modelSha256: "a".repeat(64), inference: "greedy_neural", ownershipMethod: "raw",
      actionProbabilities: g.hands[g.turn].map((card) => ({ card, legal: legal.includes(card), probability: legal.includes(card) ? 1/legal.length : 0 })),
      ownership: SEATS.flatMap((seat) => g.initialHands[seat]).map((card) => ({ card,
        status: played.includes(card) ? "played" : known.includes(card) ? "known" : "unknown",
        probabilities: { N: .25, E: .25, S: .25, W: .25 } })),
      value: { normalized: .125, predictedScore: 950, perspective: ["N", "S"].includes(controller(g)) ? "NS" : "EW", target: "actual_terminal_team_score/7600" },
      winProbability: null, winProbabilityReason: "未训练胜率头" } };
}

describe("模型诊断信息边界与完整性", () => {
  it("所有手牌动作和52张归属均验证，胜率明确为空", () => {
    const g = game(), data = response(g);
    expect(validateAnalysis(data, g, "ai")).toEqual(data);
    render(<ModelDiagnostics game={g} data={data} />);
    expect(screen.getByText("+950.0")).toBeTruthy();
    expect(screen.getByText("暂不可用")).toBeTruthy();
    expect(screen.getByLabelText("全部手牌动作概率").children).toHaveLength(13);
    expect(screen.getAllByRole("row")).toHaveLength(40);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示真值对照" }));
    expect(screen.getByText(/本局未知牌最高概率归属命中/)).toBeTruthy();
    fireEvent.click(screen.getByRole("checkbox", { name: "只看未知牌" }));
    expect(screen.getAllByRole("row")).toHaveLength(53);
  });
  it.each(["missing_actions", "duplicate_ownership", "wrong_perspective", "fake_winrate", "nan_probability", "false_known", "unnormalized", "wrong_scale"])("拒绝损坏的诊断：%s", (kind) => {
    const g = game(), data = response(g), d = data.diagnostics!;
    if (kind === "missing_actions") d.actionProbabilities.pop();
    if (kind === "duplicate_ownership") d.ownership[0] = d.ownership[1];
    if (kind === "wrong_perspective") d.value.perspective = d.value.perspective === "NS" ? "EW" : "NS";
    if (kind === "fake_winrate") Object.assign(d, { winProbability: .8 });
    if (kind === "nan_probability") d.ownership[0].probabilities.N = NaN;
    if (kind === "false_known") d.ownership.find((r) => r.status === "unknown")!.status = "known";
    if (kind === "unnormalized") d.actionProbabilities[0].probability = .99;
    if (kind === "wrong_scale") d.value.predictedScore = 10;
    expect(() => validateAnalysis(data, g, "ai")).toThrow();
  });
  it("跨局面响应和DDS附带伪造模型诊断均被拒绝", () => {
    const g = game();
    expect(() => validateAnalysis(response(g), playCard(g, legalCards(g)[0]), "ai")).toThrow();
    expect(() => validateAnalysis({ ...response(g), source: "dds" }, g, "dds")).toThrow();
  });
  it("归属真值不进入AI请求，叫牌只发送当前一手牌", () => {
    const g = newGame(1, 46);
    const request = biddingRequest(g);
    expect(request.hand.cards).toEqual(g.hands.N);
    expect(request).not.toHaveProperty("hands");
    expect(request).not.toHaveProperty("initialHands");
    const visible = analysisRequest(game(), "ai");
    expect(visible).not.toHaveProperty("hands");
    expect(visible).not.toHaveProperty("initialHands");
  });
  it("叫牌调用拒绝过期响应和非法叫品", async () => {
    const g = newGame(1, 46);
    for (const extra of [{ stateId: "old", call: "P" }, { stateId: stateId(g), call: "X" }]) {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ protocol: "bridgemind.analysis.v1", source: "bidding", ...extra }))));
      await expect(fetchBidding(g, "http://localhost:8000", new AbortController().signal)).rejects.toThrow();
    }
  });
});

describe("主界面与南北双手控制", () => {
  it("主界面分别打开对战、调试和设置", () => {
    const onPlay = vi.fn(), onDebug = vi.fn(), onSettings = vi.fn();
    render(<HomeScreen board={3} onPlay={onPlay} onDebug={onDebug} onSettings={onSettings} />);
    fireEvent.click(screen.getByRole("button", { name: /PLAY WITH AI/ }));
    fireEvent.click(screen.getByRole("button", { name: /MODEL WORKBENCH/ }));
    fireEvent.click(screen.getByRole("button", { name: "连接与设置" }));
    expect(onPlay).toHaveBeenCalledOnce(); expect(onDebug).toHaveBeenCalledOnce(); expect(onSettings).toHaveBeenCalledOnce();
  });
  it("北家是防守者时仍由人选牌，南家叫牌保持手动", () => {
    let g = (["1NT", "P", "P", "P"] as Call[]).reduce((x, c) => applyCall(x, c), newGame(2, 46));
    while (g.turn !== "N") g = playCard(g, legalCards(g)[0]);
    const onPlay = vi.fn();
    render(<MobileTable game={g} partnership manual={false} reveal={false} paused={false} reduced sound={false} level={1}
      suggested={[]} status="北家出牌" canUndo={false} auction={null} onModal={vi.fn()} onReveal={vi.fn()} onPause={vi.fn()} onSound={vi.fn()}
      onLevel={vi.fn()} onCall={vi.fn()} onPlay={onPlay} onUndo={vi.fn()} onStep={vi.fn()} onNext={vi.fn()} />);
    const hand = screen.getByLabelText("北家手牌");
    const cards = [...hand.querySelectorAll("button")].filter((b) => !b.disabled);
    expect(cards).toHaveLength(legalCards(g).length);
    fireEvent.click(cards[0]);
    fireEvent.click(screen.getByRole("button", { name: /出牌$/ }));
    expect(onPlay).toHaveBeenCalledWith("N", legalCards(g)[0]);
  });
});
