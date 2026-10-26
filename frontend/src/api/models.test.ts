import { afterEach, describe, expect, it, vi } from "vitest";
import { applyCall, legalCards, newGame, stateId } from "../bridge/engine";
import type { Call } from "../bridge/types";
import { fetchAnalysis, fetchBiddingDecision, fetchModels } from "./analysis";

afterEach(() => vi.unstubAllGlobals());
describe("模型选择接口", () => {
  it("选择模型时仍只发送合法可见手牌，并核对返回的模型标识", async () => {
    const game = (["1NT", "P", "P", "P"] as Call[]).reduce((g, call) => applyCall(g, call), newGame(1, 42));
    const response = { protocol: "bridgemind.analysis.v1", stateId: stateId(game), source: "ai", bestCards: [legalCards(game)[0]], modelId: "sl-long" };
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => response }); vi.stubGlobal("fetch", fetch);
    await fetchAnalysis(game, "ai", "http://computer:8000", new AbortController().signal, "sl-long");
    const body = JSON.parse(fetch.mock.calls[0][1].body);
    expect(body.modelId).toBe("sl-long"); expect(body.information).toBe("visible"); expect(body.hands).toBeUndefined();
    response.modelId = "default";
    await expect(fetchAnalysis(game, "ai", "http://computer:8000", new AbortController().signal, "sl-long")).rejects.toThrow("模型与选择不匹配");
  });
  it("叫牌模型标识与动作概率都必须有效", async () => {
    const game = newGame(1, 42);
    const response = { protocol: "bridgemind.analysis.v1", stateId: stateId(game), source: "bidding", call: "P", modelId: "ppo-pilot", probabilities: { P: 1 } };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => response }));
    expect((await fetchBiddingDecision(game, "http://computer:8000", new AbortController().signal, "ppo-pilot")).call).toBe("P");
    response.probabilities.P = .5;
    await expect(fetchBiddingDecision(game, "http://computer:8000", new AbortController().signal, "ppo-pilot")).rejects.toThrow("概率格式错误");
  });
  it("模型目录拒绝文件路径或重复标识", async () => {
    const row = { id: "../checkpoint.pt", name: "bad", available: true };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({ protocol: "bridgemind.models.v1", play: [row], bidding: [row] }) }));
    await expect(fetchModels("http://computer:8000", new AbortController().signal)).rejects.toThrow("模型列表格式错误");
  });
});
