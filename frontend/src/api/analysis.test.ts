import { describe, expect, it, vi, afterEach } from "vitest";
import {
  analysisRequest,
  fetchAnalysis,
  normalizeServerUrl,
  validateAnalysis,
  fetchHealth,
} from "./analysis";
import { applyCall, legalCards, playCard, stateId } from "../bridge/engine";
import { newGame } from "../bridge/engine";
import type { Call } from "../bridge/types";

const playGame = () =>
  (["1NT", "P", "P", "P"] as Call[]).reduce(
    (g, c) => applyCall(g, c),
    newGame(1, 56),
  );
const responseFor = (source: "ai" | "dds" = "ai") => {
  const g = playGame();
  return {
    g,
    data: {
      protocol: "bridgemind.analysis.v1",
      stateId: stateId(g),
      source,
      bestCards: [legalCards(g)[0]],
    },
  };
};
afterEach(() => vi.unstubAllGlobals());

describe("电脑服务健康检查", () => {
  it("检查真实服务标识及 AI / DDS 就绪状态", async () => {
    const data = {
      protocol: "bridgemind.analysis.v1",
      service: "BridgeMind LAN",
      status: "ready",
      services: {
        ai: { ready: true, model: "trained-model" },
        dds: { ready: true, version: "3.1.0" },
      },
    };
    const mock = vi.fn().mockResolvedValue(new Response(JSON.stringify(data)));
    vi.stubGlobal("fetch", mock);
    expect(await fetchHealth("http://192.168.5.118:8000/")).toEqual(data);
    expect(mock.mock.calls[0][0]).toBe(
      "http://192.168.5.118:8000/api/v1/health",
    );
  });
  it("拒绝其他网站返回的成功响应和不完整状态", async () => {
    for (const data of [
      { status: "ok" },
      { protocol: "bridgemind.analysis.v1", service: "other", status: "ready" },
      {
        protocol: "bridgemind.analysis.v1",
        service: "BridgeMind LAN",
        status: "ready",
        services: {},
      },
    ]) {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(new Response(JSON.stringify(data))),
      );
      await expect(fetchHealth("http://localhost:8000")).rejects.toThrow(
        "不是 BridgeMind",
      );
    }
  });
  it("服务离线时不显示连接成功", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("offline", { status: 503 })),
    );
    await expect(fetchHealth("http://localhost:8000")).rejects.toThrow("503");
  });
});

describe("信息隔离", () => {
  it("AI 首攻请求只包含控制者手牌，不含四手牌或原始牌局", () => {
    const g = playGame();
    const request = analysisRequest(g, "ai");
    expect(request.information).toBe("visible");
    expect("hand" in request && request.hand).toEqual({
      seat: "E",
      cards: g.hands.E,
    });
    expect("dummy" in request && request.dummy).toBeNull();
    expect(request).not.toHaveProperty("hands");
    expect(request).not.toHaveProperty("initialHands");
    for (const seat of ["N", "S", "W"] as const)
      for (const card of g.hands[seat])
        expect(JSON.stringify(request)).not.toContain(`"${card}"`);
  });
  it("明手出牌请求提供庄家控制者及明手，仍不提供防守牌", () => {
    let g = playGame();
    g = playCard(g, legalCards(g)[0]);
    const request = analysisRequest(g, "ai");
    expect(request.toPlay).toBe("S");
    expect(request.controller).toBe("N");
    expect("hand" in request && request.hand).toEqual({
      seat: "N",
      cards: g.hands.N,
    });
    expect("dummy" in request && request.dummy).toEqual({
      seat: "S",
      cards: g.hands.S,
    });
    for (const seat of ["E", "W"] as const)
      for (const card of g.hands[seat])
        expect(JSON.stringify(request)).not.toContain(`"${card}"`);
  });
  it("DDS 显式携带四家剩余牌与当前局面", () => {
    const g = playGame();
    const request = analysisRequest(g, "dds");
    expect(request.information).toBe("full");
    expect("hands" in request && request.hands).toEqual(g.hands);
    expect(request.stateId).toBe(stateId(g));
  });
  it("叫牌时不允许发送出牌分析", () =>
    expect(() => analysisRequest(newGame(), "ai")).toThrow());
  it("撤销后改走不同分支，即使步数相同，也有不同局面标识", () => {
    const g = playGame();
    const cards = legalCards(g);
    const first = playCard(g, cards[0]);
    const second = playCard(g, cards[1]);
    expect(first.revision).toBe(second.revision);
    expect(stateId(first)).not.toBe(stateId(second));
  });
});

describe("响应验证与网络边界", () => {
  it("接受全部等价最优牌，不强制伪造概率", () => {
    const { g, data } = responseFor();
    data.bestCards = legalCards(g).slice(0, 2);
    expect(validateAnalysis(data, g, "ai")).toEqual(data);
  });
  it("拒绝错误版本、来源、局面版本和非法最优牌", () => {
    const { g, data } = responseFor();
    for (const patch of [
      { protocol: "v0" },
      { source: "dds" },
      { stateId: "old:0" },
      { bestCards: [] },
      { bestCards: ["ZZ"] },
      { bestCards: [data.bestCards[0], data.bestCards[0]] },
    ])
      expect(() => validateAnalysis({ ...data, ...patch }, g, "ai")).toThrow();
  });
  it("拒绝超范围概率、余墩、非数字和重复候选", () => {
    const { g, data } = responseFor();
    const card = data.bestCards[0];
    for (const candidates of [
      [{ card, probability: 1.1 }],
      [{ card, tricks: 14 }],
      [{ card, tricks: 1.5 }],
      [{ card, probability: NaN }],
      [{ card }, { card }],
      [{ card: "ZZ" }],
    ])
      expect(() =>
        validateAnalysis({ ...data, candidates }, g, "ai"),
      ).toThrow();
  });
  it("服务地址只接受 HTTP(S)，不接受密码、查询或片段", () => {
    expect(normalizeServerUrl(" http://192.168.5.118:8000/ ")).toBe(
      "http://192.168.5.118:8000",
    );
    for (const url of [
      "javascript:alert(1)",
      "file:///x",
      "http://user:pass@host",
      "http://host?token=x",
      "http://host#x",
    ])
      expect(() => normalizeServerUrl(url)).toThrow();
  });
  it("POST AI 使用约定的端点和可见信息序列化", async () => {
    const { g, data } = responseFor();
    const mock = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify(data), { status: 200 }));
    vi.stubGlobal("fetch", mock);
    const result = await fetchAnalysis(
      g,
      "ai",
      "http://192.168.1.5:8000",
      new AbortController().signal,
    );
    expect(result.bestCards).toEqual(data.bestCards);
    const [url, options] = mock.mock.calls[0];
    expect(url).toBe("http://192.168.1.5:8000/api/v1/analysis/ai");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).not.toHaveProperty("hands");
  });
  it("HTTP 错误与服务返回非法牌会向界面抛出错误", async () => {
    const { g, data } = responseFor("dds");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("unavailable", { status: 503 })),
    );
    await expect(
      fetchAnalysis(
        g,
        "dds",
        "http://localhost:8000",
        new AbortController().signal,
      ),
    ).rejects.toThrow("503");
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ ...data, bestCards: ["ZZ"] })),
        ),
    );
    await expect(
      fetchAnalysis(
        g,
        "dds",
        "http://localhost:8000",
        new AbortController().signal,
      ),
    ).rejects.toThrow("非法");
  });
});
