import { describe, expect, it, vi } from "vitest";
import {
  applyCall,
  collectTrick,
  contractFromAuction,
  controller,
  dummySeat,
  dummyVisible,
  gameResult,
  highCardPoints,
  legalCalls,
  legalCards,
  newGame,
  nextSeat,
  playCard,
  practiceCall,
  practiceCard,
  sameSide,
  scoreContract,
  trickWinner,
} from "./engine";
import { SEATS, type Call, type Card, type Contract, type Game } from "./types";

function auction(calls: Call[], board = 1) {
  return calls.reduce((g, c) => applyCall(g, c), newGame(board, 42));
}

describe("发牌与牌局元信息", () => {
  it("局域网 HTTP 环境没有 randomUUID 时仍能发牌", () => {
    const getRandomValues = crypto.getRandomValues.bind(crypto);
    vi.stubGlobal("crypto", { getRandomValues });
    try {
      expect(newGame().id).toMatch(/^[a-f0-9]{32}$/);
    } finally {
      vi.unstubAllGlobals();
    }
  });
  it("52 张牌互不重复，四家各 13 张，HCP 总数为 40", () => {
    const g = newGame(1, 123);
    const cards = Object.values(g.hands).flat();
    expect(new Set(cards).size).toBe(52);
    expect(SEATS.map((s) => g.hands[s].length)).toEqual([13, 13, 13, 13]);
    expect(SEATS.reduce((n, s) => n + highCardPoints(g.hands[s]), 0)).toBe(40);
    expect(newGame(1, 123).hands).toEqual(g.hands);
  });
  it("发牌人与 16 副局况按复式桥牌轮转", () => {
    expect(
      Array.from({ length: 4 }, (_, i) => newGame(i + 1, 1).dealer),
    ).toEqual(SEATS);
    expect(
      Array.from({ length: 16 }, (_, i) => newGame(i + 1, 1).vulnerable),
    ).toEqual([
      [false, false],
      [true, false],
      [false, true],
      [true, true],
      [true, false],
      [false, true],
      [true, true],
      [false, false],
      [false, true],
      [true, true],
      [false, false],
      [true, false],
      [true, true],
      [false, false],
      [true, false],
      [false, true],
    ]);
  });
});

describe("叫牌", () => {
  it("开局不能加倍，再加倍；叫品必须逐级提高", () => {
    expect(legalCalls(newGame(1, 1))).toHaveLength(36);
    expect(() => applyCall(newGame(), "X")).toThrow();
    const g = auction(["1H"]);
    expect(legalCalls(g)).toContain("1S");
    expect(legalCalls(g)).not.toContain("1D");
    expect(legalCalls(g)).not.toContain("1H");
    expect(() => applyCall(g, "1H")).toThrow();
  });
  it("加倍只能由对手，再加倍只能由定约方；新叫品清除加倍", () => {
    const doubled = auction(["1H", "X"]);
    expect(legalCalls(doubled)).toContain("XX");
    expect(legalCalls(doubled)).not.toContain("X");
    expect(legalCalls(auction(["1H", "P"]))).not.toContain("X");
    expect(legalCalls(auction(["1H", "X", "P"]))).not.toContain("XX");
    expect(
      contractFromAuction(auction(["1H", "X", "XX", "2C"]).auction)?.doubled,
    ).toBe(0);
    expect(auction(["1H", "X", "XX", "P", "P", "P"]).contract?.doubled).toBe(2);
  });
  it("Pass 不会取消加倍权限，叫牌方可在间隔两次 Pass 后再加倍", () => {
    expect(legalCalls(auction(["1H", "P", "P"]))).toContain("X");
    expect(legalCalls(auction(["1H", "X", "P", "P"]))).toContain("XX");
  });
  it("四家不叫流局，但开叫后只需连续三次 Pass", () => {
    expect(auction(["P", "P", "P"]).phase).toBe("auction");
    const g = auction(["P", "P", "P", "P"]);
    expect(g.phase).toBe("finished");
    expect(g.contract).toBeNull();
    expect(gameResult(g).nsScore).toBe(0);
    expect(auction(["P", "1NT", "P", "P", "P"]).phase).toBe("play");
  });
  it("庄家是本方首次叫出最终花色者，处理不同发牌人", () => {
    const g = auction(["1H", "1S", "2H", "P", "4H", "P", "P", "P"]);
    expect(g.contract).toEqual({
      level: 4,
      strain: "H",
      declarer: "N",
      doubled: 0,
    });
    expect(g.turn).toBe("E");
    expect(
      auction(["1NT", "P", "3NT", "P", "P", "P"], 3).contract?.declarer,
    ).toBe("S");
  });
  it("7NT 后无更高叫品，仍允许合法的加倍", () => {
    expect(legalCalls(auction(["7NT"]))).toEqual(["P", "X"]);
  });
});

describe("出牌与明手", () => {
  it("必须跟牌；缺门时可打任意牌；拒绝非本方手牌", () => {
    let g = auction(["1NT", "P", "P", "P"]);
    g = { ...g, hands: { ...g.hands, E: ["HA"], S: ["H2", "SA", "C2"] } };
    g = playCard(g, "HA");
    expect(legalCards(g)).toEqual(["H2"]);
    expect(() => playCard(g, "SA")).toThrow();
    expect(() => playCard(g, "HK")).toThrow();
    g = { ...g, hands: { ...g.hands, S: ["SA", "C2"] } };
    expect(legalCards(g)).toEqual(["SA", "C2"]);
  });
  it("首攻前明手隐藏，首攻后才公开；庄家控制明手", () => {
    let g = auction(["1S", "P", "P", "P"]);
    expect(dummySeat(g)).toBe("S");
    expect(dummyVisible(g)).toBe(false);
    g = playCard(g, legalCards(g)[0]);
    expect(g.turn).toBe("S");
    expect(dummyVisible(g)).toBe(true);
    expect(controller(g)).toBe("N");
  });
  it("将牌高于首引花色；无将时垫牌不能赢墩", () => {
    const cards = [
      { seat: "N" as const, card: "HA" as Card },
      { seat: "E" as const, card: "S2" as Card },
      { seat: "S" as const, card: "HK" as Card },
      { seat: "W" as const, card: "SA" as Card },
    ];
    expect(trickWinner(cards, "S")).toBe("W");
    expect(trickWinner(cards, "NT")).toBe("N");
    expect(trickWinner(cards, "C")).toBe("N");
  });
  it("一墩四张后锁定出牌，收墩后由赢家引牌", () => {
    let g = auction(["1S", "P", "P", "P"]);
    const original = structuredClone(g);
    expect(() => collectTrick(g)).toThrow();
    for (let i = 0; i < 4; i++) g = playCard(g, legalCards(g)[0]);
    expect(legalCards(g)).toEqual([]);
    const winner = trickWinner(g.currentTrick, "S");
    g = collectTrick(g);
    expect(g.turn).toBe(winner);
    expect(g.tricks).toHaveLength(1);
    expect(g.currentTrick).toEqual([]);
    expect(original.hands).toEqual(original.initialHands);
    expect(Object.values(g.hands).flat()).toHaveLength(48);
  });
  it.each(Array.from({ length: 32 }, (_, i) => i + 1))(
    "随机合法对局 %i 能完成 52 张出牌且不丢牌",
    (seed) => {
      let g: Game = newGame(seed, seed * 27);
      // Force a variety of contracts, seats, and trumps before a complete rollout.
      g = applyCall(
        g,
        `${(seed % 7) + 1}${(["C", "D", "H", "S", "NT"] as const)[seed % 5]}`,
      );
      for (let i = 0; i < 3; i++) g = applyCall(g, "P");
      while (g.phase === "play") {
        if (g.currentTrick.length === 4) {
          g = collectTrick(g);
          continue;
        }
        const available = legalCards(g);
        const hand = g.hands[g.turn];
        const lead = g.currentTrick[0]?.card[0];
        if (lead && hand.some((c) => c[0] === lead))
          expect(available.every((c) => c[0] === lead)).toBe(true);
        g = playCard(g, available[(seed + g.revision) % available.length]);
      }
      const played = g.tricks.flatMap((t) => t.cards.map((c) => c.card));
      expect(new Set(played).size).toBe(52);
      expect(Object.values(g.hands).flat()).toHaveLength(0);
      expect(g.tricks).toHaveLength(13);
      expect(g.initialHands).toEqual(newGame(seed, seed * 27).hands);
      expect(Number.isFinite(gameResult(g).nsScore)).toBe(true);
    },
  );
});

describe("复式桥牌计分", () => {
  it.each([
    [1, "NT", 0, 7, false, 90],
    [3, "NT", 0, 9, false, 400],
    [3, "NT", 0, 9, true, 600],
    [4, "S", 0, 10, false, 420],
    [4, "H", 0, 11, true, 650],
    [5, "C", 0, 11, false, 400],
    [2, "H", 1, 8, false, 470],
    [2, "H", 2, 9, true, 1240],
    [1, "NT", 1, 7, false, 180],
    [6, "S", 0, 12, false, 980],
    [6, "NT", 0, 12, true, 1440],
    [7, "NT", 0, 13, false, 1520],
    [7, "H", 0, 13, true, 2210],
    [4, "S", 0, 9, false, -50],
    [4, "S", 0, 8, true, -200],
    [4, "H", 1, 9, false, -100],
    [4, "H", 1, 7, false, -500],
    [4, "H", 1, 6, false, -800],
    [4, "H", 1, 8, true, -500],
    [4, "H", 2, 7, true, -1600],
  ] as const)(
    "%i%s 加倍级别 %i 获得 %i 墩，局况 %s → %i",
    (level, strain, doubled, tricks, vulnerable, expected) => {
      expect(
        scoreContract(
          { level, strain, declarer: "S", doubled } as Contract,
          tricks,
          vulnerable,
        ),
      ).toBe(expected);
    },
  );
  it("得分以南北为显示视角，东西完成定约使南北分为负", () => {
    let g = auction(["P", "1NT", "P", "P", "P"]);
    while (g.phase === "play")
      g =
        g.currentTrick.length === 4
          ? collectTrick(g)
          : playCard(g, practiceCard(g));
    expect(gameResult(g).nsScore).toBe(-gameResult(g).score);
  });
});

it("本地练习叫牌始终合法，出牌不越过跟牌规则", () => {
  for (let seed = 1; seed <= 20; seed++) {
    let g = newGame(seed, seed);
    while (g.phase === "auction") {
      const call = practiceCall(g);
      expect(legalCalls(g)).toContain(call);
      g = applyCall(g, call);
    }
    if (g.phase === "play") expect(legalCards(g)).toContain(practiceCard(g));
    expect(nextSeat(g.dealer, 4)).toBe(g.dealer);
    expect(sameSide("N", "S")).toBe(true);
  }
});
