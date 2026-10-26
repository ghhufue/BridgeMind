import {
  RANKS,
  SEATS,
  STRAINS,
  SUITS,
  type AuctionEntry,
  type Call,
  type Card,
  type Contract,
  type Game,
  type Hands,
  type PlayedCard,
  type Seat,
  type Strain,
} from "./types";

export const seatName: Record<Seat, string> = {
  N: "北",
  E: "东",
  S: "南",
  W: "西",
};
export const suitSymbol: Record<Strain, string> = {
  C: "♣",
  D: "♦",
  H: "♥",
  S: "♠",
  NT: "NT",
};
export const nextSeat = (seat: Seat, offset = 1): Seat =>
  SEATS[(SEATS.indexOf(seat) + offset) % 4];
export const sameSide = (a: Seat, b: Seat) =>
  SEATS.indexOf(a) % 2 === SEATS.indexOf(b) % 2;
export const cardRank = (card: Card) => RANKS.indexOf(card[1]);
export const cardLabel = (card: Card) =>
  `${suitSymbol[card[0] as Strain]}${card[1] === "T" ? "10" : card[1]}`;
export const callLabel = (call: Call) =>
  call === "P"
    ? "Pass"
    : call === "X"
      ? "加倍"
      : call === "XX"
        ? "再加倍"
        : `${call[0]}${suitSymbol[call.slice(1) as Strain]}`;
export const sortHand = (hand: Card[]) =>
  [...hand].sort(
    (a, b) =>
      SUITS.indexOf(b[0] as (typeof SUITS)[number]) -
        SUITS.indexOf(a[0] as (typeof SUITS)[number]) ||
      cardRank(b) - cardRank(a),
  );
export const highCardPoints = (hand: Card[]) =>
  hand.reduce((sum, c) => sum + Math.max(0, cardRank(c) - 8), 0);
export const dummySeat = (g: Game) =>
  g.contract ? nextSeat(g.contract.declarer, 2) : null;
export const dummyVisible = (g: Game) =>
  g.tricks.length > 0 ||
  g.currentTrick.length > 0 ||
  (g.phase === "finished" && !!g.contract);
export const controller = (g: Game): Seat =>
  g.phase === "play" && g.turn === dummySeat(g) ? g.contract!.declarer : g.turn;
export function stateId(g: Game) {
  // An undo followed by a different move can reuse the revision. Include the
  // public action history so services never confuse those two positions.
  const actions = JSON.stringify([
    g.phase,
    g.auction,
    g.tricks,
    g.currentTrick,
  ]);
  let hash = 0xcbf29ce484222325n;
  for (const ch of actions)
    hash = BigInt.asUintN(
      64,
      (hash ^ BigInt(ch.charCodeAt(0))) * 0x100000001b3n,
    );
  return `${g.id}:${g.revision}:${hash.toString(16)}`;
}
export const trickCounts = (g: Game): [number, number] => [
  g.tricks.filter((t) => sameSide(t.winner, "S")).length,
  g.tricks.filter((t) => sameSide(t.winner, "E")).length,
];

export function newGame(board = 1, seed?: number): Game {
  let value = seed ?? crypto.getRandomValues(new Uint32Array(1))[0];
  const random = () => {
    value += 0x6d2b79f5;
    let t = Math.imul(value ^ (value >>> 15), 1 | value);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const deck = SUITS.flatMap((s) => [...RANKS].map((r) => `${s}${r}` as Card));
  for (let i = deck.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  const hands = Object.fromEntries(
    SEATS.map((s, i) => [s, sortHand(deck.slice(i * 13, i * 13 + 13))]),
  ) as Hands;
  const dealer = SEATS[(board - 1) % 4];
  const vulCycle = [0, 1, 2, 3, 1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2][
    (board - 1) % 16
  ];
  // getRandomValues also works over plain LAN HTTP; randomUUID requires a
  // secure context and would crash on an iPhone visiting http://192.168.x.x.
  const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");
  return {
    id,
    board,
    revision: 0,
    dealer,
    vulnerable: [!!(vulCycle & 1), !!(vulCycle & 2)],
    phase: "auction",
    turn: dealer,
    initialHands: structuredClone(hands),
    hands,
    auction: [],
    contract: null,
    currentTrick: [],
    tricks: [],
  };
}

const bidIndex = (call: Call) =>
  /^[1-7](C|D|H|S|NT)$/.test(call)
    ? (Number(call[0]) - 1) * 5 + STRAINS.indexOf(call.slice(1) as Strain)
    : -1;

export function legalCalls(g: Game): Call[] {
  if (g.phase !== "auction") return [];
  const lastBid = g.auction.findLast((e) => bidIndex(e.call) >= 0);
  const lastAction = g.auction.findLast((e) => e.call !== "P");
  const result: Call[] = ["P"];
  if (
    lastBid &&
    lastAction?.call !== "X" &&
    lastAction?.call !== "XX" &&
    !sameSide(g.turn, lastBid.seat)
  )
    result.push("X");
  if (lastBid && lastAction?.call === "X" && sameSide(g.turn, lastBid.seat))
    result.push("XX");
  for (let i = (lastBid ? bidIndex(lastBid.call) : -1) + 1; i < 35; i++)
    result.push(`${Math.floor(i / 5) + 1}${STRAINS[i % 5]}` as Call);
  return result;
}

export function contractFromAuction(auction: AuctionEntry[]): Contract | null {
  const lastBidIndex = auction.findLastIndex((e) => bidIndex(e.call) >= 0);
  if (lastBidIndex < 0) return null;
  const last = auction[lastBidIndex];
  const strain = last.call.slice(1) as Strain;
  const declarer = auction.find(
    (e) =>
      bidIndex(e.call) >= 0 &&
      e.call.slice(1) === strain &&
      sameSide(e.seat, last.seat),
  )!.seat;
  const actions = auction.slice(lastBidIndex + 1);
  const doubled = actions.some((e) => e.call === "XX")
    ? 2
    : actions.some((e) => e.call === "X")
      ? 1
      : 0;
  return { level: Number(last.call[0]), strain, declarer, doubled };
}

export function applyCall(g: Game, call: Call): Game {
  if (!legalCalls(g).includes(call)) throw new Error("当前叫品不合法");
  const auction = [...g.auction, { seat: g.turn, call }];
  const updated = {
    ...g,
    revision: g.revision + 1,
    auction,
    turn: nextSeat(g.turn),
  };
  const hasBid = auction.some((e) => bidIndex(e.call) >= 0);
  if (!hasBid && auction.length === 4) return { ...updated, phase: "finished" };
  if (
    hasBid &&
    auction.length >= 4 &&
    auction.slice(-3).every((e) => e.call === "P")
  ) {
    const contract = contractFromAuction(auction)!;
    return {
      ...updated,
      phase: "play",
      contract,
      turn: nextSeat(contract.declarer),
    };
  }
  return updated;
}

export function legalCards(g: Game): Card[] {
  if (g.phase !== "play" || g.currentTrick.length === 4) return [];
  const hand = g.hands[g.turn];
  const suit = g.currentTrick[0]?.card[0];
  const following = hand.filter((c) => c[0] === suit);
  return following.length ? following : hand;
}

export function trickWinner(cards: PlayedCard[], trump: Strain): Seat {
  if (!cards.length) throw new Error("空墩没有赢家");
  return cards.reduce((best, entry) => {
    if (entry.card[0] === best.card[0])
      return cardRank(entry.card) > cardRank(best.card) ? entry : best;
    return entry.card[0] === trump ? entry : best;
  }).seat;
}

export function playCard(g: Game, card: Card): Game {
  if (!legalCards(g).includes(card))
    throw new Error("必须跟出首引花色；不可重复出牌");
  const currentTrick = [...g.currentTrick, { seat: g.turn, card }];
  return {
    ...g,
    revision: g.revision + 1,
    hands: { ...g.hands, [g.turn]: g.hands[g.turn].filter((c) => c !== card) },
    currentTrick,
    turn: currentTrick.length === 4 ? g.turn : nextSeat(g.turn),
  };
}

export function collectTrick(g: Game): Game {
  if (g.phase !== "play" || g.currentTrick.length !== 4 || !g.contract)
    throw new Error("本墩尚未结束");
  const winner = trickWinner(g.currentTrick, g.contract.strain);
  const tricks = [...g.tricks, { cards: g.currentTrick, winner }];
  return {
    ...g,
    revision: g.revision + 1,
    tricks,
    currentTrick: [],
    turn: winner,
    phase: tricks.length === 13 ? "finished" : "play",
  };
}

// Duplicate bridge score, from the declarer's side. No rubber or IMP conversion.
export function scoreContract(
  contract: Contract,
  tricks: number,
  vulnerable: boolean,
): number {
  const delta = tricks - (6 + contract.level);
  const doubled = contract.doubled;
  if (delta < 0) {
    const down = -delta;
    if (!doubled) return -down * (vulnerable ? 100 : 50);
    const penalty = vulnerable
      ? 200 + 300 * (down - 1)
      : 100 + 200 * Math.min(down - 1, 2) + 300 * Math.max(down - 3, 0);
    return -penalty * (doubled === 2 ? 2 : 1);
  }
  const unit = contract.strain === "C" || contract.strain === "D" ? 20 : 30;
  const points =
    (contract.level * unit + (contract.strain === "NT" ? 10 : 0)) *
    (1 << doubled);
  let bonus = points >= 100 ? (vulnerable ? 500 : 300) : 50;
  if (contract.level === 6) bonus += vulnerable ? 750 : 500;
  if (contract.level === 7) bonus += vulnerable ? 1500 : 1000;
  if (doubled) bonus += 50 * (1 << (doubled - 1));
  const overtricks = doubled
    ? delta * (vulnerable ? 200 : 100) * (1 << (doubled - 1))
    : delta * unit;
  return points + bonus + overtricks;
}

export function gameResult(g: Game) {
  if (!g.contract) return { tricks: 0, delta: 0, score: 0, nsScore: 0 };
  const tricks = g.tricks.filter((t) =>
    sameSide(t.winner, g.contract!.declarer),
  ).length;
  const score = scoreContract(
    g.contract,
    tricks,
    g.vulnerable[sameSide(g.contract.declarer, "S") ? 0 : 1],
  );
  return {
    tricks,
    delta: tricks - 6 - g.contract.level,
    score,
    nsScore: sameSide(g.contract.declarer, "S") ? score : -score,
  };
}

// Only a transparent local practice partner. Never presented as model inference.
export function practiceCall(g: Game): Call {
  if (g.auction.some((e) => bidIndex(e.call) >= 0)) return "P";
  const hand = g.hands[g.turn];
  const points = highCardPoints(hand);
  if (points < 12) return "P";
  const lengths = STRAINS.slice(0, 4).map(
    (s) => hand.filter((c) => c[0] === s).length,
  );
  const balanced = Math.min(...lengths) >= 2 && Math.max(...lengths) <= 5;
  if (balanced && points >= 15 && points <= 17) return "1NT";
  const longest = [...SUITS]
    .reverse()
    .sort(
      (a, b) =>
        hand.filter((c) => c[0] === b).length -
        hand.filter((c) => c[0] === a).length,
    )[0];
  const suit =
    lengths[SUITS.indexOf(longest)] >= 5 || longest === "C" || longest === "D"
      ? longest
      : lengths[1] >= lengths[0]
        ? "D"
        : "C";
  return `1${suit}` as Call;
}

export function practiceCard(g: Game): Card {
  return [...legalCards(g)].sort(
    (a, b) => cardRank(a) - cardRank(b) || a.localeCompare(b),
  )[0];
}
