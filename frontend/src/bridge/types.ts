export const SEATS = ["N", "E", "S", "W"] as const;
export const SUITS = ["C", "D", "H", "S"] as const;
export const STRAINS = ["C", "D", "H", "S", "NT"] as const;
export const RANKS = "23456789TJQKA";
export type Seat = (typeof SEATS)[number];
export type Suit = (typeof SUITS)[number];
export type Strain = (typeof STRAINS)[number];
export type Card = `${Suit}${string}`;
export type Call = "P" | "X" | "XX" | `${number}${Strain}`;
export type Hands = Record<Seat, Card[]>;
export interface AuctionEntry {
  seat: Seat;
  call: Call;
}
export interface PlayedCard {
  seat: Seat;
  card: Card;
}
export interface Trick {
  cards: PlayedCard[];
  winner: Seat;
}
export interface Contract {
  level: number;
  strain: Strain;
  declarer: Seat;
  doubled: 0 | 1 | 2;
}
export interface Game {
  id: string;
  board: number;
  revision: number;
  dealer: Seat;
  vulnerable: [boolean, boolean];
  phase: "auction" | "play" | "finished";
  turn: Seat;
  initialHands: Hands;
  hands: Hands;
  auction: AuctionEntry[];
  contract: Contract | null;
  currentTrick: PlayedCard[];
  tricks: Trick[];
}
