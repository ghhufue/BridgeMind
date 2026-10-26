// Export real frontend decisions for the Python/OpenSpiel parity check.
import { createServer } from "vite";
const server = await createServer({
  configFile: false,
  cacheDir: "../.build/frontend/vite-parity",
  server: { middlewareMode: true },
  appType: "custom",
  logLevel: "silent",
});
try {
  const engine = await server.ssrLoadModule("/src/bridge/engine.ts");
  const boards = [];
  for (let seed = 1; seed <= 64; seed++) {
    let g = engine.newGame(seed, 7823 * seed);
    const initial = structuredClone(g);
    const auction = [];
    let rng = seed;
    const random = () => {
      rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0;
      return rng / 4294967296;
    };
    while (g.phase === "auction") {
      const legal = engine.legalCalls(g);
      const roll = random();
      let call;
      if (seed % 8 === 0 || roll < 0.4) call = "P";
      else if (roll < 0.6 && legal.includes("X")) call = "X";
      else if (roll < 0.7 && legal.includes("XX")) call = "XX";
      else call = legal[Math.floor(random() * legal.length)];
      auction.push({ turn: g.turn, legal, call });
      g = engine.applyCall(g, call);
    }
    const play = [];
    while (g.phase === "play") {
      if (g.currentTrick.length === 4) {
        g = engine.collectTrick(g);
        continue;
      }
      const legal = engine.legalCards(g);
      const card = legal[Math.floor(random() * legal.length)];
      play.push({
        toPlay: g.turn,
        controller: engine.controller(g),
        legal,
        card,
      });
      g = engine.playCard(g, card);
    }
    boards.push({
      initial,
      auction,
      contract: g.contract,
      play,
      tricks: g.tricks,
      result: engine.gameResult(g),
    });
  }
  const scores = [];
  for (let level = 1; level <= 7; level++)
    for (const strain of ["C", "D", "H", "S", "NT"])
      for (const doubled of [0, 1, 2])
        for (const vulnerable of [false, true])
          for (let tricks = 0; tricks <= 13; tricks++) {
            const contract = { level, strain, doubled, declarer: "S" };
            scores.push({
              contract,
              tricks,
              vulnerable,
              score: engine.scoreContract(contract, tricks, vulnerable),
            });
          }
  process.stdout.write(JSON.stringify({ boards, scores }));
} finally {
  await server.close();
}
