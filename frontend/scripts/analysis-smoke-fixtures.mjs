import { createServer } from "vite";
import { mkdirSync, writeFileSync } from "node:fs";

const server = await createServer({
  configFile: false,
  cacheDir: "../.build/frontend/vite-parity",
  server: { middlewareMode: true },
  appType: "custom",
  logLevel: "silent",
});
try {
  const e = await server.ssrLoadModule("/src/bridge/engine.ts");
  const { analysisRequest } = await server.ssrLoadModule(
    "/src/api/analysis.ts",
  );
  let game = e.newGame(3, 20261026);
  for (const call of ["1NT", "P", "P", "P"]) game = e.applyCall(game, call);
  mkdirSync("../.build/lan", { recursive: true });
  writeFileSync(
    "../.build/lan/ai-opening.json",
    JSON.stringify(analysisRequest(game, "ai")),
  );
  game = e.playCard(game, e.legalCards(game)[0]);
  writeFileSync(
    "../.build/lan/ai-dummy.json",
    JSON.stringify(analysisRequest(game, "ai")),
  );
  while (game.tricks.length < 10)
    game =
      game.currentTrick.length === 4
        ? e.collectTrick(game)
        : e.playCard(game, e.practiceCard(game));
  writeFileSync(
    "../.build/lan/ai-request.json",
    JSON.stringify(analysisRequest(game, "ai")),
  );
  writeFileSync(
    "../.build/lan/dds-request.json",
    JSON.stringify(analysisRequest(game, "dds")),
  );
  process.stdout.write(
    "Generated frontend analysis requests for real LAN smoke tests.\n",
  );
} finally {
  await server.close();
}
