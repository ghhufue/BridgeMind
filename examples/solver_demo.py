"""Run after starting bridgemind-dds. Fixture: DDS examples/hands.cpp, Apache-2.0."""

import argparse
import json

from bridgemind.dds import Position, SolverClient, parse_pbn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--auction", action="store_true", help="Also exercise the optional OpenSpiel bridge environment")
    args = parser.parse_args()
    deal = parse_pbn("N:QJ6.K652.J85.T98 873.J97.AT764.Q4 K5.T83.KQ9.A7652 AT942.AQ4.32.KJ3")
    with SolverClient(port=args.port) as client:
        output = {"service": client.get_info(), "table": client.solve_table(deal).to_dict()}
        position = Position({"N": [], "E": ["SK"], "S": ["SQ"], "W": ["SJ"]}, "NT", "N", ["SA"])
        output["position"] = client.solve_position(position).to_dict()
        if args.auction:
            from bridgemind.engine import BridgeBiddingEnv
            env = BridgeBiddingEnv(client, deal)
            for call in ("1NT", "P", "P", "P"):
                env.apply_call(call)
            output["auction_result"] = env.result()
            env.close()
        output["stats"] = client.get_stats()
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

