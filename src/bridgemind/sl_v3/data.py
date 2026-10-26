"""Certified twelve-family curriculum with disjoint full-deal symmetry orbits."""

from bridgemind.sl_v2.data import augment, orbit_key, Curriculum
from bridgemind.foundation.cases import TEMPLATES as OLD, ROLES, build_base, observe
from bridgemind.foundation.certificates import certify as certify_old
from .courses import TEMPLATES, build_case, certify

OLD_FAMILIES = tuple(k for k, v in OLD.items() if not v.get("mechanical") and not v.get("control"))


def prepare_courses(solver, *, train_seeds, valid_seeds, excluded_orbits=()):
    if set(train_seeds) & set(valid_seeds):
        raise ValueError("Overlapping curriculum seeds")
    banned, splits, audits = set(excluded_orbits), {}, {}
    families = (*OLD_FAMILIES, *TEMPLATES)
    for split, seeds in (("valid", valid_seeds), ("train", train_seeds)):
        cases, seen, rejected = [], set(), {}
        for seed in seeds:
            for family in families:
                for role in ROLES:
                    case = build_base(family, seed, role) if family in OLD else build_case(family, seed, role)
                    orbit = orbit_key(case["board"]["owners"])
                    if orbit in banned:
                        rejected["excluded_orbit"] = rejected.get("excluded_orbit", 0)+1
                        continue
                    certificate = certify_old(observe(case), solver) if family in OLD else certify(case, solver)
                    if certificate["status"] != "certified":
                        reason = certificate["reason"]
                        rejected[reason] = rejected.get(reason, 0)+1
                        continue
                    case.update(certificate=certificate, orbit=orbit)
                    cases.append(case); seen.add(orbit)
            print(f"V3 course {split}: seed {seed}, certified {len(cases)}", flush=True)
        if {c["family"] for c in cases} != set(families):
            raise ValueError("Missing a required old/new tactical family")
        splits[split] = cases
        audits[split] = {"cases": len(cases), "distinct_deal_orbits": len(seen), "seeds": list(seeds), "rejected": rejected,
                         "by_family": {f: sum(c["family"] == f for c in cases) for f in families}}
        banned.update(seen)
    return splits, audits
