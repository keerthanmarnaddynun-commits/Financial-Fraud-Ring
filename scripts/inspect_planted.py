import json

data = json.load(open("outputs/planted_ring_validation.json", encoding="utf-8"))
for algo in ["connected_components", "label_propagation"]:
    print(f"\n=== ALGORITHM: {algo} ===")
    for r in data["algorithms"][algo]["results_by_pattern_and_camouflage"]:
        print(
            f"{r['pattern']:<7} {r['camouflage']:<4} rec:{r['recovered']}/{r['planted']} "
            f"contained:{r['ring_contained_rate']:.4f} purity:{r['mean_ring_purity']:.4f} "
            f"max_comp:{r['largest_component_size_with_planted']} "
            f"homog:{r['homogeneity']:.4f} comp:{r['completeness']:.4f} "
            f"vm:{r['v_measure']:.4f} contam:{r['contamination']:.4f}"
        )
