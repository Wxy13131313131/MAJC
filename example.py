"""Synthetic one-step example, not an experimental performance result."""
import json
from majc import control

if __name__ == "__main__":
    adjusted, audit = control(
        reference=[.32, .24, .16, .08, .13, .07, 0.],
        candidate_ids=[0, 1, 2, 3],
        u=[.9] * 4, q=[[.7, .7]] * 4,
        eta_c=[.55, .48, .44, .42],
        protocol="disclosure_v1_1",
    )
    print(json.dumps({"probabilities": adjusted.tolist(), "audit": audit}, indent=2))
