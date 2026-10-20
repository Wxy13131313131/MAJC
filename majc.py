"""Standalone IND+C control: shared scores, certified entropy dual, fixed tail.

No model loading, network access, generation, or random sampling at import time.
The default protocol preserves the executed scale experiment's objective.
The disclosure-v1.1 soft reference is an explicit, separately named protocol.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

__version__ = "0.1.0"
Protocol = Literal["executed_scale", "executed_quick", "disclosure_v1_1"]
_PROTOCOLS = {
    "executed_scale": ("sl", "ind"),
    "executed_quick": ("sl", "sl"),
    "disclosure_v1_1": ("ind", "ind"),
}


@dataclass(frozen=True)
class Config:
    tau_kl: float = 0.05
    delta_kl: float = 0.02
    maxiter: int = 50
    tolerance: float = 1e-6

    def __post_init__(self):
        if (not np.isfinite([self.tau_kl, self.delta_kl, self.tolerance]).all()
                or self.tau_kl <= 0 or self.delta_kl < 0 or self.tolerance <= 0
                or isinstance(self.maxiter, bool) or not isinstance(self.maxiter, int)
                or self.maxiter < 1):
            raise ValueError("invalid optimizer configuration")


def _probability_vector(value, name, shape=None):
    out = np.asarray(value, dtype=np.float64)
    if (out.ndim != 1 or not len(out) or (shape is not None and out.shape != shape)
            or not np.isfinite(out).all() or ((out < 0) | (out > 1)).any()):
        raise ValueError(f"{name} must be a finite probability vector of matching shape")
    return out


def _reference(value):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1 or not len(out) or not np.isfinite(out).all() or (out <= 0).any():
        raise ValueError("reference must have finite, strictly positive candidate support")
    if abs(float(out.sum()) - 1) <= 1e-15:
        return out.copy()
    # Scaling first avoids overflow when callers supply unnormalized weights.
    out = out / out.max()
    out = out / out.sum()
    if (out <= 0).any():
        raise ValueError("reference normalization underflowed; no epsilon support repair")
    return out


def kl_divergence(z, p):
    """KL(z || p), in nats; zero coordinates contribute exactly zero."""
    z, p = np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if z.shape != p.shape or (z < 0).any() or (p < 0).any():
        raise ValueError("invalid KL coordinates")
    positive = z > 0
    if (p[positive] == 0).any():
        return float("inf")
    return float(z[positive] @ (np.log(z[positive]) - np.log(p[positive])))


def _signature(p, joint, usable, mean, config):
    digest = hashlib.sha256()
    for vector in (p, joint, usable, mean):
        digest.update(np.ascontiguousarray(vector, dtype="<f8").tobytes())
    digest.update(repr(config).encode("utf8"))
    return digest.hexdigest()


def _solve(p, score, soft_base, constraints, fallback, config):
    """One dual engine for IND and C; acceptance uses original primal units."""
    started = time.perf_counter()
    tau, delta, tolerance = config.tau_kl, config.delta_kl, config.tolerance
    removed = [name for name, vector, _ in constraints if np.ptp(vector) <= 1e-12]
    active = [(name, vector, bound) for name, vector, bound in constraints if name not in removed]
    names = [name for name, _, _ in active]
    scales = np.array([np.ptp(vector) for _, vector, _ in active])
    matrix = (np.array([(vector - bound) / scale
                       for (_, vector, bound), scale in zip(active, scales)])
              if active else np.empty((0, len(p))))

    def diagnostics(z):
        if z.shape != p.shape or not np.isfinite(z).all():
            return {"finite": False, "maximum_residual": None}
        kl = kl_divergence(z, p)
        soft_kl = kl_divergence(z, soft_base)
        gain = float((z - p) @ score) - tau * soft_kl
        baseline_gain = float((fallback - p) @ score) - tau * kl_divergence(fallback, soft_base)
        margins = {name: float(z @ vector) - bound for name, vector, bound in constraints}
        residuals = {"simplex": abs(float(z.sum()) - 1),
                     "negative_probability": max(0., -float(z.min())),
                     "KL": max(0., kl - delta),
                     "objective": max(0., baseline_gain - gain),
                     **{name: max(0., -value) for name, value in margins.items()}}
        return dict(finite=True, KL=kl, soft_KL=soft_kl, regularized_gain_J=gain,
                    gain_J=float((z - p) @ score), margins=margins,
                    residuals=residuals, maximum_residual=max(residuals.values()))

    def result(z, status, solver=None, certificate=None, rejected=None):
        out = dict(status=status, probabilities=z.tolist(), base=p.tolist(),
                   tau_KL=tau, delta_KL=delta, tolerance=tolerance,
                   optimizer_version="majc_shared_entropy_dual_v1",
                   removed_constant_constraints=removed,
                   iterations=int(getattr(solver, "nit", 0)),
                   physical_wall_s=time.perf_counter() - started, **diagnostics(z))
        if solver is not None:
            out.update(solver_success=bool(solver.success), solver_message=str(solver.message))
        if certificate is not None:
            out["optimality_certificate"] = certificate
        if rejected is not None:
            out["rejected_candidate"] = diagnostics(rejected)
        return out

    if (soft_base <= 0).any():
        # A numerical IND zero cannot safely define a strictly positive soft reference.
        return result(fallback.copy(), "FALLBACK_SOFT_SUPPORT_UNDERFLOW")
    if len(p) == 1 or delta == 0:
        return result(fallback.copy(), "ZERO_BUDGET")
    if np.ptp(score) == 0 and np.array_equal(soft_base, fallback):
        return result(fallback.copy(), "FLAT_OBJECTIVE")

    logp = np.log(p)
    centered = (score - float(p @ score)) / tau + np.log(soft_base) - logp
    gap_tolerance = min(1e-8, tolerance * 0.1)

    def dual(x):
        temperature = x[0]
        logits = logp + (centered + x[1:] @ matrix) / temperature
        peak = float(logits.max())
        normalizer = peak + float(np.log(np.exp(logits - peak).sum()))
        logz = logits - normalizer
        z = np.exp(logz)
        kl = float(z @ (logz - logp))
        value = temperature * normalizer + (temperature - 1) * delta
        gradient = np.concatenate(([delta - kl], matrix @ z))
        return value, gradient, z

    def objective(x):
        value, gradient, _ = dual(x)
        return value, gradient

    solved = minimize(objective, np.array([1.] + [0.] * len(names)), jac=True,
                      method="L-BFGS-B", bounds=[(1., None)] + [(0., None)] * len(names),
                      options=dict(maxiter=config.maxiter, ftol=1e-15, gtol=1e-10, maxls=40))
    x = np.asarray(solved.x, dtype=np.float64)
    if (x.shape != (1 + len(names),) or not np.isfinite(x).all()
            or x[0] < 1 or (x[1:] < 0).any()):
        return result(fallback.copy(), "SOLVER_FALLBACK", solved,
                      {"certified": False, "reason": "invalid_dual_variables"})
    value, gradient, candidate = dual(x)
    check = diagnostics(candidate)
    upper = tau * value
    gap = upper - check["regularized_gain_J"]
    projected = gradient.copy()
    at_bound = x <= np.array([1.] + [0.] * len(names)) + 1e-12
    projected[at_bound] = np.minimum(projected[at_bound], 0)
    valid = bool(check["finite"] and check["maximum_residual"] <= tolerance
                 and np.isfinite(gap) and -gap_tolerance <= gap <= gap_tolerance)
    certificate = dict(certified=valid, dual_upper_bound_regularized_gain=upper,
                       primal_regularized_gain=check["regularized_gain_J"],
                       primal_dual_gap=gap, gap_tolerance=gap_tolerance,
                       projected_dual_gradient_max=float(np.abs(projected).max()),
                       multipliers={"KL": tau * (x[0] - 1),
                                    **{name: float(tau * value / scale)
                                       for name, value, scale in zip(names, x[1:], scales)}})
    return result(candidate if valid else fallback.copy(),
                  "SOLVED" if valid else "SOLVER_FALLBACK", solved,
                  certificate, None if valid else candidate)


def optimize_ind(p, J_ind, U, M, *, config=Config()):
    """IND: maximize z@J_ind - tau*KL(z||p), protect U/M at p."""
    p = _reference(p)
    joint, usable, mean = [_probability_vector(v, n, p.shape)
                           for v, n in ((J_ind, "J_ind"), (U, "U"), (M, "M"))]
    constraints = [("U", usable, float(p @ usable)), ("M", mean, float(p @ mean))]
    result = _solve(p, joint, p, constraints, p, config)
    result["input_signature"] = _signature(p, joint, usable, mean, config)
    result["fallback_distribution"] = "SL"
    return result


def optimize_ind_c(p, J_ind, J_c, U, M, *, config=Config(),
                   protocol: Protocol = "executed_scale", ind_result=None):
    """C adjustment on the same support and original hard KL ball.

    Cached IND results must originate from optimize_ind with identical inputs.
    No failure path samples a token or reverts C to SL instead of IND.
    """
    if protocol not in _PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    p = _reference(p)
    joint, c_joint, usable, mean = [_probability_vector(v, n, p.shape) for v, n in
                                   ((J_ind, "J_ind"), (J_c, "J_c"), (U, "U"), (M, "M"))]
    if ind_result is None:
        ind_result = optimize_ind(p, joint, usable, mean, config=config)
    expected = _signature(p, joint, usable, mean, config)
    if ind_result.get("input_signature") != expected:
        raise ValueError("cached IND solve does not match the scores/reference/configuration")
    anchor = _probability_vector(ind_result["probabilities"], "IND anchor", p.shape)
    if (abs(float(anchor.sum()) - 1) > config.tolerance
            or kl_divergence(anchor, p) > config.delta_kl + config.tolerance
            or min(float((anchor - p) @ usable), float((anchor - p) @ mean)) < -config.tolerance):
        raise ValueError("IND anchor violates original constraints")
    soft_name, protection_name = _PROTOCOLS[protocol]
    protected = anchor if protection_name == "ind" else p
    constraints = [("U", usable, float(protected @ usable)),
                   ("M", mean, float(protected @ mean)),
                   ("IND", joint, float(anchor @ joint))]
    soft = anchor if soft_name == "ind" else p
    out = _solve(p, c_joint, soft, constraints, anchor, config)
    out.update(protocol=protocol, soft_reference=soft_name, protection_reference=protection_name,
               hard_reference="sl", ind_anchor=ind_result, ind_floor=float(anchor @ joint),
               fallback_distribution="IND", expected_IND=float(np.asarray(out["probabilities"]) @ joint))
    return out


def preserve_tail(reference, candidate_ids, conditional):
    """Lift a conditional candidate distribution without changing other tokens."""
    reference = _probability_vector(reference, "full reference")
    if abs(float(reference.sum()) - 1) > 1e-6:
        raise ValueError("full reference is not normalized")
    raw_ids = np.asarray(candidate_ids)
    if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu" or not len(raw_ids):
        raise ValueError("candidate ids must be a nonempty integer vector")
    ids = raw_ids.astype(np.int64)
    if len(set(ids.tolist())) != len(ids) or (ids < 0).any() or (ids >= len(reference)).any():
        raise ValueError("invalid or duplicate candidate ids")
    z = _probability_vector(conditional, "conditional", ids.shape)
    if abs(float(z.sum()) - 1) > 1e-6 or (reference[ids] <= 0).any():
        raise ValueError("candidate simplex/support mismatch")
    adjusted = reference.copy()
    mass = float(reference[ids].sum())
    adjusted[ids] = mass * z
    outside = np.ones(len(reference), dtype=bool)
    outside[ids] = False
    return adjusted, dict(candidate_mass=mass, full_probability_sum=float(adjusted.sum()),
                          tail_exactly_unchanged=bool(np.array_equal(adjusted[outside], reference[outside])),
                          full_KL=kl_divergence(adjusted, reference))


def candidate_scores(u, q, eta_c, *, weights=None):
    """u=P(U), q_j=P(r_j|U), eta_c=P(all r|U); distinct events only.

    Exact q=0/1 stays exact. Event weights affect M, never the distinct-event product.
    Caller must resolve UNKNOWNs before using this numeric API.
    """
    u = _probability_vector(u, "u")
    eta = _probability_vector(eta_c, "eta_c", u.shape)
    q = np.asarray(q, dtype=np.float64)
    if (q.ndim != 2 or q.shape[0] != len(u) or not np.isfinite(q).all()
            or ((q < 0) | (q > 1)).any()):
        raise ValueError("q must be [candidates, distinct_events] in [0,1]")
    count = q.shape[1]
    if count == 0:
        return dict(J_ind=u.copy(), J_c=u.copy(), U=u.copy(), M=u.copy(), event_count=0)
    w = np.ones(count) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (count,) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("weights must be finite positive event multiplicities")
    w = w / w.max()
    return dict(J_ind=u * q.prod(axis=1), J_c=u * eta, U=u.copy(),
                M=u * (q @ w / w.sum()), event_count=count)


def control(reference, candidate_ids, u, q, eta_c, *, weights=None,
            config=Config(), protocol: Protocol = "executed_scale", enable_c=True):
    """One deterministic control step; caller chooses top-k once and samples once.

    Candidate order belongs to the LM adapter (e.g. torch.topk tie ordering),
    so this function deliberately does not reselect candidates.
    """
    if protocol not in _PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    full = _probability_vector(reference, "reference")
    # Validate indices and full support before indexing or calling a solver.
    ids = np.asarray(candidate_ids)
    n = len(ids) if ids.ndim == 1 else 0
    if not n:
        raise ValueError("empty candidate set; apply the shared guard's explicit policy first")
    preserve_tail(full, ids, np.full(n, 1 / n))
    ids = ids.astype(np.int64)
    scores = candidate_scores(u, q, eta_c, weights=weights)
    if len(scores["U"]) != n:
        raise ValueError("prediction count does not match the candidate set")
    p = full[ids] / full[ids].sum()
    anchor = optimize_ind(p, scores["J_ind"], scores["U"], scores["M"], config=config)
    if enable_c and scores["event_count"] >= 2:
        solved = optimize_ind_c(p, scores["J_ind"], scores["J_c"], scores["U"], scores["M"],
                                config=config, protocol=protocol, ind_result=anchor)
        reason = "C_ATTEMPTED"
    else:
        solved = anchor
        reason = "C_DISABLED" if not enable_c else "FEWER_THAN_TWO_EVENTS"
    adjusted, tail = preserve_tail(full, ids, solved["probabilities"])
    return adjusted, dict(candidate_ids=ids.tolist(), event_count=scores["event_count"],
                          stage2=reason, optimizer=solved, ind_anchor=anchor,
                          protocol=protocol, **tail)
