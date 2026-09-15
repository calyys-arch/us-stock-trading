"""
G-Mamba-lite: vendored, trimmed copy of a graph-conditioned selective-state-
space model, adapted from Universal-Hybrid-AI's
`industry-packs/trading-signals` pack (python-svc/gmamba_signal.py,
commit 9ae2f54, cloned 2026-09-15) for a one-off Phase 0 kill-test of
whether that architecture finds any signal on THIS repo's own cross-
sectional universe — see docs/gmamba_phase0_results.md.

Why vendored instead of imported from that repo: Universal-Hybrid-AI is a
separate product/repo (private, calyys-arch/Universal-Hybrid-AI) with its
own release cycle; this repo has no dependency on it and should not gain
one for a single research spike. Copying ~250 lines of framework-agnostic
PyTorch (no GreyCat/UHAI runtime coupling in the original file either) is
cheaper and more honest than a git submodule for a script that may never
run again after this Phase 0 verdict.

What was trimmed from the original (disclosed, not silent):
  - The three optional structured regularization losses (eq. 32-34:
    edge-aware consistency, spectral smoothness, dynamic-graph sparsity/
    symmetry) — the original file's own docstring says these default to
    weight 0.0 ("recovers plain prediction loss when structural priors are
    unreliable"), which is exactly this spike's situation (see
    scripts/run_gmamba_phase0.py's module docstring on why static_adj is
    an all-zero placeholder here). Only prediction_loss (eq. 31, MAE) is
    kept.
  - Nothing else changed: GMambaConfig, SelectiveScanMamba,
    graph_conditioned_selective_scan, BiMamba, normalize_adjacency,
    DynamicAdjacency, mix_adjacency, GraphPropagation,
    GraphConditionedGate, GatedFusion, and GMambaSignal are byte-for-byte
    logic ports (renamed nothing, restructured nothing) of the original.
    Equation numbers in comments refer to the same source paper:
    Chen, X. & Tang, Q., Neurocomputing 680 (2026) 133280,
    doi.org/10.1016/j.neucom.2026.133280 — reimplemented there from the
    paper's equations since the paper itself ships no code.

Not part of any live/paper-trading path. Not reachable from
configs/strategy.yaml, python/backtest/param_guard.py's Chan 5-parameter
budget, or scripts/self_improve_loop.py. See scripts/run_gmamba_phase0.py
for the falsification-first discipline this is gated behind.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GMambaConfig:
    n_features: int
    d_model: int = 64
    n_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    d_attn: int = 16
    dropout: float = 0.1
    horizon: int = 1

    def __post_init__(self) -> None:
        self.d_inner = self.expand * self.d_model
        if self.dt_rank is None:
            self.dt_rank = max(1, math.ceil(self.d_model / 16))

    def param_count(self, model: nn.Module | None = None) -> int:
        """Convenience for the Phase 0 script's honesty-about-scope report
        (this is not a Chan-5-parameter model and the write-up should say
        so with an exact number, not a hand-wave)."""
        if model is None:
            model = GMambaSignal(self)
        return sum(p.numel() for p in model.parameters())


class SelectiveScanMamba(nn.Module):
    def __init__(self, cfg: GMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_model, d_inner, d_state, dt_rank, d_conv = (
            cfg.d_model, cfg.d_inner, cfg.d_state, cfg.dt_rank, cfg.d_conv,
        )
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
            padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        b, l, _ = x.shape
        x_and_res = self.in_proj(x)
        x_in, res = x_and_res.split(self.cfg.d_inner, dim=-1)

        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :l]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)

        y = self._ssm(x_in, gate)
        y = y * F.silu(res)
        return self.out_proj(y)

    def _ssm(self, x: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        d_inner, n = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x)
        delta, b_proj, c_proj = x_dbl.split(
            [self.cfg.dt_rank, self.cfg.d_state, self.cfg.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        return graph_conditioned_selective_scan(
            x, delta, A, b_proj, c_proj, self.D, gate=gate,
        )


def graph_conditioned_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    b, l, d_inner = u.shape
    n = A.shape[1]

    delta_a = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
    delta_b_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)

    if gate is not None:
        g = gate.unsqueeze(-1)
        delta_a = delta_a * g
        delta_b_u = delta_b_u * g

    state = torch.zeros(b, d_inner, n, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(l):
        state = delta_a[:, t] * state + delta_b_u[:, t]
        c_t = C[:, t, :]
        if gate is not None:
            c_t = c_t.unsqueeze(1) * gate[:, t, :].unsqueeze(-1)
            y_t = torch.einsum("bdn,bdn->bd", state, c_t)
        else:
            y_t = torch.einsum("bdn,bn->bd", state, c_t)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)
    return y + u * D


class BiMamba(nn.Module):
    def __init__(self, cfg: GMambaConfig) -> None:
        super().__init__()
        self.fwd = SelectiveScanMamba(cfg)
        self.bwd = SelectiveScanMamba(cfg)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, u: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        u_n = self.norm(u)
        y = u + self.dropout(self.fwd(u_n, gate)) + self.dropout(
            self.bwd(u_n.flip(1), gate.flip(1) if gate is not None else None).flip(1)
        )
        return y


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    v = adj.shape[-1]
    eye = torch.eye(v, device=adj.device, dtype=adj.dtype)
    a = adj + eye
    deg = a.sum(-1).clamp_min(1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(-1) * a * d_inv_sqrt.unsqueeze(-2)


class DynamicAdjacency(nn.Module):
    def __init__(self, cfg: GMambaConfig) -> None:
        super().__init__()
        self.phi = nn.Linear(cfg.d_model, cfg.d_attn, bias=False)
        self.psi = nn.Linear(cfg.d_model, cfg.d_attn, bias=False)
        self.d_attn = cfg.d_attn

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        z = self.phi(pooled)
        q = self.psi(pooled)
        scores = torch.einsum("bvd,bwd->bvw", z, q) / math.sqrt(self.d_attn)
        return F.softmax(scores, dim=-1)


def mix_adjacency(a_stat: torch.Tensor, a_dyn: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    mixed = alpha * a_stat + (1 - alpha) * a_dyn
    return normalize_adjacency(mixed)


class GraphPropagation(nn.Module):
    def __init__(self, cfg: GMambaConfig) -> None:
        super().__init__()
        self.w_s = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, adj: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("bvw,blwd->blvd", adj, u)
        return F.relu(self.w_s(propagated))


class GraphConditionedGate(nn.Module):
    def __init__(self, cfg: GMambaConfig) -> None:
        super().__init__()
        self.w_a = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_inner),
        )

    def forward(self, adj: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        neighbor_ctx = torch.einsum("bvw,blwd->blvd", adj, self.w_a(u))
        return torch.sigmoid(self.mlp(neighbor_ctx))


class GatedFusion(nn.Module):
    def __init__(self, cfg: GMambaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model * 2, cfg.d_model)

    def forward(self, temporal: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate_proj(torch.cat([temporal, spatial], dim=-1)))
        return g * temporal + (1 - g) * spatial


class GMambaSignal(nn.Module):
    """Input:  X (B, L, V, C). static_adj (V, V), already
    normalize_adjacency()'d by the caller once (or an all-zero placeholder
    -- see this module's docstring on why that is this spike's honest
    starting point). Output: y_hat (B, T, V), aux dict with mixed_adj/a_dyn.
    """

    def __init__(self, cfg: GMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.dyn_adj = DynamicAdjacency(cfg)
        self.theta_alpha = nn.Parameter(torch.zeros(1))
        self.gate = GraphConditionedGate(cfg)
        self.encoder_layers = nn.ModuleList(BiMamba(cfg) for _ in range(cfg.n_layers))
        self.graph_prop = GraphPropagation(cfg)
        self.fusion = GatedFusion(cfg)
        self.decoder = SelectiveScanMamba(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.horizon)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor) -> tuple[torch.Tensor, dict]:
        b, l, v, _ = x.shape
        u = self.in_proj(x)

        pooled = u.mean(dim=1)
        a_dyn = self.dyn_adj(pooled)
        alpha = torch.sigmoid(self.theta_alpha)
        mixed_adj = mix_adjacency(static_adj.unsqueeze(0).expand(b, -1, -1), a_dyn, alpha)
        gate = self.gate(mixed_adj, u)

        u_flat = u.permute(0, 2, 1, 3).reshape(b * v, l, self.cfg.d_model)
        gate_flat = gate.permute(0, 2, 1, 3).reshape(b * v, l, self.cfg.d_inner)
        h = u_flat
        for layer in self.encoder_layers:
            h = layer(h, gate_flat)
        u_prime = h.reshape(b, v, l, self.cfg.d_model).permute(0, 2, 1, 3)

        spatial = self.graph_prop(mixed_adj, u_prime)
        fused = self.fusion(u_prime, spatial)

        fused_flat = fused.permute(0, 2, 1, 3).reshape(b * v, l, self.cfg.d_model)
        decoded = self.decoder(fused_flat)
        decoded = decoded.reshape(b, v, l, self.cfg.d_model).permute(0, 2, 1, 3)

        y_hat = self.head(decoded[:, -1])
        y_hat = y_hat.permute(0, 2, 1)
        return y_hat, {"mixed_adj": mixed_adj, "a_dyn": a_dyn}


def prediction_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """eq. 31: mean absolute error."""
    return F.l1_loss(y_hat, y)
