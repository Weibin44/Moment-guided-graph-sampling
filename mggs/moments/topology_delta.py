"""Exact topology-specialized single-edge delta moments for m2 and m3."""

from __future__ import annotations

import numpy as np


def topology_add_deltas(state, candidate_edges):
    """Return exact ``(delta_m2, delta_m3)`` for candidate edge additions."""

    if len(candidate_edges) == 0:
        return np.empty(0), np.empty(0)

    n = state.n
    U, V = candidate_edges[:, 0], candidate_edges[:, 1]
    du, dv = state.deg[U], state.deg[V]
    Wu, Wv = state.W[U], state.W[V]
    Hu, Hv = state.H[U], state.H[V]

    du_d = np.where(du > 0, du * (du + 1), 1.0)
    dv_d = np.where(dv > 0, dv * (dv + 1), 1.0)
    dS2 = (
        1.0 / ((du + 1) * (dv + 1))
        - np.where(du > 0, Wu / du_d, 0.0)
        - np.where(dv > 0, Wv / dv_d, 0.0)
    )
    dm2 = 2.0 / n * dS2

    dil = (
        np.where(du > 0, Hu / du_d, 0.0)
        + np.where(dv > 0, Hv / dv_d, 0.0)
    )
    M_uv = state.M_lookup(U, V)
    gain_s3 = M_uv / ((du + 1) * (dv + 1))
    dm3 = 6.0 / n * (gain_s3 - dil)
    return dm2, dm3

def topology_delete_deltas(state, candidate_edges):
    """Return exact ``(delta_m2, delta_m3)`` for candidate edge deletions."""

    if len(candidate_edges) == 0:
        return np.empty(0), np.empty(0)

    n = state.n
    U, V = candidate_edges[:, 0], candidate_edges[:, 1]
    du, dv = state.deg[U], state.deg[V]
    Wu, Wv = state.W[U], state.W[V]
    Hu, Hv = state.H[U], state.H[V]

    # Exact original formula. The degree-one fallback is safe because the
    # corresponding numerator vanishes after excluding the deleted neighbor.
    du_d_del = np.where(du > 1, du * (du - 1), 1.0)
    dv_d_del = np.where(dv > 1, dv * (dv - 1), 1.0)
    dv_inv = 1.0 / dv
    du_inv = 1.0 / du
    dudv_inv = 1.0 / (du * dv)

    dS2 = ((Wu - dv_inv) / du_d_del + (Wv - du_inv) / dv_d_del - dudv_inv)
    dm2 = 2.0 / n * dS2

    M_uv = state.M_lookup(U, V)
    inter_u = M_uv / dv
    inter_v = M_uv / du
    loss_tri = M_uv / (du * dv)
    dS3 = ((Hu - inter_u) / du_d_del + (Hv - inter_v) / dv_d_del - loss_tri)
    dm3 = 6.0 / n * dS3
    return dm2, dm3

