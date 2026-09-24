"""Candidate selection for moving a graph along an ``(m2, m3)`` direction."""

from __future__ import annotations

import numpy as np

from mggs.moments.topology_delta import topology_add_deltas, topology_delete_deltas


EPS = 1e-12


def relative_moment_delta(m2, m3, original_m2, original_m3) -> np.ndarray:
    return np.asarray([
        (m2 - original_m2) / max(abs(original_m2), EPS),
        (m3 - original_m3) / max(abs(original_m3), EPS),
    ], dtype=np.float64)


def direction_record(step, m2, m3, original_m2, original_m3, direction, n_add, n_del, edge_count):
    rel = relative_moment_delta(m2, m3, original_m2, original_m3)
    projection = float(np.dot(direction, rel))
    orthogonal = float(np.linalg.norm(rel - projection * direction))
    rel_norm = float(np.linalg.norm(rel))
    cosine = float(projection / rel_norm) if rel_norm > EPS else np.nan
    orthogonal_ratio = float(orthogonal / abs(projection)) if abs(projection) > EPS else np.nan
    return {
        "step": int(step), "m2": float(m2), "m3": float(m3),
        "delta_m2": float(m2 - original_m2), "delta_m3": float(m3 - original_m3),
        "rel_delta_m2": float(rel[0]), "rel_delta_m3": float(rel[1]),
        "moment_l2_dist": rel_norm, "projected_displacement": projection,
        "orthogonal_drift": orthogonal, "cosine_to_direction": cosine,
        "orthogonal_ratio": orthogonal_ratio, "n_add": int(n_add),
        "n_del": int(n_del), "edit_count": int(n_add + n_del),
        "edge_count": int(edge_count),
    }


def allowed_add_edge(u, v, add_filter) -> bool:
    if not add_filter:
        return True
    labels = add_filter.get("y")
    if labels is not None:
        same_label = labels[int(u)] == labels[int(v)]
        if add_filter.get("same_label_only", False):
            return bool(same_label)
        if add_filter.get("diff_label_only", False):
            return bool(not same_label)
    return True


def sample_non_edges(state, count, max_rounds=20, add_filter=None) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.int64)
    result = set()
    batch = max(4 * count, 1024)
    for _ in range(max_rounds):
        u_nodes = np.random.randint(state.n, size=batch)
        v_nodes = np.random.randint(state.n, size=batch)
        for u, v in zip(u_nodes.tolist(), v_nodes.tolist()):
            if u == v or v in state.nbrs[u] or not allowed_add_edge(u, v, add_filter):
                continue
            result.add((min(int(u), int(v)), max(int(u), int(v))))
            if len(result) >= count:
                break
        if len(result) >= count:
            break
    return np.asarray(list(result), dtype=np.int64).reshape(-1, 2)


def protected_delete_edge(edge, delete_filter) -> bool:
    if not delete_filter:
        return False
    u, v = int(edge[0]), int(edge[1])
    allowed = delete_filter.get("allowed_edges")
    if allowed is not None and (min(u, v), max(u, v)) not in allowed:
        return True
    labels = delete_filter.get("y")
    if labels is not None:
        same_label = labels[u] == labels[v]
        if delete_filter.get("same_label_only", False) and not same_label:
            return True
        if same_label:
            if delete_filter.get("protect_same_label", False):
                return True
            if delete_filter.get("protect_train_same_label", False):
                train_mask = delete_filter["train_mask"]
                return bool(train_mask[u] and train_mask[v])
    return False


def sample_edges(state, count, delete_filter=None) -> np.ndarray:
    if not state.edge_list or count == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = [edge for edge in state.edge_list if not protected_delete_edge(edge, delete_filter)]
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    if count < 0 or count >= len(edges):
        return np.asarray(edges, dtype=np.int64)
    selected = np.random.choice(len(edges), count, replace=False)
    return np.asarray([edges[index] for index in selected], dtype=np.int64)


def choose_directional_edit(
    state, direction, original_m2, original_m3, add_candidates, delete_candidates,
    add_filter=None, delete_filter=None,
):
    additions = sample_non_edges(state, add_candidates, add_filter=add_filter)
    deletions = sample_edges(state, delete_candidates, delete_filter=delete_filter)
    best = None
    if len(additions):
        dm2, dm3 = topology_add_deltas(state, additions)
        gains = direction[0] * dm2 / max(abs(original_m2), EPS) + direction[1] * dm3 / max(abs(original_m3), EPS)
        index = int(np.argmax(gains))
        best = {"op": "add", "u": int(additions[index, 0]), "v": int(additions[index, 1]),
                "dm2": float(dm2[index]), "dm3": float(dm3[index]), "gain": float(gains[index])}
    if len(deletions):
        dm2, dm3 = topology_delete_deltas(state, deletions)
        gains = direction[0] * dm2 / max(abs(original_m2), EPS) + direction[1] * dm3 / max(abs(original_m3), EPS)
        index = int(np.argmax(gains))
        candidate = {"op": "del", "u": int(deletions[index, 0]), "v": int(deletions[index, 1]),
                     "dm2": float(dm2[index]), "dm3": float(dm3[index]), "gain": float(gains[index])}
        if best is None or candidate["gain"] > best["gain"]:
            best = candidate
    return best


def choose_ray_constrained_edit(
    state, direction, original_m2, original_m3, add_candidates, delete_candidates,
    min_gain, min_cosine, orthogonal_penalty, add_filter=None, delete_filter=None,
):
    additions = sample_non_edges(state, add_candidates, add_filter=add_filter)
    deletions = sample_edges(state, delete_candidates, delete_filter=delete_filter)
    best = None
    current_rel = relative_moment_delta(state.m2, state.m3, original_m2, original_m3)
    current_projection = float(np.dot(direction, current_rel))

    def consider(candidates, dm2, dm3, operation):
        nonlocal best
        if not len(candidates):
            return
        rel_m2 = (state.m2 + dm2 - original_m2) / max(abs(original_m2), EPS)
        rel_m3 = (state.m3 + dm3 - original_m3) / max(abs(original_m3), EPS)
        projection = direction[0] * rel_m2 + direction[1] * rel_m3
        norm = np.sqrt(rel_m2 ** 2 + rel_m3 ** 2)
        cosine = np.full_like(projection, -np.inf, dtype=float)
        valid_norm = norm > EPS
        cosine[valid_norm] = projection[valid_norm] / norm[valid_norm]
        orthogonal = np.sqrt(np.maximum(norm ** 2 - projection ** 2, 0.0))
        projected_gain = projection - current_projection
        valid = (projected_gain > min_gain) & (cosine >= min_cosine)
        if not np.any(valid):
            return
        score = np.where(valid, projected_gain - orthogonal_penalty * orthogonal, -np.inf)
        index = int(np.argmax(score))
        candidate = {
            "op": operation, "u": int(candidates[index, 0]), "v": int(candidates[index, 1]),
            "dm2": float(dm2[index]), "dm3": float(dm3[index]),
            "gain": float(projected_gain[index]), "score": float(score[index]),
            "projected_after": float(projection[index]),
            "orthogonal_after": float(orthogonal[index]), "cosine_after": float(cosine[index]),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if len(additions):
        consider(additions, *topology_add_deltas(state, additions), "add")
    if len(deletions):
        consider(deletions, *topology_delete_deltas(state, deletions), "del")
    return best


def apply_edit(state, edit) -> None:
    if edit["op"] == "add":
        state.apply_add(edit["u"], edit["v"])
    elif edit["op"] == "del":
        state.apply_del(edit["u"], edit["v"])
    else:
        raise ValueError(f"Unknown operation: {edit['op']}")
