"""One location owns graph edits and their tracked moment updates."""
from __future__ import annotations


def apply_moment_edit(state, operation, u, v, deltas):
    updated = {}
    for name, delta in deltas.items():
        current = getattr(state, name)
        if current is None:
            raise ValueError(f"State moment {name!r} has not been initialized.")
        updated[name] = float(current) + float(delta)
    if operation == "add":
        state.apply_add(int(u), int(v))
    elif operation in ("delete", "del"):
        state.apply_del(int(u), int(v))
    else:
        raise ValueError(f"Unknown edit operation: {operation}")
    for name, value in updated.items():
        setattr(state, name, value)


def run_steps(budget, step, *, observe=None):
    """Run one persistent trajectory without restarting search at checkpoints."""
    for index in range(budget):
        result = step(index)
        if observe is not None:
            observe(index + 1, result)
