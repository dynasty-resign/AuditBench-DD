from __future__ import annotations

import re


PROJECT = "TTD-v1"
EXPECTED_GPU = "0"
_ASSIGNMENT = re.compile(
    r"(?:^|[;&|\s])(?:export\s+)?CUDA_VISIBLE_DEVICES\s*=\s*['\"]?([^'\"\s;&|]+)"
)


def bind_remote_command(command: str) -> str:
    """Reject conflicting literal bindings and bind every remote command to GPU0."""
    values = _ASSIGNMENT.findall(command)
    conflicts = [value for value in values if value != EXPECTED_GPU]
    if conflicts:
        raise ValueError(
            f"{PROJECT} is pinned to GPU{EXPECTED_GPU}; "
            f"conflicting CUDA_VISIBLE_DEVICES value(s): {', '.join(conflicts)}"
        )
    return (
        f"export CUDA_VISIBLE_DEVICES={EXPECTED_GPU}; "
        f"export TTD_PROJECT_GPU_INDEX={EXPECTED_GPU}; "
        f"{command}"
    )
