# bcdp/utils/linalg.py

from __future__ import annotations

from typing import Optional

import torch


# -------------------------------------------------
# 1. SVD utilities
# -------------------------------------------------

def svd_right_vectors(
    X: torch.Tensor,
    *,
    center: bool = False,
    k: Optional[int] = None,
) -> torch.Tensor:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D [N,d], got shape {tuple(X.shape)}")

    if center:
        X = X - X.mean(dim=0, keepdim=True)

    # --- SVD on CPU does not support float16 ---
    # Force float32 for numerical stability and compatibility.
    if X.device.type == "cpu" and X.dtype in (torch.float16, torch.bfloat16):
        X = X.float()

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.T  # [d, r]

    if k is not None:
        if k <= 0 or k > V.shape[1]:
            raise ValueError(f"Invalid k={k} for V shape {tuple(V.shape)}")
        V = V[:, :k]

    return V



# -------------------------------------------------
# 2. Orthonormalization
# -------------------------------------------------

def orthonormalize(B: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize columns of B using QR.

    Args:
        B: [d, k]

    Returns:
        Q: [d, k] with orthonormal columns
    """
    if B.ndim != 2:
        raise ValueError(f"B must be 2D [d,k], got shape {tuple(B.shape)}")

    Q, R = torch.linalg.qr(B)
    return Q


# -------------------------------------------------
# 3. Projection from basis
# -------------------------------------------------

def projection_from_basis(B: torch.Tensor) -> torch.Tensor:
    """
    Construct projection matrix W = B B^T.

    Assumes columns of B are orthonormal.

    Args:
        B: [d, k]

    Returns:
        W: [d, d]
    """
    if B.ndim != 2:
        raise ValueError(f"B must be 2D [d,k], got {tuple(B.shape)}")

    return B @ B.T


# -------------------------------------------------
# 4. Projection from V + mask
# -------------------------------------------------

def projection_from_V_and_mask(
    V: torch.Tensor,
    m: torch.Tensor,
) -> torch.Tensor:
    """
    Build projection matrix from singular vectors V and mask m.

    Args:
        V: [d, r]  (orthonormal columns)
        m: [r]     continuous or binary mask

    Returns:
        W: [d, d]
    """
    if V.ndim != 2:
        raise ValueError(f"V must be 2D [d,r], got {tuple(V.shape)}")

    if m.ndim != 1:
        raise ValueError(f"m must be 1D [r], got {tuple(m.shape)}")

    if V.shape[1] != m.shape[0]:
        raise ValueError(
            f"Mask size mismatch: V has r={V.shape[1]}, mask has {m.shape[0]}"
        )

    # Multiply columns by mask
    V_masked = V * m.unsqueeze(0)  # [d,r]

    return V_masked @ V_masked.T
