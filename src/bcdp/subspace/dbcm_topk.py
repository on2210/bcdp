# bcdp/subspace/dbcm_topk.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch

from bcdp.utils.linalg import svd_right_vectors, projection_from_basis


@dataclass
class TopKSubspace:
    basis: torch.Tensor      # [d,k]
    projection: torch.Tensor # [d,d]
    k: int


def dbcm_topk_from_activations(
    X: torch.Tensor,
    k: int,
    *,
    center: bool = True,
) -> TopKSubspace:
    """
    Compute a top-k SVD subspace from activations X.

    Args:
        X: [N,d] activation matrix
        k: number of singular directions to keep
        center: whether to mean-center X before SVD

    Returns:
        TopKSubspace
    """

    V = svd_right_vectors(X, center=center)  # [d,r]
    B = V[:, :k]                             # [d,k]
    W = projection_from_basis(B)              # [d,d]

    return TopKSubspace(
        basis=B,
        projection=W,
        k=k,
    )
