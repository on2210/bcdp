# bcdp/subspace/dbcm.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn

from bcdp.trace.site import Site
from bcdp.intervention.interchange_plan import InterchangePlan


# ============================================================
# 1. Mask parameterization (paper-faithful)
# ============================================================

class MaskParam(nn.Module):
    """
    Continuous mask constrained to [0,1].
    """

    def __init__(self, r: int):
        super().__init__()
        self.m = nn.Parameter(torch.ones(r)*0.5)

    def forward(self):
        return self.m

    def clamp_(self):
        with torch.no_grad():
            self.m.data.clamp_(0.0, 1.0)

    def l1_penalty(self):
        return self.m.sum()

    def binary(self):
        return (self.m > 0.5).float()


# ============================================================
# 2. Result container
# ============================================================

@dataclass
class DBCMResult:
    mask: torch.Tensor
    projection: torch.Tensor
    basis: torch.Tensor


# ============================================================
# 3. Trainer
# ============================================================

class DBCM:

    def __init__(
        self,
        *,
        runner,
        site: Site,
        lambda_l1: float = 1e-3,
        lr: float = 1e-2,
        steps: int = 200,
        device: torch.device,
    ):
        self.runner = runner
        self.site = site
        self.lambda_l1 = lambda_l1
        self.lr = lr
        self.steps = steps
        self.device = device

    # --------------------------------------------------------

    def fit_epoch(
        self,
        *,
        V: torch.Tensor,
        loader_o,
        loader_c,
        epochs: int = 1,
        max_batches: int | None = None,
        log_every: int = 50,
    ) -> DBCMResult:
        """
        Paper-faithful:
        - Train continuous m in [0,1] with Adam for one epoch over paired batches
        - Clamp m after each update
        - Eval: binarize by rounding m -> {0,1}
        """
        V = V.to(self.device)
        r = V.shape[1]

        mask_module = MaskParam(r).to(self.device)
        optimizer = torch.optim.Adam(mask_module.parameters(), lr=self.lr)

        step = 0
        for ep in range(epochs):
            for bi, (batch_o, batch_c) in enumerate(zip(loader_o, loader_c)):
                if max_batches is not None and bi >= max_batches:
                    break

                optimizer.zero_grad()

                m = mask_module()
                W = self._build_projection(V, m)
                plan = self._build_plan(batch_c, W)

                result = self.runner.run(batch_o, plan, require_grads=True)
                logits = result.outputs.logits[:, -1, :]

                labels_cf = batch_c["labels"].to(self.device)
                selected = logits.gather(1, labels_cf.unsqueeze(1)).squeeze(1)
                objective = selected.mean()

                loss = -objective + self.lambda_l1 * mask_module.l1_penalty()

                loss.backward()
                optimizer.step()
                mask_module.clamp_()

                if log_every and (step % log_every == 0):
                    with torch.no_grad():
                        m_now = mask_module()
                        near0 = float((m_now < 0.05).float().mean().item())
                        near1 = float((m_now > 0.95).float().mean().item())
                    print(
                        f"[DBCM] step={step:04d} loss={loss.item():.4f} "
                        f"mask_sum={mask_module().sum().item():.2f} near0={near0:.2f} near1={near1:.2f}"
                    )

                step += 1

        with torch.no_grad():
            m_bin = mask_module.binary()              # round -> {0,1}
            final_W = self._build_projection(V, m_bin)
            final_basis = V[:, m_bin.bool()]

        return DBCMResult(mask=m_bin.detach(), projection=final_W.detach(), basis=final_basis.detach())

    def fit(
        self,
        *,
        V: torch.Tensor,
        batch_o: Dict[str, Any],
        batch_c: Dict[str, Any],
    ) -> DBCMResult:

        V = V.to(self.device)
        r = V.shape[1]

        mask_module = MaskParam(r).to(self.device)
        optimizer = torch.optim.Adam(mask_module.parameters(), lr=self.lr)

        for step in range(self.steps):

            optimizer.zero_grad()

            m = mask_module()  # already in [0,1] after clamp
            W = self._build_projection(V, m)

            plan = self._build_plan(batch_c, W)

            result = self.runner.run(
                batch_o,
                plan,
                require_grads=True,
            )

            logits = result.outputs.logits[:, -1, :]

            # ---- Paper-faithful objective ----
            # maximize logit of counterfactual label
            labels_cf = batch_c["labels"].to(self.device)
            selected = logits.gather(1, labels_cf.unsqueeze(1)).squeeze(1)
            objective = selected.mean()

            loss = -objective + self.lambda_l1 * mask_module.l1_penalty()

            loss.backward()
            optimizer.step()

            mask_module.clamp_()

            if step % 20 == 0:
                print(
                    f"[DBCM] step={step:04d} "
                    f"loss={loss.item():.4f} "
                    f"mask_sum={mask_module().sum().item():.2f}"
                )

        final_mask = mask_module.binary()
        final_W = self._build_projection(V, final_mask)
        final_basis = V[:, final_mask.bool()]

        return DBCMResult(
            mask=final_mask.detach(),
            projection=final_W.detach(),
            basis=final_basis.detach(),
        )

    # --------------------------------------------------------

    def _build_projection(self, V, m):
        V_scaled = V * m.unsqueeze(0)   # [d,k]
        W = V_scaled @ V.T              # [d,d]  == V diag(m) V^T
        return W

    def _build_plan(self, batch_c, W):

        def donor_builder(_):
            return batch_c

        return InterchangePlan(
            donor_builder=donor_builder,
            site=self.site,
            W=W,
        )
