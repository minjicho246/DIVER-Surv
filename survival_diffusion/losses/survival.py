"""Cox proportional-hazards loss used by DIVER-Surv."""

import torch
import torch.nn as nn


class CoxPHLoss(nn.Module):
    """Compute the negative Cox partial log-likelihood stably."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, risk_pred, event, time):
        risk = risk_pred.reshape(-1)
        event = event.float().reshape(-1)
        time = time.reshape(-1)

        order = torch.argsort(time, descending=True)
        risk = risk[order]
        event = event[order]

        log_partial_likelihood = risk - torch.logcumsumexp(risk, dim=0)
        return -(log_partial_likelihood * event).sum() / (event.sum() + self.eps)
