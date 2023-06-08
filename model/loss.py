import torch
import torch.nn as nn


def masked_logsumexp(
        x: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1
) -> torch.Tensor:
    """Computes logsumexp over elements of a tensor specified by a mask (two-level)
    in a numerically stable way.

    Parameters
    ----------
    x
        The input tensor.
    mask
        A tensor with the same shape as `x` with 1s in positions that should
        be used for logsumexp computation and 0s everywhere else.
    dim
        The dimension of `x` over which logsumexp is computed. Default -1 uses
        the last dimension.

    Returns
    -------
    torch.Tensor
        Tensor containing the logsumexp of each row of `x` over `dim`.
    """
    max_val, _ = (x * mask).max(dim=dim)
    max_val = torch.clamp_min(max_val, 0)
    return torch.log(
        torch.sum(torch.exp((x - max_val.unsqueeze(dim)) * mask) * mask,
                  dim=dim)) + max_val


class PartialLikelihood(nn.Module):
    """Computes the negative log-likelihood of a batch of model predictions."""

    def __init__(
            self, c1, order='2', reduction="mean"):
        super(PartialLikelihood, self).__init__()
        assert reduction in ["mean", "sum"], "reduction must be one of 'mean', 'sum'"
        self.reduction = reduction
        self.c1 = c1
        self.order = order

    def forward(self, risk_pred, true_times, true_indicator, model):
        eps = 1e-20
        risk_pred = risk_pred.reshape(-1, 1)
        true_times = true_times.reshape(-1, 1)
        true_indicator = true_indicator.reshape(-1, 1)
        mask = torch.ones(true_times.shape[0], true_times.shape[0]).to(true_times.device)
        mask[(true_times.T - true_times) > 0] = 0
        max_risk = risk_pred.max()
        log_loss = torch.exp(risk_pred - max_risk) * mask
        log_loss = torch.sum(log_loss, dim=0)
        log_loss = torch.log(log_loss + eps).reshape(-1, 1) + max_risk
        # Sometimes in the batch we got all censoring data, so the denominator gets 0 and throw nan.
        # Solution: Consider increase the batch size. After all the nll should be performed on the whole dataset.
        # Based on equation 2&3 in https://arxiv.org/pdf/1606.00931.pdf
        nll = -torch.sum((risk_pred - log_loss) * true_indicator) / torch.sum(true_indicator)

        if self.reduction == "mean":
            nll = nll / risk_pred.shape[0]
        elif self.reduction == "sum":
            nll = nll

        for name, w in model.named_parameters():
            if "weight" in name:
                if self.order == '1':
                    # We can also use torch.norm(w, p=self.order) to calculate the L1 and L2 norm
                    nll += self.c1 / 2 * torch.sum(w.abs())
                elif self.order == '2':
                    # Sqrt is not needed because they are essentially the same,
                    # but this way is more convenient to calculate gradients
                    nll += self.c1 / 2 * torch.sum(w.pow(2))
                elif self.order == '21':
                    # ||w||_{2,1} = \sum_{i=1}^{d} \sqrt {\sum_{j=1}^{c} w_{i,j}^2}
                    # We can also use torch.norm(torch.norm(w, dim=1, p=2), p=1) to calculate L21 norm
                    nll += self.c1 / 2 * torch.sum(torch.sum(w.pow(2), dim=1).sqrt())
                else:
                    raise ValueError("Order must be one of '1', '2', '21'")
        return nll


class LikelihoodMTLR(nn.Module):
    """Computes the negative log-likelihood of a batch of model predictions."""

    def __init__(
            self, c1, order='2', reduction="mean"):
        super(LikelihoodMTLR, self).__init__()
        assert reduction in ["mean", "sum"], "reduction must be one of 'mean', 'sum'"
        self.reduction = reduction
        self.c1 = c1
        self.order = order

    def forward(self, logits, target, model):
        censored = target.sum(dim=1) > 1
        nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else 0
        nll_uncensored = (logits[~censored] * target[~censored]).sum() if (~censored).any() else 0

        # the normalising constant
        norm = torch.logsumexp(logits, dim=1).sum()

        nll_total = -(nll_censored + nll_uncensored - norm)
        if self.reduction == "mean":
            nll_total = nll_total / target.size(0)
        elif self.reduction == "sum":
            nll_total = nll_total

        for name, w in model.named_parameters():
            if "weight" in name:
                if self.order == '1':
                    # We can also use torch.norm(w, p=self.order) to calculate the L1 and L2 norm
                    nll_total += self.c1 / 2 * torch.sum(w.abs())
                elif self.order == '2':
                    # Sqrt is not needed because they are essentially the same,
                    # but this way is more convenient to calculate gradients
                    nll_total += self.c1 / 2 * torch.sum(w.pow(2))
                elif self.order == '21':
                    # ||w||_{2,1} = \sum_{i=1}^{d} \sqrt {\sum_{j=1}^{c} w_{i,j}^2}
                    # We can also use torch.norm(torch.norm(w, dim=1, p=2), p=1) to calculate L21 norm
                    nll_total += self.c1 / 2 * torch.sum(torch.sum(w.pow(2), dim=1).sqrt())
                else:
                    raise ValueError("Order must be one of '1', '2', '21'")
        return nll_total

