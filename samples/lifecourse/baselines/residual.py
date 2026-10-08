# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""The residual network: a spline Cox model plus a small network (paper section 4.7).

The linear predictor is the fitted spline Cox predictor (the base, an offset)
plus the output of a multilayer perceptron on the preprocessed inputs whose
last layer starts at zero, so training begins exactly at the spline model. The
network is trained on the Cox partial likelihood (Breslow handling of ties) by
Adam with weight decay and dropout. The number of epochs is the one that
minimises the partial likelihood on a validation share held out of the training
subjects, among epochs 0 (the spline model itself) to `EPOCHS`; the model is
then refitted on all training subjects for that many epochs. With epoch 0
chosen the predictions are the spline model's.

`resnet-cox-all-w<width>-l<layers>` uses every input; a block name after
`-only-` gives the network that block alone (the base keeps every input):
`exam`, `diet`, `history` or `questionnaire`. `-shuffled` trains on outcomes
permuted among the training subjects, the leakage control: it must not improve
on the spline model.
"""
import numpy as np

import models
from models import SplineCoxNet, Preprocessor, YEARS, breslow, fit_coxnet, select_coxnet

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
EPOCHS = 300
BLOCKS = ("exam", "diet", "history", "questionnaire")


def block_mask(names, block):
    """Which preprocessed columns belong to `block` (`None` is all of them).

    Names are those of `Preprocessor.names`: numeric inputs, `missing:<input>`
    indicators, then `<question>=<answer>` columns."""
    keep = []
    for n in names:
        base = n[len("missing:"):] if n.startswith("missing:") else n
        categorical = "=" in n
        if block is None:
            keep.append(True)
        elif block == "questionnaire":
            keep.append(categorical)
        elif block == "diet":
            keep.append(not categorical and base.startswith("diet:"))
        elif block == "history":
            keep.append(not categorical and base.startswith("hist:"))
        elif block == "exam":
            keep.append(not categorical and not base.startswith(("diet:", "hist:")))
        else:
            raise ValueError(f"unknown block {block!r}")
    return np.array(keep, dtype=bool)


def tie_structure(time):
    """Order by time descending and, per position, the last position holding
    the same time: the risk set of a Breslow event includes its whole tied group."""
    order = np.argsort(-time, kind="stable")
    t = time[order]
    end = np.empty(len(t), dtype=np.int64)
    last = len(t) - 1
    for i in range(len(t) - 1, -1, -1):
        if i == len(t) - 1 or t[i] != t[i + 1]:
            last = i
        end[i] = last
    return order, end


class PartialLikelihood:
    """Negative Cox partial log-likelihood per event, with a fixed ordering."""

    def __init__(self, time, event):
        import torch
        self.order, end = tie_structure(np.asarray(time, dtype=np.float64))
        self.end = torch.as_tensor(end)
        self.event = torch.as_tensor(np.asarray(event, dtype=bool)[self.order]).double()
        self.torch = torch

    def __call__(self, lp):
        lp = lp[self.torch.as_tensor(self.order)]
        log_risk = self.torch.logcumsumexp(lp, 0)[self.end]
        return -((lp - log_risk) * self.event).sum() / self.event.sum().clamp(min=1.0)


def make_net(width, layers, n_in, seed):
    import torch
    torch.manual_seed(seed)
    parts, d = [], n_in
    for _ in range(layers):
        parts += [torch.nn.Linear(d, width), torch.nn.ReLU(), torch.nn.Dropout(DROPOUT)]
        d = width
    last = torch.nn.Linear(d, 1)
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)
    return torch.nn.Sequential(*parts, last).double()


def train(x_fit, base_fit, t_fit, e_fit, width, layers, seed, epochs, x_val=None, base_val=None,
          t_val=None, e_val=None):
    """Train the residual network on the fit rows. Returns (net, validation losses
    by epoch from 0, or None without validation rows)."""
    import torch
    torch.set_num_threads(1)
    net = make_net(width, layers, x_fit.shape[1], seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    xf, bf = torch.as_tensor(x_fit), torch.as_tensor(base_fit)
    loss_fit = PartialLikelihood(t_fit, e_fit)
    losses = None
    if x_val is not None:
        xv, bv = torch.as_tensor(x_val), torch.as_tensor(base_val)
        loss_val = PartialLikelihood(t_val, e_val)
        net.eval()
        with torch.no_grad():
            losses = [float(loss_val(bv + net(xv).squeeze(1)))]
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss_fit(bf + net(xf).squeeze(1)).backward()
        opt.step()
        if losses is not None:
            net.eval()
            with torch.no_grad():
                losses.append(float(loss_val(bv + net(xv).squeeze(1))))
    net.eval()
    return net, losses


def predict_offset(net, x):
    import torch
    with torch.no_grad():
        return net(torch.as_tensor(x)).squeeze(1).numpy()


class ResidualCox(SplineCoxNet):
    """Spline Cox base plus a zero-started network on the preprocessed inputs."""

    def __init__(self, width=64, layers=2, block=None, shuffled=False):
        self.width, self.layers, self.block, self.shuffled = width, layers, block, shuffled
        self.inputs = "all"
        tag = f"-only-{block}" if block else ""
        self.name = f"resnet-cox-all-w{width}-l{layers}{tag}" + ("-shuffled" if shuffled else "")

    def _net_inputs(self, prep, d, rows, mask):
        return prep.transform(d, rows)[:, mask]

    def fit_predict(self, ctx):
        d = ctx.data
        fit, val = ctx.inner()
        t_fit, c_fit = ctx.outcome(fit)
        t_val, c_val = ctx.outcome(val)
        # 1. Base on the inner split: penalty and, with the network, the number of epochs.
        prep = Preprocessor(d, self.inputs).fit(d, fit)
        splines = [self._spline(prep.transform(d, fit)[:, [j]]) for j in range(len(self.num_cols(prep)))]
        m_fit, m_val = self._matrix(prep, splines, d, fit), self._matrix(prep, splines, d, val)
        l1, alphas = select_coxnet(m_fit, t_fit, c_fit >= 0, m_val, t_val, c_val >= 0)
        beta = fit_coxnet(m_fit, t_fit, c_fit >= 0, l1, alphas)
        mask = block_mask(prep.names(d), self.block)
        e_fit = c_fit >= 0
        if self.shuffled:
            rng = np.random.RandomState(ctx.seed)
            perm = rng.permutation(len(fit))
            t_net, e_net = t_fit[perm], e_fit[perm]
        else:
            t_net, e_net = t_fit, e_fit
        _, losses = train(self._net_inputs(prep, d, fit, mask), m_fit @ beta, t_net, e_net,
                          self.width, self.layers, ctx.seed, EPOCHS,
                          self._net_inputs(prep, d, val, mask), m_val @ beta, t_val, c_val >= 0)
        best = int(np.argmin(losses))
        ctx.chosen.update(l1_ratio=l1, alpha=alphas[-1], epochs=best,
                          validation_gain=float(losses[0] - losses[best]))
        # 2. Refit on all training subjects for that many epochs.
        prep = Preprocessor(d, self.inputs).fit(d, ctx.train)
        splines = [self._spline(prep.transform(d, ctx.train)[:, [j]]) for j in range(len(self.num_cols(prep)))]
        m_tr, m_te = self._matrix(prep, splines, d, ctx.train), self._matrix(prep, splines, d, ctx.test)
        t_tr, c_tr = ctx.outcome(ctx.train)
        beta = fit_coxnet(m_tr, t_tr, c_tr >= 0, l1, alphas)
        lp_tr, lp_te = m_tr @ beta, m_te @ beta
        if best > 0:
            mask = block_mask(prep.names(d), self.block)
            if self.shuffled:
                perm = np.random.RandomState(ctx.seed + 1).permutation(len(ctx.train))
                t_net, e_net = t_tr[perm], (c_tr >= 0)[perm]
            else:
                t_net, e_net = t_tr, c_tr >= 0
            net, _ = train(self._net_inputs(prep, d, ctx.train, mask), lp_tr, t_net, e_net,
                           self.width, self.layers, ctx.seed, best)
            lp_tr = lp_tr + predict_offset(net, self._net_inputs(prep, d, ctx.train, mask))
            lp_te = lp_te + predict_offset(net, self._net_inputs(prep, d, ctx.test, mask))
        ctx.chosen["nonzero_coefficients"] = int(np.sum(beta != 0))
        h0 = breslow(lp_tr, t_tr, c_tr >= 0, YEARS.astype(float))
        return 1.0 - np.exp(-np.outer(np.exp(lp_te), h0)), None


WIDTHS_LAYERS = ((16, 1), (64, 1), (256, 1), (64, 2), (64, 3))


def register(registry):
    for w, l in WIDTHS_LAYERS:
        registry[f"resnet-cox-all-w{w}-l{l}"] = lambda w=w, l=l: ResidualCox(w, l)
    for block in BLOCKS:
        registry[f"resnet-cox-all-w64-l2-only-{block}"] = lambda b=block: ResidualCox(64, 2, b)
    registry["resnet-cox-all-w64-l2-shuffled"] = lambda: ResidualCox(64, 2, shuffled=True)


register(models.REGISTRY)
