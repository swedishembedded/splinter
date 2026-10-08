# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements long-horizon risk prediction from cohort and
# survey data for its clients. If your team needs expertise in building,
# validating and proving time-to-event models on real health records, you
# can procure our services by sending an email to info@swedishembedded.com.
"""The conventional baselines, each fitted on training subjects only.

Every baseline implements `fit_predict(ctx)` and returns the all-cause
cumulative incidence at years 1..15 for each test subject (an array of
shape [n_test, 15]) and, for the competing-risk baselines, the cumulative
incidence of each cause (a dict from death code to the same shape).
Hyperparameters are chosen only on a validation share held out of the
training subjects (`Context.inner`); the final model is fitted on the
training subjects as they are, never on test subjects.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp

from features import CODES, Preprocessor

YEARS = np.arange(1, 16)
MONTHS_PER_YEAR = 12
GRID = np.arange(0, 15 * MONTHS_PER_YEAR + 1) / MONTHS_PER_YEAR  # monthly, 0..15 years
INNER_VALIDATION_SHARE = 0.2


@dataclass
class Context:
    """What a baseline is given: the data, the fold and a seed."""

    data: dict
    train: np.ndarray  # row indices
    test: np.ndarray
    seed: int
    horizons: dict  # cycle (source) -> follow-up every survivor reached, years
    chosen: dict = field(default_factory=dict)  # hyperparameters picked, for the manifest

    def inner(self):
        """(fit rows, validation rows): a seeded split of the training rows."""
        rng = np.random.RandomState(self.seed)
        order = rng.permutation(len(self.train))
        n_val = int(round(INNER_VALIDATION_SHARE * len(order)))
        return np.sort(self.train[order[n_val:]]), np.sort(self.train[order[:n_val]])

    def outcome(self, rows):
        return self.data["time"][rows], self.data["cause"][rows]


# --- Kaplan-Meier and Aalen-Johansen -----------------------------------------

def aalen_johansen(time, cause, at):
    """All-cause survival and per-cause cumulative incidence at times `at`."""
    order = np.argsort(time, kind="stable")
    t, c = time[order], cause[order]
    n = len(t)
    uniq, first = np.unique(t, return_index=True)
    at_risk = n - first
    surv_before = 1.0
    cif = np.zeros((len(CODES), len(uniq)))
    surv = np.zeros(len(uniq))
    cum = np.zeros(len(CODES))
    bounds = np.append(first, n)
    for j in range(len(uniq)):
        seg = c[bounds[j]:bounds[j + 1]]
        for k in range(len(CODES)):
            cum[k] += surv_before * np.sum(seg == k) / at_risk[j]
        surv_before *= 1.0 - np.sum(seg >= 0) / at_risk[j]
        cif[:, j] = cum
        surv[j] = surv_before
    idx = np.searchsorted(uniq, at, side="right") - 1
    out_cif = np.where(idx[None, :] >= 0, cif[:, np.maximum(idx, 0)], 0.0)
    out_surv = np.where(idx >= 0, surv[np.maximum(idx, 0)], 1.0)
    return out_surv, out_cif


class KaplanMeier:
    """Population survival: every subject gets the training subjects' curve."""

    name = "km"

    def fit_predict(self, ctx):
        time, cause = ctx.outcome(ctx.train)
        surv, cif = aalen_johansen(time, cause, YEARS.astype(float))
        n = len(ctx.test)
        return (np.tile(1.0 - surv, (n, 1)),
                {code: np.tile(cif[k], (n, 1)) for k, code in enumerate(CODES)})


# --- Cox ---------------------------------------------------------------------

def partial_loglik(lp, time, event):
    """Breslow partial log-likelihood per event (higher is better)."""
    order = np.argsort(-time, kind="stable")
    lp_o, ev_o, t_o = lp[order], event[order].astype(bool), time[order]
    # Risk set of a subject: everyone with time >= its time (ties together).
    log_risk = np.logaddexp.accumulate(lp_o)
    last = np.searchsorted(-t_o, -t_o, side="right") - 1
    return float(np.sum(lp_o[ev_o] - log_risk[last[ev_o]]) / max(ev_o.sum(), 1))


def breslow(lp, time, event, at):
    """Cumulative baseline hazard at times `at`, for the linear predictor `lp`."""
    order = np.argsort(time, kind="stable")
    lp_o, t_o, ev_o = lp[order], time[order], event[order].astype(bool)
    shift = lp_o.max()
    risk = np.exp(lp_o - shift)
    at_risk = np.cumsum(risk[::-1])[::-1]
    uniq, first = np.unique(t_o, return_index=True)
    events = np.add.reduceat(ev_o.astype(float), first)
    h = np.cumsum(events / at_risk[first]) * np.exp(-shift)
    idx = np.searchsorted(uniq, at, side="right") - 1
    return np.where(idx >= 0, h[np.maximum(idx, 0)], 0.0)


def _coxnet(x, time, event, l1_ratio, alphas=None):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    y = np.zeros(len(time), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = event.astype(bool), time
    kw = dict(l1_ratio=l1_ratio, max_iter=100000, tol=1e-7)
    if alphas is None:
        kw.update(n_alphas=40, alpha_min_ratio=0.0001)
    else:
        kw.update(alphas=alphas)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CoxnetSurvivalAnalysis(**kw).fit(x, y)


L1_RATIOS = (0.01, 0.5)


def select_coxnet(x_fit, t_fit, e_fit, x_val, t_val, e_val):
    """(l1_ratio, alphas) by validation partial likelihood: the decreasing
    path of penalties down to the best one, so a refit warm-starts along it
    (a cold start at a small penalty may not converge)."""
    best = None
    for l1 in L1_RATIOS:
        model = _coxnet(x_fit, t_fit, e_fit, l1)
        scores = [partial_loglik(x_val @ model.coef_[:, j], t_val, e_val) for j in range(len(model.alphas_))]
        j = int(np.argmax(scores))
        if best is None or scores[j] > best[0]:
            best = (scores[j], l1, [float(a) for a in model.alphas_[: j + 1]])
    return best[1], best[2]


def fit_coxnet(x, time, event, l1, alphas):
    """Coefficients at the last (smallest) penalty of the path `alphas`."""
    return _coxnet(x, time, event, l1, alphas=alphas).coef_[:, -1]


class CoxNet:
    """Regularised (ridge to elastic net) Cox on all-cause death."""

    def __init__(self, inputs):
        self.inputs, self.name = inputs, f"cox-net-{inputs}"

    def fit_predict(self, ctx):
        d = ctx.data
        fit, val = ctx.inner()
        prep = Preprocessor(d, self.inputs).fit(d, fit)
        t_fit, c_fit = ctx.outcome(fit)
        t_val, c_val = ctx.outcome(val)
        l1, alphas = select_coxnet(prep.transform(d, fit), t_fit, c_fit >= 0,
                                  prep.transform(d, val), t_val, c_val >= 0)
        ctx.chosen.update(l1_ratio=l1, alpha=alphas[-1])
        prep = Preprocessor(d, self.inputs).fit(d, ctx.train)
        x_tr, x_te = prep.transform(d, ctx.train), prep.transform(d, ctx.test)
        t_tr, c_tr = ctx.outcome(ctx.train)
        beta = fit_coxnet(x_tr, t_tr, c_tr >= 0, l1, alphas)
        ctx.chosen["nonzero_coefficients"] = int(np.sum(beta != 0))
        h0 = breslow(x_tr @ beta, t_tr, c_tr >= 0, YEARS.astype(float))
        return 1.0 - np.exp(-np.outer(np.exp(x_te @ beta), h0)), None


class SplineCoxNet:
    """The regularised Cox model with a nonlinear term per continuous input.

    Each continuous column of the shared `Preprocessor` output (median
    imputation, log transform, scaling, clipping) is replaced by a cubic
    B-spline basis, so the model is additive but not restricted to a line in
    each input: the harness can then tell whether a neural model beats a
    strong survival model, not only a linear one. Missing-value indicators
    and categorical indicators pass through unchanged. The knots (quantiles
    of the training columns), the preprocessing and the penalty are all
    fitted on the fold's training subjects only; the constant extrapolation
    keeps a value far outside the training range at the basis value at the
    edge, so it cannot explode."""

    N_KNOTS = 5  # five knots at quantiles of the training column, the end ones included

    def __init__(self, inputs):
        self.inputs, self.name = inputs, f"spline-cox-net-{inputs}"

    @staticmethod
    def num_cols(prep):
        return prep.num_cols

    def _spline(self, x):
        """A `SplineTransformer` for one continuous column, fit on `x` (the
        training values) alone."""
        from sklearn.preprocessing import SplineTransformer
        return SplineTransformer(n_knots=self.N_KNOTS, degree=3, knots="quantile",
                                extrapolation="constant").fit(x)

    def _matrix(self, prep, splines, d, rows):
        """The `Preprocessor` matrix with the continuous columns replaced by
        their spline bases; indicator columns stay in place."""
        x = prep.transform(d, rows)
        n = len(self.num_cols(prep))
        z = np.hstack([spline.transform(x[:, [j]]) for j, spline in enumerate(splines)])
        return np.hstack([z, x[:, n:]])

    def fit_predict(self, ctx):
        d = ctx.data
        fit, val = ctx.inner()
        prep = Preprocessor(d, self.inputs).fit(d, fit)
        splines = [self._spline(prep.transform(d, fit)[:, [j]])
                   for j in range(len(self.num_cols(prep)))]
        t_fit, c_fit = ctx.outcome(fit)
        t_val, c_val = ctx.outcome(val)
        l1, alphas = select_coxnet(self._matrix(prep, splines, d, fit), t_fit, c_fit >= 0,
                                  self._matrix(prep, splines, d, val), t_val, c_val >= 0)
        ctx.chosen.update(l1_ratio=l1, alpha=alphas[-1])
        ctx.chosen["spline_columns"] = int(self._matrix(prep, splines, d, fit).shape[1])
        prep = Preprocessor(d, self.inputs).fit(d, ctx.train)
        splines = [self._spline(prep.transform(d, ctx.train)[:, [j]])
                  for j in range(len(self.num_cols(prep)))]
        x_tr, x_te = self._matrix(prep, splines, d, ctx.train), self._matrix(prep, splines, d, ctx.test)
        t_tr, c_tr = ctx.outcome(ctx.train)
        beta = fit_coxnet(x_tr, t_tr, c_tr >= 0, l1, alphas)
        ctx.chosen["nonzero_coefficients"] = int(np.sum(beta != 0))
        h0 = breslow(x_tr @ beta, t_tr, c_tr >= 0, YEARS.astype(float))
        return 1.0 - np.exp(-np.outer(np.exp(x_te @ beta), h0)), None


class SplineCoxNetWithout(SplineCoxNet):
    """The spline Cox model on every input except the named categorical ones: a
    sensitivity analysis for an input known to be defective (the 1999-2000 cycle lacks
    `told_weak_kidneys`, which it asks under another name)."""

    def __init__(self, drop_categorical=("told_weak_kidneys",)):
        self.drop = tuple(drop_categorical)
        self.inputs, self.name = None, "spline-cox-net-without-" + "-".join(self.drop)

    def fit_predict(self, ctx):
        d = ctx.data
        missing = [c for c in self.drop if c not in [str(x) for x in d["cat_names"]]]
        if missing:
            raise ValueError(f"no such categorical input: {missing}")
        self.inputs = ([str(x) for x in d["num_names"]],
                       [str(x) for x in d["cat_names"] if str(x) not in self.drop])
        self.inputs = (tuple(self.inputs[0]), tuple(self.inputs[1]))
        return super().fit_predict(ctx)


class CauseSpecificCox:
    """One regularised Cox model per cause (the others censor at death),
    combined into cumulative incidence by the Aalen-Johansen formula."""

    def __init__(self, inputs):
        self.inputs, self.name = inputs, f"cs-cox-{inputs}"

    def fit_predict(self, ctx):
        d = ctx.data
        fit, val = ctx.inner()
        prep_inner = Preprocessor(d, self.inputs).fit(d, fit)
        x_fit, x_val = prep_inner.transform(d, fit), prep_inner.transform(d, val)
        t_fit, c_fit = ctx.outcome(fit)
        t_val, c_val = ctx.outcome(val)
        prep = Preprocessor(d, self.inputs).fit(d, ctx.train)
        x_tr, x_te = prep.transform(d, ctx.train), prep.transform(d, ctx.test)
        t_tr, c_tr = ctx.outcome(ctx.train)
        dh = np.zeros((len(ctx.test), len(GRID) - 1, len(CODES)))
        chosen = {}
        for k, code in enumerate(CODES):
            l1, alphas = select_coxnet(x_fit, t_fit, c_fit == k, x_val, t_val, c_val == k)
            chosen[code] = dict(l1_ratio=l1, alpha=alphas[-1])
            beta = fit_coxnet(x_tr, t_tr, c_tr == k, l1, alphas)
            h0 = np.diff(breslow(x_tr @ beta, t_tr, c_tr == k, GRID))
            dh[:, :, k] = np.exp(x_te @ beta)[:, None] * h0[None, :]
        ctx.chosen.update(chosen)
        return cumulative_incidence(dh)


def cumulative_incidence(dh, names=CODES):
    """All-cause and per-cause cumulative incidence at years 1..15 from
    cause-specific hazard increments on the monthly grid [n, months, causes]
    (the causes named `names`), each month's hazards held constant within it."""
    total = dh.sum(axis=2)
    surv_end = np.exp(-np.cumsum(total, axis=1))
    surv_start = np.concatenate([np.ones((dh.shape[0], 1)), surv_end[:, :-1]], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total[:, :, None] > 0, dh / total[:, :, None], 0.0)
    inc = (surv_start * (1.0 - np.exp(-total)))[:, :, None] * share
    cif = np.cumsum(inc, axis=1)
    at = YEARS * MONTHS_PER_YEAR - 1  # increments index m covers month m+1
    per_cause = {name: cif[:, at, k] for k, name in enumerate(names)}
    return per_cause_sum(per_cause), per_cause


def per_cause_sum(per_cause):
    return sum(per_cause.values())


# --- Fixed-horizon logistic with IPCW ----------------------------------------

def censoring_survival(time, died):
    """Kaplan-Meier of the censoring distribution: returns a function G(t-)
    (probability of being uncensored just before t)."""
    order = np.argsort(time, kind="stable")
    t, cens = time[order], ~died[order]
    uniq, first = np.unique(t, return_index=True)
    at_risk = len(t) - first
    events = np.add.reduceat(cens.astype(float), first)
    g = np.cumprod(1.0 - events / at_risk)
    def before(x):
        idx = np.searchsorted(uniq, x, side="left") - 1
        return np.where(idx >= 0, g[np.maximum(idx, 0)], 1.0)
    return before


def ipcw_sample(time, cause, h):
    """Labels, weights and keep-mask for the event 'death by h years'.

    Subjects censored before h carry no label (weight zero); the others are
    weighted by the inverse of the probability of remaining uncensored until
    their death or h. Kaplan-Meier of the censoring distribution from the
    given subjects."""
    died = cause >= 0
    g = censoring_survival(time, died)
    label = died & (time <= h)
    known = label | (time > h)
    when = np.where(label, time, h)
    weight = np.where(known, 1.0 / np.maximum(g(when), 1e-3), 0.0)
    return label.astype(int), weight, known


C_GRID = (0.001, 0.01, 0.1, 1.0)


class LogisticIPCW:
    """Death by each horizon as a logistic model, weighted for censoring.

    A horizon is fitted on the training subjects whose cycle was followed that
    long (the same rule the evaluation applies), then the horizons' risks are
    joined by a monotone interpolation through zero at year 0."""

    def __init__(self, inputs, horizons, name):
        self.inputs, self.horizons, self.name = inputs, tuple(horizons), name

    def _risk(self, ctx, h):
        from sklearn.linear_model import LogisticRegression
        d = ctx.data
        supports = np.array([ctx.horizons[str(s)] >= h for s in d["source"][ctx.train]])
        rows = ctx.train[supports]
        order = np.random.RandomState(ctx.seed + h).permutation(len(rows))
        n_val = int(round(INNER_VALIDATION_SHARE * len(rows)))
        val, fit = np.sort(rows[order[:n_val]]), np.sort(rows[order[n_val:]])
        def sample(r):
            t, c = ctx.outcome(r)
            return ipcw_sample(t, c, h)
        prep = Preprocessor(d, self.inputs).fit(d, fit)
        x_fit, x_val = prep.transform(d, fit), prep.transform(d, val)
        y_f, w_f, k_f = sample(fit)
        y_v, w_v, k_v = sample(val)
        best = None
        for c_reg in C_GRID:
            m = LogisticRegression(C=c_reg, max_iter=3000).fit(x_fit[k_f], y_f[k_f], sample_weight=w_f[k_f] / w_f[k_f].mean())
            p = np.clip(m.predict_proba(x_val[k_v])[:, 1], 1e-9, 1 - 1e-9)
            loss = -np.sum(w_v[k_v] * (y_v[k_v] * np.log(p) + (1 - y_v[k_v]) * np.log(1 - p))) / w_v[k_v].sum()
            if best is None or loss < best[0]:
                best = (loss, c_reg)
        ctx.chosen[f"C_{h}y"] = best[1]
        prep = Preprocessor(d, self.inputs).fit(d, rows)
        y, w, keep = sample(rows)
        m = LogisticRegression(C=best[1], max_iter=3000).fit(prep.transform(d, rows)[keep], y[keep], sample_weight=w[keep] / w[keep].mean())
        ctx.chosen[f"n_{h}y"] = int(keep.sum())
        return m.predict_proba(prep.transform(d, ctx.test))[:, 1]

    def fit_predict(self, ctx):
        risks = np.column_stack([self._risk(ctx, h) for h in self.horizons])
        risks = np.maximum.accumulate(risks, axis=1)  # a cumulative incidence never falls
        xs = np.concatenate([[0.0], np.array(self.horizons, dtype=float)])
        ys = np.hstack([np.zeros((len(risks), 1)), risks])
        cif = PchipInterpolator(xs, ys, axis=1)(YEARS.astype(float))
        return np.clip(cif, 0.0, 1.0), None


# --- Gradient-boosted survival -----------------------------------------------

GBS_TREES = 300


class GradientBoosted:
    """scikit-survival's GradientBoostingSurvivalAnalysis (Cox loss) on all
    inputs. The number of trees is chosen by validation partial likelihood
    along the boosting path; the model is fitted on the training subjects
    outside the validation share."""

    def __init__(self, inputs="all", max_depth=3, learning_rate=0.1, name="gbs-all"):
        self.inputs, self.max_depth, self.learning_rate = inputs, max_depth, learning_rate
        self.name = name

    def fit_predict(self, ctx):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        d = ctx.data
        fit, val = ctx.inner()
        prep = Preprocessor(d, self.inputs).fit(d, fit)
        x_fit, x_val, x_te = prep.transform(d, fit), prep.transform(d, val), prep.transform(d, ctx.test)
        t_fit, c_fit = ctx.outcome(fit)
        t_val, c_val = ctx.outcome(val)
        y = np.zeros(len(fit), dtype=[("event", bool), ("time", float)])
        y["event"], y["time"] = c_fit >= 0, t_fit
        model = GradientBoostingSurvivalAnalysis(
            loss="coxph", n_estimators=GBS_TREES, learning_rate=self.learning_rate, max_depth=self.max_depth,
            subsample=0.5, min_samples_leaf=50, max_features=0.5, random_state=ctx.seed)
        model.fit(x_fit, y)
        scores = [partial_loglik(lp, t_val, c_val >= 0) for lp in model.staged_predict(x_val)]
        best = int(np.argmax(scores))
        ctx.chosen.update(n_trees=best + 1, max_depth=self.max_depth, learning_rate=self.learning_rate, subsample=0.5,
                          min_samples_leaf=50, max_features=0.5)
        lp_fit = next(lp for i, lp in enumerate(model.staged_predict(x_fit)) if i == best)
        lp_te = next(lp for i, lp in enumerate(model.staged_predict(x_te)) if i == best)
        h0 = breslow(lp_fit, t_fit, c_fit >= 0, YEARS.astype(float))
        return 1.0 - np.exp(-np.outer(np.exp(lp_te), h0)), None


REGISTRY = {
    "km": KaplanMeier,
    "cox-net-standard": lambda: CoxNet("standard"),
    "cox-net-all": lambda: CoxNet("all"),
    "cs-cox-agesex": lambda: CauseSpecificCox("agesex"),
    "cs-cox-standard": lambda: CauseSpecificCox("standard"),
    "cs-cox-all": lambda: CauseSpecificCox("all"),
    "logit-ipcw-standard": lambda: LogisticIPCW("standard", (5, 10, 15), "logit-ipcw-standard"),
    "logit-ipcw-all": lambda: LogisticIPCW("all", (5, 10, 15), "logit-ipcw-all"),
    "logit-ipcw-yearly-all": lambda: LogisticIPCW("all", range(1, 16), "logit-ipcw-yearly-all"),
    "gbs-all": GradientBoosted,
    # The same model with a larger step: the first reaches its tree limit,
    # so a faster learner is reported beside it, never chosen on test folds.
    "gbs-fast-all": lambda: GradientBoosted("all", 3, 0.25, "gbs-fast-all"),
    "spline-cox-net-standard": lambda: SplineCoxNet("standard"),
    "spline-cox-net-all": lambda: SplineCoxNet("all"),
    "spline-cox-net-without-told_weak_kidneys": SplineCoxNetWithout,
}


# The residual networks need torch, which the other baselines do not: importing
# the module registers them (it registers itself), and only where torch exists.
try:
    import residual  # noqa: E402,F401
except ModuleNotFoundError as e:
    if e.name != "torch":
        raise
