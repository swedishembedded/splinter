# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible evidence and reporting pipelines
# for health-data prediction for its clients. If your team needs expertise in
# presenting validated survival-model results, you can procure our services by
# sending an email to info@swedishembedded.com.
"""Figures of the paper, drawn from the numbers of its result tables.

    python figures.py <out dir> [<drivers-all.json>]

Needs matplotlib (not part of the health environment: use a separate venv).
Every figure is checked for overlapping text and text outside the image, and the
script exits with an error naming each offender, so none can ship unnoticed.
The drivers-all.json file (`baselines/drivers.py`) adds the two variable figures.
"""
# Numbers are copied from the result tables under resources/health/m6 and from paper.md.
import sys
import matplotlib
import matplotlib.text
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1]
INK, MUTE, GRID = "#1b2a3a", "#6b7886", "#e3e8ee"
NAVY, TEAL, AMBER, RED, SLATE = "#17324d", "#0f9d8a", "#e0a030", "#c8503c", "#9aa7b5"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": SLATE, "axes.labelcolor": INK, "xtick.color": MUTE,
    "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})


def title(fig, head, sub):
    fig.text(0.04, 0.965, head, fontsize=16, fontweight="bold", va="top", color=INK)
    fig.text(0.04, 0.915, sub, fontsize=10.5, va="top", color=MUTE)


def foot(fig, text):
    fig.text(0.04, 0.02, text, fontsize=8.3, color=MUTE, va="bottom")


PROBLEMS = []


def check_text(fig, name):
    """Record every pair of overlapping text boxes and every text box outside the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figbox = fig.bbox
    boxes = []
    unseen = set()  # tick labels beyond the axis limits are created but never drawn
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    unseen.update((id(tick.label1), id(tick.label2)))
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) not in unseen and t.get_visible() and t.get_text().strip():
            b = t.get_window_extent(renderer)
            if b.width > 0:
                boxes.append((t.get_text().replace("\n", " / ")[:40], b))
    for label, b in boxes:
        if b.x0 < figbox.x0 - 1 or b.x1 > figbox.x1 + 1 or b.y0 < figbox.y0 - 1 or b.y1 > figbox.y1 + 1:
            PROBLEMS.append(f"{name}: outside the image: {label!r}")
    for i, (la, a) in enumerate(boxes):
        for lb, b in boxes[i + 1:]:
            if a.x0 < b.x1 - 1 and b.x0 < a.x1 - 1 and a.y0 < b.y1 - 1 and b.y0 < a.y1 - 1:
                PROBLEMS.append(f"{name}: overlap: {la!r} and {lb!r}")


def save(fig, name):
    check_text(fig, name)
    fig.savefig(f"{OUT}/{name}.png")
    plt.close(fig)


# 1. Overview schematic ------------------------------------------------------
def fig_overview():
    fig = plt.figure(figsize=(12, 6.2))
    title(fig, "One examination, many questions",
          "NHANES 1999-2018 linked to mortality: 56,253 adults, 8,355 deaths, 5x5 cross-validation, rules fixed before results")
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.78]); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)

    def box(x, y, w, h, head, body, color, fc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=color, lw=1.6))
        ax.text(x + 0.15, y + h - 0.2, head, fontsize=10.5, fontweight="bold", color=color, va="top")
        ax.text(x + 0.15, y + h - 0.62, body, fontsize=8.8, color=INK, va="top", linespacing=1.45)

    box(0.1, 2.0, 2.5, 2.2, "Inputs", "age, sex, race\nbody size, blood pressure\nlab panel (blood, urine)\nlifestyle and history\n8 accelerometer summaries", NAVY)
    box(3.5, 2.0, 2.7, 2.2, "Models", "elastic-net Cox\nspline Cox  (best)\nlogistic, piecewise-exp.\ndeep set encoder\nresidual net, series CNN", TEAL)
    box(7.1, 4.0, 4.8, 1.75, "Survival", "all-cause death, 0-15 y\nIBS 0.0460  |  C(10y) 0.901", NAVY)
    box(7.1, 2.1, 4.8, 1.75, "Cause and time", "CVD, cancer, other deaths  |  expected years lived\ndiabetes or hypertension on the certificate", NAVY)
    box(7.1, 0.2, 4.8, 1.75, "Present state", "8 prevalent conditions (AUROC 0.79-0.92)\nundiagnosed disease screens (weak)", NAVY)
    for y in (4.87, 2.97, 1.07):
        ax.annotate("", xy=(7.05, y), xytext=(6.25, 3.1), arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.3))
    ax.annotate("", xy=(3.45, 3.1), xytext=(2.65, 3.1), arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))
    box(0.1, 0.2, 6.1, 1.45, "Method guards", "pre-registered decision rules | corrected resampled t-test\ncluster bootstrap (cycle x stratum x PSU) | hash-chained run log", AMBER)
    save(fig, "fig1-overview")


# 2. Model ladder ------------------------------------------------------------
def fig_ladder():
    rows = [("age and sex only", 0.05311, SLATE), ("conventional risk factors", 0.04952, SLATE),
            ("deep set encoder (all inputs)", 0.04711, AMBER), ("additive piecewise-exponential", 0.04649, SLATE),
            ("elastic-net Cox (all inputs)", 0.04613, TEAL), ("logistic, inverse-weighted", 0.04605, TEAL),
            ("spline Cox (all inputs)", 0.04600, NAVY)]
    fig = plt.figure(figsize=(11, 5.6)); title(fig, "Using all examination inputs beats the conventional score",
        "Integrated Brier score over 0-15 years, lower is better; 25 held-out folds")
    ax = fig.add_axes([0.28, 0.14, 0.68, 0.68])
    ys = range(len(rows))
    for y, (n, v, c) in zip(ys, rows):
        ax.barh(y, v, color=c, height=0.62)
        ax.text(v + 0.0004, y, f"{v:.4f}", va="center", fontsize=10, fontweight="bold" if c == NAVY else None)
    ax.set_yticks(list(ys)); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0.040, 0.056)
    ax.invert_yaxis(); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlabel("integrated Brier score (axis starts at 0.040)")
    ax.annotate("", xy=(0.0460, 7.0), xytext=(0.04952, 7.0), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax.text(0.0478, 7.3, "-7% vs conventional", ha="center", color=RED, fontsize=9.5, fontweight="bold")
    ax.set_ylim(7.6, -0.6)
    foot(fig, "Spline vs elastic-net: -0.00013 [-0.00057, +0.00031], not resolved. The deep encoder is +0.0011 behind the spline Cox model.")
    save(fig, "fig2-model-ladder")


# 3. Learning curves ---------------------------------------------------------
NUDGE = {"elastic-net Cox": 0.00005, "logistic": -0.00030, "spline Cox": -0.00060}


def fig_curves():
    shares = [10, 25, 50, 75, 100]
    d = {"elastic-net Cox": ([0.04785, 0.04702, 0.04633, 0.04618, 0.04613], TEAL),
         "spline Cox": ([0.04818, 0.04716, 0.04649, 0.04604, 0.04600], NAVY),
         "logistic": ([0.04891, 0.04755, 0.04636, 0.04620, 0.04605], SLATE),
         "additive piecewise-exp.": ([0.05241, 0.04948, 0.04748, 0.04685, 0.04645], AMBER),
         "deep set encoder": ([0.05325, 0.04986, 0.04780, 0.04727, 0.04696], RED)}
    fig = plt.figure(figsize=(10, 5.8)); title(fig, "The classical models are flat; the flexible ones still improve",
        "Mean integrated Brier score vs share of training subjects, 25 folds, identical test folds")
    ax = fig.add_axes([0.09, 0.13, 0.72, 0.68])
    for n, (v, c) in d.items():
        ax.plot(shares, v, "-o", color=c, lw=2.2, ms=5)
        ax.text(102, v[-1] + NUDGE.get(n, 0), n, color=c, va="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0.0450, 0.0535); ax.set_xlim(5, 135); ax.set_xticks(shares); ax.set_xlabel("share of training subjects (%)")
    ax.set_ylabel("integrated Brier score"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.text(45, 0.0528, "fitted asymptotes (power law):\nadditive PE 0.0423, deep encoder 0.0444,\nCox models' intervals still reach 0.028-0.046", fontsize=8.8, color=MUTE, va="top")
    foot(fig, "Extrapolation from five points is a hypothesis for a larger data set, not an estimate of what more subjects would give.")
    save(fig, "fig3-learning-curves")


# 4. Forest of pre-registered tests -----------------------------------------
def fig_forest():
    rows = [("Spline vs linear Cox, conventional inputs", -0.00062, -0.00106, -0.00019, TEAL, "resolved gain"),
            ("Spline vs linear Cox, all inputs", -0.00013, -0.00057, 0.00031, SLATE, "not resolved"),
            ("Accelerometer summaries added (Brier 10 y)", -0.00089, -0.00167, -0.00011, TEAL, "narrow gain"),
            ("Residual network on spline Cox", -0.00008, -0.00030, 0.00015, SLATE, "not useful"),
            ("Deep encoder, 3-seed average vs spline Cox", 0.00018, -0.00081, 0.00118, SLATE, "not resolved"),
            ("Deep encoder, one seed vs spline Cox", 0.00111, 0.00019, 0.00204, RED, "resolved loss")]
    fig = plt.figure(figsize=(11, 5.6)); title(fig, "What the pre-registered tests resolved, and what they did not",
        "Difference in Brier score (negative = better). 95% intervals; dashed line = the 0.001 gain the rule required")
    ax = fig.add_axes([0.37, 0.14, 0.58, 0.68])
    for y, (n, m, lo, hi, c, tag) in enumerate(rows):
        ax.plot([lo, hi], [y, y], color=c, lw=3, solid_capstyle="round")
        ax.plot(m, y, "o", color=c, ms=8, mec="white", mew=1.4)
        ax.text(0.00235, y, tag, va="center", fontsize=9, color=c, fontweight="bold")
    ax.axvline(0, color=INK, lw=1); ax.axvline(-0.001, color=AMBER, lw=1.4, ls="--")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9.8)
    ax.set_ylim(5.7, -0.5); ax.set_xlim(-0.0022, 0.0034); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlabel("difference in Brier score")
    ax.text(-0.00095, 5.45, "required gain", color=AMBER, fontsize=8.8, ha="left")
    foot(fig, "No flexible model met the pre-registered bar for beating the spline Cox model. The honest result is a ceiling for one examination.")
    save(fig, "fig4-forest")


# 5. Cause-specific AUC ------------------------------------------------------
def fig_causes():
    causes = ["cardiovascular", "cancer", "other causes"]
    a = [0.9059, 0.8620, 0.8570]; s = [0.9258, 0.8796, 0.8794]; f = [0.9379, 0.8938, 0.9112]
    d = [+0.01213, +0.01421, +0.03182]
    fig = plt.figure(figsize=(10.5, 5.6)); title(fig, "The full input set predicts every cause of death better",
        "Time-dependent AUC at 10 years, cause-specific Cox models, 25 folds")
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.68]); w = 0.25
    for i, (lab, v, c) in enumerate([("age and sex", a, SLATE), ("conventional factors", s, AMBER), ("all inputs", f, NAVY)]):
        xs = [k + (i - 1) * w for k in range(3)]
        ax.bar(xs, v, w * 0.92, color=c, label=lab)
        for x, y in zip(xs, v): ax.text(x, y + 0.002, f"{y:.3f}", ha="center", fontsize=8.6)
    for k, dv in enumerate(d):
        ax.text(k + w, 0.9625 if k == 0 else 0.9625, f"+{dv:.3f}", ha="center", color=TEAL, fontsize=10, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(causes, fontsize=11); ax.set_ylim(0.80, 0.975)
    ax.set_ylabel("AUC (axis starts at 0.80)"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncols=3, fontsize=9.5, bbox_to_anchor=(1, 1.04))
    foot(fig, "Green: AUC gain of all inputs over conventional factors; every interval excludes zero.\nDiabetes on the certificate adds +0.022 to +0.025 AUC beyond the all-cause ranker.")
    save(fig, "fig5-causes")


# 6. Calibration of expected years lived ------------------------------------
def fig_rmst():
    pred = [6.862, 9.158, 9.634, 9.793, 9.867, 9.911, 9.938, 9.956, 9.969, 9.981]
    obs = [6.866, 9.155, 9.678, 9.877, 9.914, 9.938, 9.936, 9.948, 9.983, 9.997]
    fig = plt.figure(figsize=(10, 5.6)); title(fig, "Expected years lived are calibrated",
        "Predicted vs observed restricted mean survival over 10 years, ten equal-weight risk groups")
    ax = fig.add_axes([0.08, 0.13, 0.50, 0.70])
    ax.plot([6.6, 10.1], [6.6, 10.1], color=SLATE, lw=1.2, ls="--")
    ax.scatter(pred, obs, s=70, color=NAVY, zorder=3, edgecolor="white")
    ax.set_xlabel("predicted years lived in 10"); ax.set_ylabel("observed years lived in 10")
    ax.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_xlim(6.6, 10.1); ax.set_ylim(6.6, 10.1)
    ax.text(6.75, 9.9, "highest-risk 10%:\npredicted 6.86, observed 6.87", fontsize=9, color=INK, va="top")
    ax2 = fig.add_axes([0.66, 0.13, 0.30, 0.70])
    for i, (n, g, c) in enumerate([("age, sex", 0.045, SLATE), ("conventional", 0.036, AMBER), ("all inputs, Cox", 0.035, TEAL), ("spline Cox", 0.025, NAVY)]):
        ax2.barh(i, g, color=c, height=0.6); ax2.text(g + 0.001, i, f"{g:.3f}", va="center", fontsize=9.5)
    ax2.set_yticks(range(4)); ax2.set_yticklabels(["age, sex", "conventional", "all inputs, Cox", "spline Cox"])
    ax2.invert_yaxis(); ax2.set_xlim(0, 0.06); ax2.set_xlabel("mean absolute gap (years)")
    ax2.set_title("gap between predicted and observed", fontsize=9.5, color=MUTE, loc="left")
    foot(fig, "Pre-registered calibration rule: gap at most 0.25 years. All four models pass; the spline Cox model has the smallest gap, 0.025 years.")
    save(fig, "fig6-life-expectancy")


# 7. Prevalent conditions ----------------------------------------------------
def fig_prevalent():
    data = [("depression", .5631, .7332, .8191), ("sleep problem", .6034, .6715, .7869),
            ("anaemia", .6605, .7907, .9194), ("diabetes", .7415, .8431, .8945),
            ("diabetes (fasting)", .745, .8339, .8793), ("kidney markers", .7485, .787, .8447),
            ("high cholesterol", .7309, .7506, .79), ("hypertension", .7823, .8195, .84),
            ("osteoporosis", .8479, .8521, .8699)]
    data.sort(key=lambda r: r[3] - r[1], reverse=True)
    fig = plt.figure(figsize=(10.5, 6.2)); title(fig, "What the examination reveals about the present state",
        "AUROC for conditions already present: age and sex, without the inputs that define the label, and with all inputs")
    ax = fig.add_axes([0.20, 0.13, 0.75, 0.68])
    for y, (n, a, nd, fu) in enumerate(data):
        ax.plot([a, fu], [y, y], color=GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter(a, y, s=55, color=SLATE, zorder=3); ax.scatter(nd, y, s=70, color=TEAL, zorder=3)
        ax.scatter(fu, y, s=70, color=NAVY, zorder=3)
        ax.text(fu + 0.006, y, f"{fu:.2f}", va="center", fontsize=9, color=NAVY)
    ax.set_yticks(range(len(data))); ax.set_yticklabels([d[0] for d in data], fontsize=10.5)
    ax.invert_yaxis(); ax.set_xlim(0.52, 0.96); ax.set_xlabel("AUROC"); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for lab, c, x in [("age and sex", SLATE, 0.54), ("without defining inputs", TEAL, 0.64), ("all inputs", NAVY, 0.79)]:
        ax.scatter(x, -1.05, s=55, color=c, clip_on=False); ax.text(x + 0.008, -1.05, lab, va="center", fontsize=9.3, clip_on=False)
    foot(fig, "Gradient-boosted classifier, 25 folds. 'All inputs' includes correlates measured at the same visit, so it reads as detection, not prediction.")
    save(fig, "fig7-prevalent-conditions")


# 8. Screening reality check -------------------------------------------------
def fig_screen():
    data = [("undiagnosed\nhigh cholesterol", .124, .231), ("undiagnosed\nkidney markers", .430, .452),
            ("undiagnosed\nhypertension", .391, .427), ("undiagnosed\ndiabetes", .371, .443)]
    fig = plt.figure(figsize=(10.5, 5.6)); title(fig, "Screening for undiagnosed disease is still weak",
        "Sensitivity at 90% specificity: age and BMI alone vs the best non-defining model")
    ax = fig.add_axes([0.08, 0.14, 0.60, 0.68]); w = 0.34
    for k, (n, b, m) in enumerate(data):
        ax.bar(k - w / 2, b, w * 0.92, color=SLATE); ax.bar(k + w / 2, m, w * 0.92, color=TEAL)
        ax.text(k - w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(k + w / 2, m + 0.01, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([d[0] for d in data], fontsize=9.3); ax.set_ylim(0, 0.62)
    ax.set_ylabel("sensitivity"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.bar(0, 0, color=SLATE, label="age + BMI"); ax.bar(0, 0, color=TEAL, label="model, no defining inputs")
    ax.legend(frameon=False, loc="upper left", fontsize=9.2)
    ax2 = fig.add_axes([0.74, 0.14, 0.22, 0.68]); ax2.axis("off")
    ax2.text(0, 0.95, "Positive predictive value", fontsize=10, fontweight="bold", va="top")
    ax2.text(0, 0.80, "undiagnosed diabetes\nat 2% prevalence", fontsize=9.3, color=MUTE, va="top")
    ax2.text(0, 0.55, "9%", fontsize=34, fontweight="bold", color=RED, va="top")
    ax2.text(0, 0.28, "about nine of ten\npositive calls are\nfalse alarms", fontsize=9.3, va="top")
    foot(fig, "The model beats age+BMI at matched specificity for all four (a post hoc reading); the pre-registered rule credits only cholesterol.")
    save(fig, "fig8-screening")


# 9. Agent audit -------------------------------------------------------------
def fig_agent():
    fig = plt.figure(figsize=(11, 5.4)); title(fig, "Auditing the research agent",
        "A local 27B coding agent worked under a supervising model; every run sits in a hash-chained log")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.72]); ax.axis("off"); ax.set_xlim(-0.2, 11); ax.set_ylim(0, 6)
    ax.text(0, 5.7, "Substantive tasks", fontsize=11, fontweight="bold", va="top")
    for i in range(4):
        ok = i < 2
        ax.add_patch(FancyBboxPatch((i * 1.2, 4.0), 1.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=TEAL if ok else "white", ec=TEAL if ok else RED, lw=2))
        ax.text(i * 1.2 + 0.5, 4.5, "ok" if ok else "no", ha="center", va="center", color="white" if ok else RED, fontsize=12, fontweight="bold")
    ax.text(0, 3.65, "2 accepted, 2 not accepted", fontsize=9.5, color=MUTE, va="top")
    ax.text(0, 2.75, "Held-out adapter test (rule fixed beforehand)", fontsize=11, fontweight="bold", va="top")
    for name, n, y, c in [("candidate adapter", 0, 2.15, RED), ("model in use", 1, 1.5, NAVY)]:
        ax.text(0, y, name, fontsize=9.5, va="center")
        for j in range(8):
            ax.add_patch(FancyBboxPatch((2.2 + j * 0.45, y - 0.16), 0.36, 0.32, boxstyle="round,pad=0.01,rounding_size=0.05",
                                        fc=c if j < n else "white", ec=c, lw=1.3))
        ax.text(2.2 + 8 * 0.45 + 0.05, y, f"{n} of 8", fontsize=9.5, va="center", color=c, fontweight="bold")
    ax.text(0, 0.85, "Adapter rejected; model in use unchanged.", fontsize=9.5, color=MUTE, va="top")
    ax.text(7.3, 5.7, "Infrastructure defects repaired first", fontsize=11, fontweight="bold", va="top")
    items = ["reasoning block spending the output budget", "greedy decoding repeating one action",
             "server context overflow", "unreadable old log format"]
    for i, t in enumerate(items):
        ax.scatter(7.4, 4.95 - i * 0.62, s=70, color=AMBER); ax.text(7.6, 4.95 - i * 0.62, t, va="center", fontsize=9.8)
    ax.text(7.3, 2.35, "Two defects in the supervisor's own acceptance\nchecks surfaced later. Each repair has a\nregression specification.", fontsize=9.5, color=MUTE, va="top")
    save(fig, "fig9-agent-audit")


# 10-11. Variable importance and direction ----------------------------------
def load_drivers():
    import json
    with open(sys.argv[2]) as f:
        return json.load(f)


def fig_blocks():
    d = load_drivers()["blocks"]
    age = next(b for b in d if b["name"] == "age")
    rest = [b for b in d if b["name"] != "age"]
    fig = plt.figure(figsize=(11, 6.4)); title(fig, "Which groups of measurements the model relies on",
        "Fall in held-out partial log-likelihood when a group is shuffled among test subjects, 5 folds, bars show mean and SE")
    ax = fig.add_axes([0.27, 0.14, 0.62, 0.68])
    for y, b in enumerate(rest):
        ax.barh(y, b["drop"], color=NAVY if b["drop"] >= 0.02 else SLATE, height=0.62)
        ax.errorbar(b["drop"], y, xerr=b["se"], color=INK, capsize=3, lw=1.2)
        ax.text(b["drop"] + b["se"] + 0.002, y, f"{b['drop']:.3f}", va="center", fontsize=9.5)
    ax.set_yticks(range(len(rest))); ax.set_yticklabels([b["name"] for b in rest]); ax.invert_yaxis()
    ax.set_xlim(0, 0.085); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlabel("fall in partial log-likelihood per event (nats)")
    fig.text(0.60, 0.30, f"Age alone: {age['drop']:.2f}\n(off scale, 20 times the\nnext group)", fontsize=10.5,
             color=RED, fontweight="bold", va="top")
    foot(fig, "Post hoc grouping of related inputs. Correlated inputs share credit when shuffled one at a time, so groups are the fairer reading.")
    save(fig, "fig10-variable-groups")


def fig_direction():
    skip = {"race_ethnicity", "education", "marital"}
    out = []
    for r in load_drivers()["rows"][:36]:
        if not r["informative"] or r["name"] in skip:
            continue
        for k, v in r["ratios"].items():
            out.append((f"{r['name']}  ({k})", v["hr"], v["low"], v["high"]))
    out.sort(key=lambda r: r[1])
    fig = plt.figure(figsize=(11, 0.34 * len(out) + 2.4)); title(fig, "Direction and size of the association with death",
        "Hazard ratio between the 90th and 10th percentile (or level vs reference); range over 5 folds; log scale")
    h = 0.34 * len(out) + 2.4
    ax = fig.add_axes([0.34, 1.0 / h * 1.0, 0.60, 1 - 2.6 / h])
    for y, (n, hr, lo, hi) in enumerate(out):
        c = RED if hr > 1.0 else TEAL
        ax.plot([lo, hi], [y, y], color=c, lw=2.4, alpha=0.55); ax.plot(hr, y, "o", color=c, ms=7)
    ax.axvline(1, color=INK, lw=1); ax.set_xscale("log"); ax.set_xlim(0.5, 60)
    ax.set_xticks([0.5, 1, 2, 5, 10, 30]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "30"])
    ax.set_yticks(range(len(out))); ax.set_yticklabels([o[0] for o in out], fontsize=9.3)
    ax.set_ylim(-0.7, len(out) - 0.3); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlabel("hazard ratio (red: higher hazard, teal: lower hazard)")
    foot(fig, "Associations given all other inputs, not effects of changing them. Low ALT, and low weight at fixed waist and BMI,\nprobably mark frailty and body composition, so high values here are not advice.")
    save(fig, "fig11-direction")


for f in (fig_overview, fig_ladder, fig_curves, fig_forest, fig_causes, fig_rmst, fig_prevalent, fig_screen, fig_agent) + ((fig_blocks, fig_direction) if len(sys.argv) > 2 else ()):
    f()
if PROBLEMS:
    print("\n".join(PROBLEMS))
    sys.exit(1)
print("ok: no overlapping or overflowing text")
