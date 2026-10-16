# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements the measurement of fine-tuned language
# models for its clients. If your team needs expertise in reporting an
# evaluation's statistics reproducibly, you can procure our services by
# sending an email to info@swedishembedded.com.

"""The LaTeX formatting the paper's table builders share."""

GAP = "@@gap"  # marks a table row that is followed by extra vertical space


def fmt3(x):
    return f"{x:.3f}"


def fmt_p(p):
    """A p-value to three significant digits; very small ones in scientific form."""
    if p < 1e-3:
        mantissa, exponent = f"{p:.1e}".split("e")
        return rf"\ensuremath{{{mantissa}\times10^{{{int(exponent)}}}}}"
    return f"{p:.3g}"


def fmt_points(x):
    """Percentage points with an explicit sign and a typeset minus."""
    text = f"{100 * x:+.1f}"
    if text in ("+0.0", "-0.0"):
        return "0.0"
    return text.replace("-", "$-$")


def thousands(value):
    """An integer with a thin space between groups of three digits."""
    return f"{int(value):,}".replace(",", "\\,")


def table(spec, header, body):
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    out += [line.removesuffix(GAP) + r" \\" + (r" \addlinespace" if line.endswith(GAP) else "")
            for line in body]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def tex_name(name):
    return name.replace("_", r"\_")
