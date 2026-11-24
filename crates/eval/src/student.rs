// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for decisions that stay honest
// when only a handful of units were measured, for its clients. If your team
// needs expertise in small-sample evaluation statistics then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Student's t distribution: the quantile a confidence bound over a few
//! units needs. A mean of `n` units, its standard error estimated from the
//! same units, is t-distributed with `n - 1` degrees of freedom; the normal
//! quantile in its place is the bound of a known variance, and over four
//! units it is optimistic (1.645 against 2.353 at a one-sided level of 0.95).
//!
//! The distribution function is the regularized incomplete beta
//! `F(t) = 1 - I_{df / (df + t^2)}(df / 2, 1 / 2) / 2` for `t >= 0`
//! (Abramowitz and Stegun 26.7.1), computed by Lentz's continued fraction
//! with a Lanczos log-gamma; the quantile inverts it by bisection.
//! Agreement with `scipy.stats.t` is proven by
//! `validation/oracles/statistics/student_t.py`.

/// `ln Γ(x)` for `x > 0` (Lanczos, g = 7, nine terms; relative error near
/// 1e-15).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x) Γ(1 - x) = π / sin(πx).
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut sum = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// The continued fraction of the incomplete beta function (Lentz).
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=300 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < TINY { TINY } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < TINY { TINY } else { c };
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        d = if d.abs() < TINY { TINY } else { d };
        c = 1.0 + aa / c;
        c = if c.abs() < TINY { TINY } else { c };
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// The regularized incomplete beta `I_x(a, b)`.
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    // The fraction converges fast below the mean; use the symmetry above it.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_fraction(b, a, 1.0 - x) / b
    }
}

/// `P(T <= t)` for `T` t-distributed with `df` degrees of freedom; `None`
/// unless `df > 0` and `t` is not NaN.
#[must_use]
pub fn cdf(t: f64, df: f64) -> Option<f64> {
    // `df > 0` read as a range so that NaN is outside it.
    if !(f64::MIN_POSITIVE..=f64::INFINITY).contains(&df) || t.is_nan() {
        return None;
    }
    let tail = 0.5 * incomplete_beta(df / 2.0, 0.5, df / (df + t * t));
    Some(if t >= 0.0 { 1.0 - tail } else { tail })
}

/// The `level` quantile of the t distribution with `df` degrees of freedom:
/// the `t` with `P(T <= t) = level`. `None` unless `0 < level < 1` and
/// `df > 0`.
#[must_use]
pub fn quantile(level: f64, df: f64) -> Option<f64> {
    let open_unit = level > 0.0 && level < 1.0;
    if !open_unit || !(f64::MIN_POSITIVE..=f64::INFINITY).contains(&df) {
        return None;
    }
    if level < 0.5 {
        return quantile(1.0 - level, df).map(|t| -t);
    }
    // Bracket, then bisect to the resolution of f64.
    let mut high = 1.0;
    while cdf(high, df)? < level {
        high *= 2.0;
        if !high.is_finite() {
            return None;
        }
    }
    let mut low = 0.0;
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if mid <= low || mid >= high {
            break;
        }
        if cdf(mid, df)? < level {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(0.5 * (low + high))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-derivable points: t with one degree of freedom is Cauchy
    /// (`F(1) = 3/4`), with two it has the closed form
    /// `F(t) = 1/2 + t / (2 sqrt(2 + t^2))`; and the distribution is
    /// symmetric. The full grid is checked against scipy (`student_t`).
    #[test]
    fn the_distribution_meets_its_closed_forms() {
        assert!((cdf(1.0, 1.0).unwrap() - 0.75).abs() < 1e-14);
        for t in [0.3_f64, 1.0, 2.5, 7.0] {
            let closed = 0.5 + t / (2.0 * (2.0 + t * t).sqrt());
            assert!((cdf(t, 2.0).unwrap() - closed).abs() < 1e-13, "t = {t}");
            assert!((cdf(-t, 5.0).unwrap() + cdf(t, 5.0).unwrap() - 1.0).abs() < 1e-14);
        }
        assert!((quantile(0.75, 1.0).unwrap() - 1.0).abs() < 1e-12);
        assert!(quantile(0.5, 3.0).unwrap().abs() < 1e-12);
        assert_eq!(quantile(1.0, 3.0), None);
        assert_eq!(quantile(0.9, 0.0), None);
    }
}
