#!/usr/bin/env python3
"""Generate DuckDB sqllogictest files for all stochastic distributions.

Uses scipy.stats to compute reference values. Run from project root:
    python3 scripts/generate_tests.py
"""

import math
import os
import numpy as np
from scipy import stats

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test", "sql")


def fmt(v, decimals=6):
    """Format a float for test output, rounding to `decimals` places."""
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    r = round(float(v), decimals)
    # Remove trailing zeros but keep at least one decimal
    s = f"{r:.{decimals}f}"
    return s


def fmt_range(lo, hi):
    """Format a range/support pair as DuckDB ARRAY text."""
    lo_s = "-inf" if lo == float("-inf") else fmt(lo)
    hi_s = "inf" if hi == float("inf") else fmt(hi)
    return f"[{lo_s}, {hi_s}]"


class Distribution:
    """Base class for defining a distribution's test configuration."""

    def __init__(
        self,
        name,
        scipy_dist,
        params,
        param_sql,
        test_x,
        *,
        is_discrete=False,
        has_sample=True,
        sample_type="R",  # R=DOUBLE, I=BIGINT, B=BOOLEAN
        disabled_props=None,
        range_val=None,
        support_val=None,
        mode_fn=None,
        quantile_is_int=False,
        skip_quantile=False,
        x_fmt=None,  # "bigint" to force ::BIGINT cast on x values
    ):
        self.name = name
        self.scipy_dist = scipy_dist
        self.params = params  # dict of scipy params
        self.param_sql = param_sql  # SQL params string e.g. "0.0, 1.0"
        self.test_x = test_x  # list of x values to test pdf/cdf at
        self.is_discrete = is_discrete
        self.has_sample = has_sample
        self.sample_type = sample_type
        self.disabled_props = disabled_props or set()
        self.range_val = range_val  # (lo, hi) tuple, None = use support
        self.support_val = support_val  # (lo, hi) tuple, None = compute
        self.mode_fn = mode_fn  # callable returning mode value
        # Whether the x/call parameter in SQL needs to be an integer (for BIGINT params)
        self.x_is_int = is_discrete
        # Whether quantile/quantile_complement return BIGINT (skip round())
        self.quantile_is_int = quantile_is_int
        self.skip_quantile = skip_quantile
        self.x_fmt = x_fmt

    def dist(self):
        return self.scipy_dist(**self.params)

    def pdf(self, x):
        if self.is_discrete:
            return self.scipy_dist.pmf(x, **self.params)
        return self.scipy_dist.pdf(x, **self.params)

    def logpdf(self, x):
        if self.is_discrete:
            return self.scipy_dist.logpmf(x, **self.params)
        return self.scipy_dist.logpdf(x, **self.params)

    def cdf(self, x):
        return self.scipy_dist.cdf(x, **self.params)

    def sf(self, x):
        return self.scipy_dist.sf(x, **self.params)

    def logcdf(self, x):
        return self.scipy_dist.logcdf(x, **self.params)

    def logsf(self, x):
        return self.scipy_dist.logsf(x, **self.params)

    def ppf(self, p):
        return self.scipy_dist.ppf(p, **self.params)

    def isf(self, p):
        return self.scipy_dist.isf(p, **self.params)

    def hazard(self, x):
        return float(self.pdf(x)) / float(self.sf(x))

    def chf(self, x):
        return -float(self.logsf(x))

    def mean(self):
        return self.scipy_dist.mean(**self.params)

    def std(self):
        return self.scipy_dist.std(**self.params)

    def var(self):
        return self.scipy_dist.var(**self.params)

    def median(self):
        return self.scipy_dist.median(**self.params)

    def skewness(self):
        return float(self.scipy_dist.stats(moments="s", **self.params))

    def kurtosis_excess(self):
        return float(self.scipy_dist.stats(moments="k", **self.params))

    def kurtosis(self):
        return self.kurtosis_excess() + 3.0

    def mode(self):
        if self.mode_fn:
            return self.mode_fn()
        raise NotImplementedError(f"mode not defined for {self.name}")

    def get_support(self):
        if self.support_val:
            return self.support_val
        return self.scipy_dist.support(**self.params)

    def get_range(self):
        if self.range_val:
            return self.range_val
        return self.get_support()


def generate_test(d: Distribution) -> str:
    """Generate a complete .test file for a distribution."""
    lines = []

    def add(s=""):
        lines.append(s)

    prefix = f"dist_{d.name}"
    x0 = d.test_x[0]
    x1 = d.test_x[1] if len(d.test_x) > 1 else d.test_x[0]

    def fmt_x(v):
        """Format x value for SQL."""
        if d.x_fmt == "bigint":
            return f"{int(v)}::BIGINT"
        if d.x_is_int:
            return str(int(v))
        return str(v)

    add(f"# name: test/sql/{d.name}.test")
    add(f"# description: test {d.name} distribution")
    add("# group: [sql]")
    add()
    add("require stochastic")
    add()

    # --- PDF ---
    add(f"# Test {d.name} PDF")
    for x in d.test_x[:3]:
        val = d.pdf(x)
        add("query R")
        add(f"SELECT round({prefix}_pdf({d.param_sql}, {fmt_x(x)}), 6);")
        add("----")
        add(fmt(val))
        add()

    # --- Log PDF ---
    add(f"# Test {d.name} log PDF")
    val = d.logpdf(x0)
    add("query R")
    add(f"SELECT round({prefix}_log_pdf({d.param_sql}, {fmt_x(x0)}), 6);")
    add("----")
    add(fmt(val))
    add()

    # --- CDF ---
    add(f"# Test {d.name} CDF")
    for x in d.test_x[:3]:
        val = d.cdf(x)
        add("query R")
        add(f"SELECT round({prefix}_cdf({d.param_sql}, {fmt_x(x)}), 6);")
        add("----")
        add(fmt(val))
        add()

    # --- CDF complement ---
    add(f"# Test {d.name} CDF complement")
    val = d.sf(x0)
    add("query R")
    add(f"SELECT round({prefix}_cdf_complement({d.param_sql}, {fmt_x(x0)}), 6);")
    add("----")
    add(fmt(val))
    add()

    # --- CDF + CDF complement = 1 ---
    add(f"# Verify CDF + CDF complement = 1")
    add("query R")
    add(
        f"SELECT round({prefix}_cdf({d.param_sql}, {fmt_x(x1)}) + {prefix}_cdf_complement({d.param_sql}, {fmt_x(x1)}), 1);"
    )
    add("----")
    add("1.0")
    add()

    # --- Log CDF ---
    add(f"# Test {d.name} log CDF")
    val = d.logcdf(x1)
    add("query R")
    add(f"SELECT round({prefix}_log_cdf({d.param_sql}, {fmt_x(x1)}), 6);")
    add("----")
    add(fmt(val))
    add()

    # --- Log CDF complement ---
    add(f"# Test {d.name} log CDF complement")
    val = d.logsf(x1)
    add("query R")
    add(f"SELECT round({prefix}_log_cdf_complement({d.param_sql}, {fmt_x(x1)}), 6);")
    add("----")
    add(fmt(val))
    add()

    # --- Quantile ---
    if not d.skip_quantile:
        add(f"# Test {d.name} quantile (inverse CDF)")
        for p in [0.5, 0.95]:
            val = d.ppf(p)
            if d.quantile_is_int:
                add("query I")
                add(f"SELECT {prefix}_quantile({d.param_sql}, {p});")
                add("----")
                add(str(int(val)))
            else:
                add("query R")
                add(f"SELECT round({prefix}_quantile({d.param_sql}, {p}), 6);")
                add("----")
                add(fmt(val))
            add()

        # --- Quantile complement ---
        add(f"# Test {d.name} quantile complement")
        val = d.isf(0.05)
        if d.quantile_is_int:
            add("query I")
            add(f"SELECT {prefix}_quantile_complement({d.param_sql}, 0.05);")
            add("----")
            add(str(int(val)))
        else:
            add("query R")
            add(f"SELECT round({prefix}_quantile_complement({d.param_sql}, 0.05), 6);")
            add("----")
            add(fmt(val))
        add()

    # --- Hazard ---
    add(f"# Test {d.name} hazard function")
    val = d.hazard(x0)
    add("query R")
    add(f"SELECT round({prefix}_hazard({d.param_sql}, {fmt_x(x0)}), 6);")
    add("----")
    add(fmt(val))
    add()

    # --- CHF ---
    add(f"# Test {d.name} cumulative hazard function")
    val = d.chf(x0)
    add("query R")
    add(f"SELECT round({prefix}_chf({d.param_sql}, {fmt_x(x0)}), 6);")
    add("----")
    add(fmt(val))
    add()

    # --- Properties ---
    props = [
        ("mean", lambda: d.mean()),
        ("stddev", lambda: d.std()),
        ("variance", lambda: d.var()),
        ("mode", lambda: d.mode()),
        ("median", lambda: d.median()),
        ("skewness", lambda: d.skewness()),
        ("kurtosis", lambda: d.kurtosis()),
        ("kurtosis_excess", lambda: d.kurtosis_excess()),
    ]

    for prop_name, fn in props:
        if prop_name in d.disabled_props:
            continue
        try:
            val = fn()
            if val is None or (isinstance(val, float) and (math.isnan(val))):
                continue
            add(f"# Test {d.name} {prop_name}")
            add("query R")
            add(f"SELECT round({prefix}_{prop_name}({d.param_sql}), 6);")
            add("----")
            add(fmt(val))
            add()
        except (NotImplementedError, ValueError, ZeroDivisionError):
            continue

    # --- Range (verify it executes, Boost.Math may return DBL_MAX instead of inf) ---
    add(f"# Test {d.name} range executes without error")
    add("query I")
    add(f"SELECT {prefix}_range({d.param_sql}) IS NOT NULL;")
    add("----")
    add("true")
    add()

    # --- Support ---
    add(f"# Test {d.name} support executes without error")
    add("query I")
    add(f"SELECT {prefix}_support({d.param_sql}) IS NOT NULL;")
    add("----")
    add("true")
    add()

    # --- Sample ---
    if d.has_sample:
        add(f"# Test {d.name} sample returns varying values")
        add("query I")
        add(
            f"SELECT COUNT(DISTINCT {prefix}_sample({d.param_sql})) > 1 FROM generate_series(1, 100);"
        )
        add("----")
        add("true")
        add()

    return "\n".join(lines)


# ============================================================================
# Distribution definitions
# ============================================================================

DISTRIBUTIONS = []


def add_dist(d):
    DISTRIBUTIONS.append(d)


# --- normal ---
add_dist(
    Distribution(
        "normal",
        stats.norm,
        {"loc": 0.0, "scale": 1.0},
        "0.0, 1.0",
        [0.0, 1.0, -1.0],
        mode_fn=lambda: 0.0,
        range_val=(-math.inf, math.inf),
        support_val=(-math.inf, math.inf),
    )
)

# --- beta ---
add_dist(
    Distribution(
        "beta",
        stats.beta,
        {"a": 2.0, "b": 5.0},
        "2.0, 5.0",
        [0.3, 0.5, 0.1],
        mode_fn=lambda: (2.0 - 1) / (2.0 + 5.0 - 2),  # (a-1)/(a+b-2)
    )
)

# --- binomial ---
add_dist(
    Distribution(
        "binomial",
        stats.binom,
        {"n": 10, "p": 0.5},
        "10::BIGINT, 0.5",
        [5, 3, 7],
        is_discrete=True,
        sample_type="I",
        disabled_props={"mean", "stddev"},
        mode_fn=lambda: float(int((10 + 1) * 0.5)),  # floor((n+1)p)
        range_val=(0.0, 10.0),
        quantile_is_int=True,
    )
)

# --- cauchy ---
add_dist(
    Distribution(
        "cauchy",
        stats.cauchy,
        {"loc": 0.0, "scale": 1.0},
        "0.0, 1.0",
        [0.0, 1.0, -1.0],
        disabled_props={
            "mean",
            "stddev",
            "variance",
            "skewness",
            "kurtosis",
            "kurtosis_excess",
        },
        mode_fn=lambda: 0.0,
        range_val=(-math.inf, math.inf),
        support_val=(-math.inf, math.inf),
    )
)

# --- chi_squared ---
add_dist(
    Distribution(
        "chi_squared",
        stats.chi2,
        {"df": 5},
        "5.0",
        [3.0, 5.0, 1.0],
        mode_fn=lambda: max(5.0 - 2, 0),  # df - 2
    )
)

# --- exponential ---
# Extension uses rate, scipy uses scale=1/rate
add_dist(
    Distribution(
        "exponential",
        stats.expon,
        {"scale": 1.0},  # rate=1.0, so scale=1/1=1.0
        "1.0",
        [0.5, 1.0, 2.0],
        mode_fn=lambda: 0.0,
    )
)

# --- extreme_value (Gumbel) ---
add_dist(
    Distribution(
        "extreme_value",
        stats.gumbel_r,
        {"loc": 0.0, "scale": 1.0},
        "0.0, 1.0",
        [0.0, 1.0, -1.0],
        disabled_props={"mean", "stddev"},
        mode_fn=lambda: 0.0,  # mode = location
        range_val=(-math.inf, math.inf),
        support_val=(-math.inf, math.inf),
    )
)

# --- fisher_f ---
add_dist(
    Distribution(
        "fisher_f",
        stats.f,
        {"dfn": 5, "dfd": 10},
        "5.0, 10.0",
        [1.0, 2.0, 0.5],
        disabled_props={"mean", "stddev"},
        mode_fn=lambda: ((5.0 - 2) / 5.0) * (10.0 / (10.0 + 2)),  # (d1-2)/d1 * d2/(d2+2)
    )
)

# --- gamma ---
add_dist(
    Distribution(
        "gamma",
        stats.gamma,
        {"a": 2.0, "scale": 1.0},
        "2.0, 1.0",
        [1.0, 2.0, 0.5],
        mode_fn=lambda: (2.0 - 1) * 1.0,  # (alpha-1)*beta
    )
)

# --- geometric ---
# Boost.Math geometric: P(k) = p*(1-p)^k, k=0,1,2,...
# scipy.stats.geom: P(k) = p*(1-p)^(k-1), k=1,2,3,...
# So our pdf(p, k) = geom.pmf(k+1, p) and cdf(p, k) = geom.cdf(k+1, p)
class GeometricDist(Distribution):
    def __init__(self):
        super().__init__(
            "geometric",
            stats.geom,
            {"p": 0.5},
            "0.5",
            [0, 1, 2],
            is_discrete=True,
            sample_type="I",
            mode_fn=lambda: 0.0,  # mode is always 0 for geometric (failures before first success)
        )

    def pdf(self, x):
        return stats.geom.pmf(x + 1, **self.params)

    def logpdf(self, x):
        return stats.geom.logpmf(x + 1, **self.params)

    def cdf(self, x):
        return stats.geom.cdf(x + 1, **self.params)

    def sf(self, x):
        return stats.geom.sf(x + 1, **self.params)

    def logcdf(self, x):
        return stats.geom.logcdf(x + 1, **self.params)

    def logsf(self, x):
        return stats.geom.logsf(x + 1, **self.params)

    def ppf(self, p):
        # Boost.Math returns real-valued quantile: solve 1-(1-prob)^(x+1) = p
        prob = self.params["p"]
        return math.log(1 - p) / math.log(1 - prob) - 1

    def isf(self, p):
        # isf(p) = ppf(1-p)
        return self.ppf(1 - p)

    def mean(self):
        return (1 - 0.5) / 0.5  # (1-p)/p

    def std(self):
        p = 0.5
        return math.sqrt((1 - p) / (p * p))

    def var(self):
        p = 0.5
        return (1 - p) / (p * p)

    def median(self):
        return float(self.ppf(0.5))

    def skewness(self):
        p = 0.5
        return (2 - p) / math.sqrt(1 - p)

    def kurtosis_excess(self):
        p = 0.5
        return 6 + (p * p) / (1 - p)

    def get_support(self):
        return (0.0, math.inf)

    def get_range(self):
        return (0.0, math.inf)


add_dist(GeometricDist())

# --- laplace ---
add_dist(
    Distribution(
        "laplace",
        stats.laplace,
        {"loc": 0.0, "scale": 1.0},
        "0.0, 1.0",
        [0.0, 1.0, -1.0],
        mode_fn=lambda: 0.0,
        range_val=(-math.inf, math.inf),
        support_val=(-math.inf, math.inf),
    )
)

# --- logistic ---
add_dist(
    Distribution(
        "logistic",
        stats.logistic,
        {"loc": 0.0, "scale": 1.0},
        "0.0, 1.0",
        [0.0, 1.0, -1.0],
        has_sample=False,
        disabled_props={"mean", "stddev"},
        mode_fn=lambda: 0.0,
        range_val=(-math.inf, math.inf),
        support_val=(-math.inf, math.inf),
    )
)

# --- lognormal ---
# Extension: lognormal(mu, sigma) where mu,sigma are params of underlying normal
# Scipy: lognorm(s=sigma, scale=exp(mu))
add_dist(
    Distribution(
        "lognormal",
        stats.lognorm,
        {"s": 1.0, "scale": math.exp(0.0)},  # mu=0, sigma=1
        "0.0, 1.0",
        [1.0, 2.0, 0.5],
        mode_fn=lambda: math.exp(0.0 - 1.0 * 1.0),  # exp(mu - sigma^2)
    )
)

# --- negative_binomial ---
add_dist(
    Distribution(
        "negative_binomial",
        stats.nbinom,
        {"n": 10, "p": 0.5},
        "10::BIGINT, 0.5",
        [5, 10, 3],
        is_discrete=True,
        sample_type="I",
        disabled_props={"mean", "stddev"},
        mode_fn=lambda: float(
            max(0, int(math.floor((10 - 1) * (1 - 0.5) / 0.5)))
        ),  # floor((r-1)(1-p)/p)
        quantile_is_int=True,
    )
)

# --- pareto ---
# Extension: pareto(shape, minimum) where param1→Boost's scale (x_m), param2→Boost's shape (alpha)
# So dist_pareto_*(1.0, 3.0, ...) creates Boost pareto(scale=1.0, shape=3.0)
# For scipy: pareto(b=alpha, scale=x_m) = pareto(b=3.0, scale=1.0)
add_dist(
    Distribution(
        "pareto",
        stats.pareto,
        {"b": 3.0, "scale": 1.0},  # shape=3.0 (alpha), scale=1.0 (x_m)
        "1.0, 3.0",  # Extension: param1=scale=1.0, param2=shape=3.0
        [1.5, 2.0, 3.0],
        has_sample=False,
        mode_fn=lambda: 1.0,  # x_m (scale parameter)
    )
)

# --- poisson ---
add_dist(
    Distribution(
        "poisson",
        stats.poisson,
        {"mu": 5.0},
        "5.0",
        [5, 3, 7],
        is_discrete=True,
        sample_type="I",
        mode_fn=lambda: 5.0,  # floor(lambda) when lambda is integer
    )
)

# --- rayleigh ---
add_dist(
    Distribution(
        "rayleigh",
        stats.rayleigh,
        {"scale": 1.0},
        "1.0",
        [0.5, 1.0, 2.0],
        has_sample=False,
        mode_fn=lambda: 1.0,  # mode = sigma
    )
)

# --- students_t ---
add_dist(
    Distribution(
        "students_t",
        stats.t,
        {"df": 10},
        "10.0",
        [0.0, 1.0, -1.0],
        mode_fn=lambda: 0.0,
        range_val=(-math.inf, math.inf),
        support_val=(-math.inf, math.inf),
    )
)

# --- uniform_real ---
add_dist(
    Distribution(
        "uniform_real",
        stats.uniform,
        {"loc": 0.0, "scale": 1.0},  # uniform on [0, 1]
        "0.0, 1.0",
        [0.5, 0.25, 0.75],
        mode_fn=lambda: 0.0,  # Boost.Math returns lower bound for uniform mode
    )
)

# --- uniform_int ---
# Extension uses boost::math::uniform_distribution (continuous) for math, discrete for sampling.
# So math functions (pdf, cdf, etc.) behave as continuous uniform on [min, max].
add_dist(
    Distribution(
        "uniform_int",
        stats.uniform,
        {"loc": 1.0, "scale": 5.0},  # continuous uniform on [1, 6]
        "1::DOUBLE, 6::DOUBLE",
        [2, 3, 5],
        x_fmt="bigint",  # x param is registered as BIGINT
        sample_type="I",
        disabled_props={"mean", "stddev", "variance", "median"},  # BIGINT truncation
        mode_fn=lambda: 1.0,  # Boost returns lower bound for uniform mode
        range_val=(1.0, 6.0),
        support_val=(1.0, 6.0),
        quantile_is_int=True,
    )
)

# --- weibull ---
add_dist(
    Distribution(
        "weibull",
        stats.weibull_min,
        {"c": 1.5, "scale": 1.0},
        "1.5, 1.0",
        [0.5, 1.0, 2.0],
        disabled_props={"mean", "stddev"},
        mode_fn=lambda: 1.0 * ((1.5 - 1) / 1.5) ** (1 / 1.5)
        if 1.5 > 1
        else 0.0,  # scale*((k-1)/k)^(1/k)
    )
)

# --- bernoulli ---
add_dist(
    Distribution(
        "bernoulli",
        stats.bernoulli,
        {"p": 0.5},
        "0.5",
        [0, 1],
        is_discrete=True,
        sample_type="B",
        mode_fn=lambda: 0.0,  # Boost.Math returns 0 when p=0.5 (ambiguous)
    )
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for d in DISTRIBUTIONS:
        content = generate_test(d)
        path = os.path.join(OUTPUT_DIR, f"{d.name}.test")
        with open(path, "w") as f:
            f.write(content)
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
