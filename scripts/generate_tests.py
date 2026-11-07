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


def is_half_way(v, decimals=6):
    """True when v lies exactly half-way between two values at `decimals` places.

    Native and wasm builds round such values in opposite directions, so tests round them to one place less.
    """
    scaled = float(v) * 10**decimals
    return abs(scaled - math.floor(scaled) - 0.5) < 1e-9


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
        disabled_props=None,
        range_val=None,
        support_val=None,
        mode_fn=None,
        quantile_is_int=False,
        skip_quantile=False,
        x_fmt=None,  # "bigint" to force ::BIGINT cast on x values
        extra_tests=None,  # hand-written sqllogictest text appended to the file
    ):
        self.name = name
        self.scipy_dist = scipy_dist
        self.params = params  # dict of scipy params
        self.param_sql = param_sql  # SQL params string e.g. "0.0, 1.0"
        self.test_x = test_x  # list of x values to test pdf/cdf at
        self.is_discrete = is_discrete
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
        self.extra_tests = extra_tests

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
        decimals = 6
        if is_half_way(val):
            decimals = 5
            exact = f"{float(val):.7f}"
            down, up = exact[1:-1], f"{float(exact[:-1]) + 1e-6:.6f}"[1:]
            add(f"# round to 5 dp: the exact value is {exact}, a half-way point at 6 dp where")
            add(f"# native and wasm builds round in opposite directions ({down} vs {up}).")
        add("query R")
        add(f"SELECT round({prefix}_cdf({d.param_sql}, {fmt_x(x)}), {decimals});")
        add("----")
        add(fmt(val, decimals))
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

    # Sample moments are checked with tolerances in sampling.test, and parameter
    # validation, boundary quantiles and infinite evaluation points in validation.test.
    if d.extra_tests:
        add(d.extra_tests.strip("\n"))
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
        # Discrete quantiles are the smallest integer x with CDF(x) = 1-(1-prob)^(x+1) >= p
        prob = self.params["p"]
        return float(math.ceil(math.log(1 - p) / math.log(1 - prob) - 1 - 1e-12))

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
        disabled_props={"mean", "stddev"},
        mode_fn=lambda: float(
            max(0, int(math.floor((10 - 1) * (1 - 0.5) / 0.5)))
        ),  # floor((r-1)(1-p)/p)
        quantile_is_int=True,
    )
)

PARETO_EXTRA_TESTS = """
# Parameter metadata agrees with the established (scale, shape) runtime order.
query T
SELECT DISTINCT parameters[1:2] FROM duckdb_functions()
WHERE function_name LIKE 'dist_pareto_%' AND function_type = 'scalar';
----
[scale, shape]

query RR
SELECT dist_pareto_mean(1,3), dist_pareto_mean(2,3);
----
1.5	3.0
"""

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
        mode_fn=lambda: 1.0,  # x_m (scale parameter)
        extra_tests=PARETO_EXTRA_TESTS,
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
# Not generated: dist_uniform_int_* use an exact integer implementation (BIGINT bounds, discrete pmf),
# so test/sql/uniform_int.test is written by hand.

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
        mode_fn=lambda: 0.0,  # Boost.Math returns 0 when p=0.5 (ambiguous)
    )
)


# ============================================================================
# validation.test: parameter validation, boundary quantiles and infinite
# evaluation points for every univariate distribution
# ============================================================================


class Validation:
    """Validation checks for one distribution.

    params: SQL literals of valid parameters; types: "D" (DOUBLE) or "I" (BIGINT) per parameter;
    invalid: (params SQL, expected error) pairs; q0 / q1: expected quantile at p = 0 and p = 1
    (a value, or "error: <message>"); x: a valid evaluation point; discrete: BIGINT evaluation points.
    """

    def __init__(self, name, params, names, types, invalid, q0, q1, x, *, discrete=False, finite_errors=None):
        self.name = name
        self.params = params
        self.names = names
        self.types = types
        self.invalid = invalid
        self.q0 = q0
        self.q1 = q1
        self.x = x
        self.discrete = discrete
        # Expected error for a non-finite parameter, when the distribution's own checks report it first
        self.finite_errors = finite_errors or {}


INF = "inf"
NINF = "-inf"

VALIDATIONS = [
    Validation("arcsine", ["0.0", "1.0"], ["min", "max"], "DD",
               [("1.0, 0.0", "arcsine: min must be < max, got min=1 max=0")], "0.0", "1.0", "0.25"),
    Validation("bernoulli", ["0.5"], ["prob"], "D",
               [("1.5", "bernoulli: prob must be between 0 and 1, got 1.5")], "0.0", "1.0", "1", discrete=True),
    Validation("beta", ["2.0", "5.0"], ["alpha", "beta"], "DD",
               [("0.0, 5.0", "beta: alpha must be > 0, got 0"), ("2.0, -1.0", "beta: beta must be > 0, got -1")],
               "0.0", "1.0", "0.5"),
    Validation("binomial", ["10", "0.5"], ["trials", "prob"], "ID",
               [("0, 0.5", "binomial: trials must be > 0, got 0"),
                ("10, -0.1", "binomial: prob must be between 0 and 1, got -0.1")], "0", "10", "5", discrete=True),
    Validation("cauchy", ["0.0", "1.0"], ["location", "scale"], "DD",
               [("0.0, 0.0", "cauchy: scale must be > 0, got 0")], NINF, INF, "0.5"),
    Validation("chi_squared", ["5.0"], ["degrees_of_freedom"], "D",
               [("0.0", "chi_squared: degrees_of_freedom must be > 0, got 0")], "0.0", INF, "3.0"),
    Validation("exponential", ["1.0"], ["rate"], "D",
               [("-1.0", "exponential: rate must be > 0, got -1")], "0.0", INF, "0.5"),
    Validation("extreme_value", ["0.0", "1.0"], ["location", "scale"], "DD",
               [("0.0, 0.0", "extreme_value: scale must be > 0, got 0")], NINF, INF, "0.5"),
    Validation("fisher_f", ["5.0", "10.0"], ["d1", "d2"], "DD",
               [("0.0, 10.0", "fisher_f: d1 must be > 0, got 0"), ("5.0, 0.0", "fisher_f: d2 must be > 0, got 0")],
               "0.0", INF, "0.5"),
    Validation("gamma", ["2.0", "1.0"], ["shape", "scale"], "DD",
               [("0.0, 1.0", "gamma: shape must be > 0, got 0"), ("2.0, -1.0", "gamma: scale must be > 0, got -1")],
               "0.0", INF, "1.5"),
    Validation("geometric", ["0.5"], ["prob"], "D",
               [("0.0", "geometric: prob must be > 0, got 0"),
                ("1.5", "geometric: prob must be between 0 and 1, got 1.5")], "0.0", INF, "2", discrete=True),
    Validation("hypergeometric", ["10", "5", "50"], ["successes", "draws", "population"], "III",
               [("-1, 5, 50", "hypergeometric: successes must be >= 0, got -1"),
                ("10, 5, 0", "hypergeometric: population must be > 0, got 0"),
                ("60, 5, 50", "hypergeometric: successes and draws must be <= population")],
               "0", "5", "2", discrete=True),
    Validation("inverse_chi_squared", ["5.0", "1.0"], ["degrees_of_freedom", "scale"], "DD",
               [("0.0, 1.0", "inverse_chi_squared: degrees_of_freedom must be > 0, got 0"),
                ("5.0, 0.0", "inverse_chi_squared: scale must be > 0, got 0")], "0.0", INF, "0.5"),
    Validation("inverse_gamma", ["3.0", "2.0"], ["shape", "scale"], "DD",
               [("0.0, 2.0", "inverse_gamma: shape must be > 0, got 0"),
                ("3.0, 0.0", "inverse_gamma: scale must be > 0, got 0")], "0.0", INF, "0.5"),
    Validation("inverse_gaussian", ["1.0", "2.0"], ["mean", "shape"], "DD",
               [("0.0, 2.0", "inverse_gaussian: mean must be > 0, got 0"),
                ("1.0, 0.0", "inverse_gaussian: shape must be > 0, got 0")], "0.0", INF, "0.5"),
    Validation("kolmogorov_smirnov", ["10.0"], ["sample_size"], "D",
               [("0.0", "kolmogorov_smirnov: sample_size must be > 0, got 0")], "0.0", INF, "0.3"),
    Validation("laplace", ["0.0", "1.0"], ["location", "scale"], "DD",
               [("0.0, 0.0", "laplace: scale must be > 0, got 0")], NINF, INF, "0.5"),
    Validation("logistic", ["0.0", "1.0"], ["location", "scale"], "DD",
               [("0.0, 0.0", "logistic: scale must be > 0, got 0")], NINF, INF, "0.5"),
    Validation("lognormal", ["0.0", "1.0"], ["mean", "stddev"], "DD",
               [("0.0, 0.0", "lognormal: stddev must be > 0, got 0")], "0.0", INF, "0.5"),
    Validation("negative_binomial", ["10", "0.5"], ["successes", "prob"], "ID",
               [("0, 0.5", "negative_binomial: successes must be > 0, got 0"),
                ("10, 0.0", "negative_binomial: prob must be > 0, got 0")],
               "0", "error: negative_binomial: quantile is inf and cannot be returned as BIGINT", "5", discrete=True),
    Validation("non_central_beta", ["2.0", "3.0", "1.0"], ["alpha", "beta", "noncentrality"], "DDD",
               [("0.0, 3.0, 1.0", "non_central_beta: alpha must be > 0, got 0"),
                ("2.0, 3.0, -1.0", "non_central_beta: noncentrality must be >= 0, got -1"),
                ("2.0, 3.0, 1e9", "non_central_beta: noncentrality must be <= 100000000, got 1000000000")],
               "0.0", "1.0", "0.5"),
    Validation("non_central_chi_squared", ["4.0", "2.0"], ["degrees_of_freedom", "noncentrality"], "DD",
               [("0.0, 2.0", "non_central_chi_squared: degrees_of_freedom must be > 0, got 0"),
                ("4.0, -1.0", "non_central_chi_squared: noncentrality must be >= 0, got -1")], "0.0", INF, "3.0"),
    Validation("non_central_f", ["5.0", "10.0", "2.0"], ["d1", "d2", "noncentrality"], "DDD",
               [("0.0, 10.0, 2.0", "non_central_f: d1 must be > 0, got 0"),
                ("5.0, 10.0, -1.0", "non_central_f: noncentrality must be >= 0, got -1")], "0.0", INF, "1.5"),
    Validation("non_central_t", ["10.0", "1.0"], ["degrees_of_freedom", "noncentrality"], "DD",
               [("0.0, 1.0", "non_central_t: degrees_of_freedom must be > 0, got 0"),
                ("10.0, 2000.0", "non_central_t: absolute value of noncentrality must be <= 1000, got 2000")],
               NINF, INF, "1.5"),
    Validation("normal", ["0.0", "1.0"], ["mean", "stddev"], "DD",
               [("0.0, 0.0", "normal: stddev must be > 0, got 0")], NINF, INF, "0.5"),
    Validation("pareto", ["1.0", "3.0"], ["scale", "shape"], "DD",
               [("0.0, 3.0", "pareto: scale must be > 0, got 0"), ("1.0, 0.0", "pareto: shape must be > 0, got 0")],
               "1.0", INF, "1.5"),
    Validation("poisson", ["5.0"], ["rate"], "D",
               [("0.0", "poisson: rate must be > 0, got 0")], "0.0", INF, "3", discrete=True),
    Validation("rayleigh", ["1.0"], ["scale"], "D",
               [("0.0", "rayleigh: scale must be > 0, got 0")], "0.0", INF, "0.5"),
    Validation("skew_normal", ["0.0", "1.0", "4.0"], ["location", "scale", "shape"], "DDD",
               [("0.0, 0.0, 4.0", "skew_normal: scale must be > 0, got 0")], NINF, INF, "0.5"),
    Validation("students_t", ["10.0"], ["degrees_of_freedom"], "D",
               [("0.0", "students_t: degrees_of_freedom must be > 0, got 0")], NINF, INF, "1.5"),
    Validation("triangular", ["0.0", "0.5", "1.0"], ["lower", "mode", "upper"], "DDD",
               [("0.0, 2.0, 1.0", "triangular: parameters must satisfy lower <= mode <= upper and lower < upper"),
                ("1.0, 1.0, 1.0", "triangular: parameters must satisfy lower <= mode <= upper and lower < upper")],
               "0.0", "1.0", "0.25"),
    Validation("uniform_real", ["0.0", "1.0"], ["min", "max"], "DD",
               [("1.0, 1.0", "uniform_real: min must be < max, got min=1 max=1")], "0.0", "1.0", "0.5"),
    Validation("weibull", ["1.5", "1.0"], ["shape", "scale"], "DD",
               [("0.0, 1.0", "weibull: shape must be > 0, got 0"), ("1.5, 0.0", "weibull: scale must be > 0, got 0")],
               "0.0", INF, "0.5"),
]


def generate_validation_test() -> str:
    lines = []

    def add(s=""):
        lines.append(s)

    def expect(sql, expected, qtype):
        if expected.startswith("error: "):
            add("statement error")
            add(sql)
            add("----")
            add(expected[len("error: "):])
        else:
            add(f"query {qtype}")
            add(sql)
            add("----")
            add(expected)
        add()

    add("# name: test/sql/validation.test")
    add("# description: parameter validation, boundary quantiles and infinite evaluation points for every univariate distribution")
    add("# group: [sql]")
    add("# Generated by scripts/generate_tests.py; dist_uniform_int_* are covered by uniform_int.test")
    add()
    add("require stochastic")
    add()
    for v in VALIDATIONS:
        prefix = f"dist_{v.name}"
        params = ", ".join(v.params)
        add(f"# --- {v.name} ---")
        add()
        for params_sql, message in v.invalid:
            add("statement error")
            add(f"SELECT {prefix}_cdf({params_sql}, {v.x});")
            add("----")
            add(message)
            add()
        # NaN and infinite parameters are rejected by every function, including sampling
        for i, (pname, ptype) in enumerate(zip(v.names, v.types)):
            if ptype != "D":
                continue
            for literal, fn, spelled in (("'nan'::DOUBLE", "pdf", "nan"), ("'inf'::DOUBLE", "sample", "inf"),
                                         ("'-inf'::DOUBLE", "mean", "-inf")):
                args = list(v.params)
                args[i] = literal
                call = ", ".join(args) + (f", {v.x}" if fn == "pdf" else "")
                message = v.finite_errors.get((pname, spelled), f"{v.name}: {pname} must be finite, got {spelled}")
                add("statement error")
                add(f"SELECT {prefix}_{fn}({call});")
                add("----")
                add(message)
                add()
        # Probabilities outside [0, 1] are rejected
        add("statement error")
        add(f"SELECT {prefix}_quantile({params}, 1.5);")
        add("----")
        add(f"{v.name}: Probability argument is 1.5, but must be >= 0 and <= 1")
        add()
        add("statement error")
        add(f"SELECT {prefix}_quantile({params}, 'nan'::DOUBLE);")
        add("----")
        add(f"{v.name}: ")
        add()
        # The quantiles at p = 0 and p = 1 are the ends of the support
        qtype = "I" if v.q0.lstrip("-").isdigit() and v.discrete and "." not in v.q0 else "R"
        expect(f"SELECT {prefix}_quantile({params}, 0.0);", v.q0, qtype)
        expect(f"SELECT {prefix}_quantile({params}, 1.0);", v.q1, qtype)
        expect(f"SELECT {prefix}_quantile_complement({params}, 1.0);", v.q0, qtype)
        expect(f"SELECT {prefix}_quantile_complement({params}, 0.0);", v.q1, qtype)
        # Infinite evaluation points are allowed: the CDF is 0 at -inf and 1 at +inf
        if not v.discrete:
            add("query RRRR")
            add(f"SELECT {prefix}_cdf({params}, '-inf'::DOUBLE), {prefix}_cdf({params}, 'inf'::DOUBLE), "
                f"{prefix}_pdf({params}, '-inf'::DOUBLE), {prefix}_pdf({params}, 'inf'::DOUBLE);")
            add("----")
            add("0.0\t1.0\t0.0\t0.0")
            add()
            add("statement error")
            add(f"SELECT {prefix}_cdf({params}, 'nan'::DOUBLE);")
            add("----")
            add(f"{v.name}: ")
            add()
    return "\n".join(lines).rstrip("\n") + "\n"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for d in DISTRIBUTIONS:
        content = generate_test(d)
        path = os.path.join(OUTPUT_DIR, f"{d.name}.test")
        with open(path, "w") as f:
            f.write(content)
        print(f"Generated {path}")
    path = os.path.join(OUTPUT_DIR, "validation.test")
    with open(path, "w") as f:
        f.write(generate_validation_test())
    print(f"Generated {path}")


if __name__ == "__main__":
    main()
