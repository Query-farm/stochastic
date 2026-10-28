# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DuckDB extension called "stochastic" developed by Query.Farm that provides comprehensive statistical distribution functions to DuckDB. The extension enables advanced statistical analysis, probability calculations, random sampling, distribution fitting and hypothesis tests directly within SQL queries.

The extension supports 36 probability distributions (normal, beta, binomial, poisson, the non-central family, the multivariate Dirichlet and multinomial, etc.) with functions for PDF/PMF, CDF, quantile, sampling, and distribution properties (mean, variance, etc.), plus maximum likelihood fit aggregates and hypothesis test aggregates.

The `v2.0` branch targets DuckDB v2.0 and is compiled as C++17.

## Build Commands

This project uses a Makefile wrapper around CMake. All build commands are run from the project root:

```bash
# Build release version (default) needed to run tests
VCPKG_TOOLCHAIN_PATH=`pwd`/vcpkg/scripts/buildsystems/vcpkg.cmake GEN=ninja make release

# Build debug version
VCPKG_TOOLCHAIN_PATH=`pwd`/vcpkg/scripts/buildsystems/vcpkg.cmake GEN=ninja make debug

# Run tests (uses release build)
make test

# Run tests with debug build
make test_debug
```

### Running a Single Test

To run a specific test file:
```bash
./build/release/test/unittest "test/sql/normal.test"
```


All extension functions should be documented inside of DuckDB with CreateScalarFunctionInfo or CreateAggregateFunctionInfo or the appropriate type for the function.  This documentation of the function should include examples, parameter types and parameter names.  The function should be categorized.

When making changes the version should always be updated to the current date plus an ordinal counter in the form of YYYYMMDDCC.


## Architecture

### Extension Structure

**Entry Point (`src/stochastic_extension.cpp`)**: The `STOCHASTIC_DISTRIBUTIONS(X)` macro lists every distribution; it declares and calls each `Load_<name>_distribution()`. `Load_fit_functions()` and `Load_hypothesis_tests()` register the aggregates. Also includes Query.Farm telemetry (can be opted out via `QUERY_FARM_TELEMETRY_OPT_OUT` environment variable).

**Distribution Specs (`src/distribution_*.cpp`)**: Each distribution is a small spec struct deriving from `DistributionSpecBase` (see the comment at the top of `src/include/utils.hpp`):
- `name`, `display_name`, `params` (a `std::tuple` of DOUBLE/BIGINT types), `param_names`
- `boost_type` (a Boost.Math distribution using `StochasticPolicy`, or `StochasticDiscretePolicy` for discrete ones), `sampler_type` and `sample_t`
- example arguments for the generated documentation, and `static void Validate(params...)`
- flags: `discrete` (adds `pmf`/`log_pmf`, integer quantiles, no KS test), `quantile_t`, `check_sample_range`

Both `boost_type` and `sampler_type` are constructed from the parameters in declaration order. The file's `Load_<name>_distribution()` calls `RegisterStandardDistribution<Spec>(loader)`. `distribution_uniform_int.cpp` is an exception: it uses an exact integer implementation and registers its functions individually with `RegisterSampling` / `RegisterEvaluation`. The multivariate `distribution_dirichlet.cpp` and `distribution_multinomial.cpp` take LIST parameters and use the helpers in `src/include/multivariate.hpp`; they provide sample, pdf/pmf, log_pdf/log_pmf, mean and variance (per component).

**Registration and Execution (`src/include/utils.hpp`)**:
- `RegisterStandardDistribution<Spec>` registers sample, samples, pdf/log_pdf (and pmf/log_pmf), the CDFs, quantiles, hazard, properties, range/support and (for continuous distributions) the KS test.
- `ExecuteDistribution` is the single scalar executor for any number of parameters: it reads inputs with `Vector::Values<T>()`, writes with `FlatVector::Writer`, propagates NULLs, builds the distribution once when the parameters are constant, and returns a constant result when every input is constant (except for VOLATILE sampling).
- Validation runs `Spec::Validate` first and then rejects any non-finite parameter.
- Properties are wrapped in `UndefinedAsNull`, so undefined moments (e.g. the Cauchy mean) return NULL.
- `DiscreteQuantile` returns the smallest integer x with CDF(x) >= p.
- `CheckSampleRange` rejects integer sampler parameters whose samples could exceed 2^53.
- `DiscreteQuantile` finds discrete quantiles (and medians) by galloping and bisecting on the CDF, since Boost's integer search does not finish for large quantiles; quantiles above 2^53 are rejected.
- A spec can override `DistributionParams`, `Pdf`, `LogPdf`, `LogCdf`, `LogCdfComplement`, `Hazard`, `Quantile`, `QuantileComplement`, `Mode` or `Median` from `DistributionSpecBase` where Boost is wrong or does not terminate (Kolmogorov-Smirnov, logistic, non-central chi-squared and F).
- Results that are NaN are reported as errors, CDFs and PMFs are checked to lie in [0, 1], and Boost exceptions raised outside our policy are turned into `InvalidInputException` naming the distribution.
- The non-central chi-squared, F and beta accept noncentrality up to `MAX_NONCENTRALITY` (1e8) and the non-central t up to 1000 in absolute value; beyond that Boost's series take too long or diverge.

**Boost.Math Policy (`src/include/stochastic_policy.hpp`)**: Domain, pole, evaluation and rounding errors throw `MathError`, which the executors turn into `InvalidInputException` prefixed with the distribution name. Overflow returns ±inf. `promote_double<false>` keeps calculations in DOUBLE (long double is software emulated and ~100x slower on Linux aarch64). `StochasticUndefinedPolicy` adds `assert_undefined<false>`, which turns always-undefined properties (Cauchy moments, non-central beta skewness and kurtosis) into runtime errors instead of compile errors; it is kept off the other distributions because it breaks linking of the non-central t with Apple clang.

**Samplers (`src/include/samplers.hpp`, `src/include/custom_random.hpp`)**: Samplers for distributions the C++ standard library lacks. `InverseTransformSampler<Dist, T>` samples any Boost distribution via its quantile function.

**Aggregates (`src/include/aggregate_utils.hpp`, `src/fit_functions.cpp`, `src/hypothesis_tests.cpp`)**: `RegisterStructAggregate<OP>` builds a STRUCT-of-DOUBLE aggregate from an operation struct (`inputs`, `result_names`, a trivially movable `State`, `Update`, `Combine`, `Finalize`, `Destroy`). `RunningMoments` (mergeable Welford moments), `CollectedValues` and `GroupParameters` are shared building blocks. Fits declare `param_names`, `Count` and `LogLikelihood`; `FitAggregate<OP>` in `src/fit_functions.cpp` appends `log_likelihood`, `aic` and `bic` to the parameters. Fits without sufficient statistics (Weibull, logistic, Cauchy, ...) collect the sample with `CollectedFit`.

**RNG (`src/include/rng_adapter.hpp`)**: Each chunk seeds a local `std::mt19937_64` from DuckDB's per-client `RandomEngine`, so `SELECT setseed(x)` makes sampling reproducible, including in parallel.

**Types (`src/include/distribution_traits.hpp`)**: `logical_type_map` maps C++ types (double, int64_t, bool, pairs for ARRAY results) to DuckDB LogicalTypes.

### Adding a New Distribution

1. Create `src/distribution_<name>.cpp` with a spec struct and `Load_<name>_distribution()` following an existing file (sources are picked up by the `src/distribution_*.cpp` glob in `CMakeLists.txt`)
2. If the standard library has no sampler, add one to `src/include/samplers.hpp` or use `InverseTransformSampler<boost_type>`
3. Add `X(<name>)` to `STOCHASTIC_DISTRIBUTIONS` in `src/stochastic_extension.cpp`
4. Create `test/sql/<name>.test` with reference values from an independent implementation (e.g. SciPy) and sample moment checks
5. Parameter names must not be `x`, `p` or `n` (they collide with call arguments; see `test/sql/parameter_names.test`)
6. Run the fuzzer described in `test/fuzz/README.md` over the new functions

### Testing

Tests use DuckDB's `.test` format (see `test/sql/normal.test` for examples). Test files:
- Start with metadata comments (`# name:`, `# description:`, `# group:`)
- Use `require stochastic` to ensure extension is loaded
- Support `statement error`, `query R` (float), `query I` (integer), `query T` (text)
- Can test multiple columns with `query RR`, `query RRR`, etc.

`make test` runs DuckDB's `scripts/ci/run_tests.py`, which needs Python 3.10 or newer. `./build/release/test/unittest "test/sql/*"` runs the same tests directly. In the v2.0 shell the statically linked extension is not loaded automatically; use `build/release/duckdb -unsigned -c "LOAD 'build/release/extension/stochastic/stochastic.duckdb_extension'; ..."`.

### Dependencies

The extension depends on:
- **Boost.Math**: For statistical distribution calculations (PDF, CDF, quantile)
- **Boost.Random**: For random number generation from distributions
- Dependencies are managed via vcpkg (see `vcpkg.json`)

### Build System

The project uses:
- **CMakeLists.txt**: Defines extension sources and links Boost libraries
- **Makefile**: Thin wrapper that delegates to `extension-ci-tools/makefiles/duckdb_extension.Makefile`
- **extension_config.cmake**: Tells DuckDB build system to load the stochastic extension
- **extension-ci-tools/**: Git submodule with shared DuckDB extension build infrastructure

The `duckdb/` directory is a git submodule containing the DuckDB source code that the extension builds against.

### Function Naming Convention

All functions follow the pattern: `dist_<distribution>_<operation>`

Examples:
- `dist_normal_sample(mean, stddev)` - Generate random sample
- `dist_normal_samples(mean, stddev, n)` - List of n random samples
- `dist_normal_pdf(mean, stddev, x)` - Probability density at x
- `dist_normal_cdf(mean, stddev, x)` - Cumulative distribution at x
- `dist_normal_quantile(mean, stddev, p)` - Inverse CDF (p-th quantile)
- `dist_normal_mean(mean, stddev)` - Distribution mean
- `dist_normal_variance(mean, stddev)` - Distribution variance
- `dist_normal_fit(x)` - Aggregate: maximum likelihood parameters, log-likelihood, AIC and BIC as a STRUCT
- `dist_normal_ks_test(x, mean, stddev)` - Aggregate: Kolmogorov-Smirnov test

Discrete distributions also have `pmf` / `log_pmf`. The hypothesis tests that are not tied to a distribution are `welch_t_test`, `student_t_test`, `one_sample_t_test` and `chi_squared_test`.

Every function is categorized as `[statistics, <kind>, <distribution>]` where kind is one of sampling, density, cumulative, quantile, hazard, property, fit or hypothesis_test. Tests that are not tied to a distribution omit the distribution.

### Code Organization

```
src/
├── stochastic_extension.cpp          # Extension entry point and distribution list
├── distribution_*.cpp                # One file per distribution (36 files)
├── fit_functions.cpp                 # dist_<name>_fit aggregates (22 distributions)
├── hypothesis_tests.cpp              # t-tests and chi-squared test aggregates
├── query_farm_telemetry.cpp          # Optional telemetry
└── include/
    ├── utils.hpp                     # Specs, scalar executor, registration, KS test
    ├── stochastic_policy.hpp         # Boost.Math error policy
    ├── aggregate_utils.hpp           # STRUCT aggregate harness
    ├── samplers.hpp                  # Additional samplers
    ├── custom_random.hpp             # Beta, Laplace, logistic, Pareto, Rayleigh samplers
    ├── distribution_traits.hpp       # C++ to DuckDB type mapping
    ├── rng_adapter.hpp               # Per-chunk RNG seeded from DuckDB's RandomEngine
    └── ...

test/sql/
└── *.test                            # DuckDB test files

duckdb/                               # Git submodule - DuckDB source
extension-ci-tools/                   # Git submodule - shared build infrastructure
```

### Telemetry

The extension sends anonymous usage telemetry to Query.Farm (opt-out via environment variable). This happens once per extension load and includes DuckDB version, platform, and extension version. This telemetry is opt-out and does not need to be documented as Query.Farm documents it on its own website.
