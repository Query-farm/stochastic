# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DuckDB extension called "stochastic" developed by Query.Farm that provides comprehensive statistical distribution functions to DuckDB. The extension enables advanced statistical analysis, probability calculations, and random sampling directly within SQL queries.

The extension supports 20+ probability distributions (normal, beta, binomial, poisson, etc.) with functions for PDF/PMF, CDF, quantile, sampling, and distribution properties (mean, variance, etc.).

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

The extension follows the DuckDB extension template structure with these key components:

**Entry Point (`src/stochastic_extension.cpp`)**: Registers all distribution functions via `Load_<distribution>_distribution()` functions. Also includes Query.Farm telemetry (can be opted out via `QUERY_FARM_TELEMETRY_OPT_OUT` environment variable).

**Distribution Implementations (`src/distribution_*.cpp`)**: Each distribution is in its own file (e.g., `distribution_normal.cpp`, `distribution_beta.cpp`). Each file:
- Uses preprocessor macros to define distribution name and type mappings
- Specializes `distribution_traits` template for both Boost.Math and Boost.Random versions
- Registers all functions (sample, pdf, cdf, quantile, mean, variance, etc.) using the `REGISTER` macro

**Function Registration System (`src/include/utils.hpp`)**: Template-based system that:
- Uses `distribution_traits` to determine parameter types and names
- Automatically generates function names with pattern: `dist_<prefix>_<function>` (e.g., `dist_normal_pdf`)
- Handles both unary (1 param) and binary (2 param) distributions
- Optimizes for constant vectors by creating distribution once and reusing

**RNG Management (`src/rng_utils.cpp`, `src/include/rng_utils.hpp`)**: Thread-local Mersenne Twister RNG (`boost::random::mt19937`) with:
- Fixed global seed (12345) for reproducibility
- Unique per-thread seeding to ensure thread-safety
- Thread index mapping for deterministic parallel execution

**Type System (`src/include/distribution_traits.hpp`, `src/include/callable_traits.hpp`)**:
- `distribution_traits`: Maps C++ types to DuckDB LogicalTypes
- `callable_traits`: Template metaprogramming for extracting function signatures
- `logical_type_map`: Supports double, int64_t, and pair<double,double> (for ranges)

### Adding a New Distribution

To add a new distribution:

1. Create `src/distribution_<name>.cpp` following the pattern in existing files
2. Define the distribution traits with parameter names, types, and validation
3. Register all desired functions (sample, pdf, cdf, etc.) in the `Load_<name>_distribution` function
4. Add the source file to `EXTENSION_SOURCES` in `CMakeLists.txt`
5. Add forward declaration and call to `Load_<name>_distribution()` in `src/stochastic_extension.cpp`
6. Create test file `test/sql/<name>.test` with comprehensive test cases

### Testing

Tests use DuckDB's `.test` format (see `test/sql/normal.test` for examples). Test files:
- Start with metadata comments (`# name:`, `# description:`, `# group:`)
- Use `require stochastic` to ensure extension is loaded
- Support `statement error`, `query R` (float), `query I` (integer), `query T` (text)
- Can test multiple columns with `query RR`, `query RRR`, etc.

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
- `dist_normal_pdf(mean, stddev, x)` - Probability density at x
- `dist_normal_cdf(mean, stddev, x)` - Cumulative distribution at x
- `dist_normal_quantile(mean, stddev, p)` - Inverse CDF (p-th quantile)
- `dist_normal_mean(mean, stddev)` - Distribution mean
- `dist_normal_variance(mean, stddev)` - Distribution variance

### Code Organization

```
src/
├── stochastic_extension.cpp          # Extension entry point
├── distribution_*.cpp                # Individual distribution implementations (20+ files)
├── rng_utils.cpp                     # Thread-local RNG management
├── query_farm_telemetry.cpp         # Optional telemetry
└── include/
    ├── utils.hpp                     # Template-based function registration system
    ├── distribution_traits.hpp       # Type system for distributions
    ├── callable_traits.hpp           # Template metaprogramming utilities
    ├── rng_utils.hpp                 # RNG declarations
    └── ...

test/sql/
└── *.test                            # DuckDB test files

duckdb/                               # Git submodule - DuckDB source
extension-ci-tools/                   # Git submodule - shared build infrastructure
```

### Key Implementation Details

**Constant Vector Optimization**: When all parameters are constant, the distribution object is created once and reused for all rows (see `DistributionSampleUnary` and `DistributionSampleBinary` in `utils.hpp`).

**Parameter Validation**: Each distribution validates its parameters in `distribution_traits::ValidateParameters()` and throws `InvalidInputException` for invalid inputs.

**Thread Safety**: The thread-local RNG ensures thread-safe random number generation without locks in the hot path.

**Telemetry**: The extension sends anonymous usage telemetry to Query.Farm (opt-out via environment variable). This happens once per extension load and includes DuckDB version, platform, and extension version. This telemetry is opt-out and does not need to be documented as Query.Farm documents it on its own website.
