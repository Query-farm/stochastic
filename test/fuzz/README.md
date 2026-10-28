# Fuzzing

`run.sh` calls every function of the extension with edge-case arguments and reports anything other than a clean
result or an `Invalid Input Error` that names the distribution.

```bash
test/fuzz/run.sh                          # release build, results in build/fuzz
test/fuzz/run.sh build/debug build/fuzz   # debug build (DuckDB assertions and vector verification)
```

It needs Python 3.9 or newer and takes under a minute for the release build on a large machine. `SHARDS` sets the
number of parallel CLI processes (default: the CPU count) and `STALL` the seconds without output after which a
statement counts as hung (default 20; use more for debug builds).

## What it does

1. Reads the function list from `duckdb_functions()`.
2. Scalar functions: starting from each function's documented example, every argument is replaced in turn by each
   value of a pool (0, -0, ±1, subnormals, ±1e300, the largest DOUBLE, ±inf, NaN, NULL, extreme BIGINTs, empty and
   degenerate lists, ...), plus 300 random combinations of pool values. Each call is its own statement.
3. Aggregates run over edge-case datasets (empty, one value, constant, huge, tiny, with inf, NaN or NULL).
4. A second pass re-runs every successful scalar call as one row of a table, so the flat (non-constant) execution
   path is checked against the constant one.

`watchdog.sh` runs one shard of statements through the DuckDB CLI. When output stalls it records the running
statement as hung, kills the CLI and resumes after it; a crash is recorded the same way.

## What it reports

- hangs and crashes
- errors other than `Invalid Input Error`, or messages that do not start with the distribution name or that leak
  Boost or C++ internals
- NaN results, probabilities outside [0, 1], log-probabilities above 0, negative densities, hazards and variances
- NaN in aggregate results
- functions that never succeed
- differences between the constant and flat passes

A clean run prints only the count of `Invalid Input Error`s, which is expected: most random parameter combinations
are invalid.
