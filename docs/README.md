<p align="center">
  <a href="https://query.farm">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://query.farm/media-kit/logo/wordmark-dark.svg">
      <img alt="Query.Farm" src="https://query.farm/media-kit/logo/wordmark-light.svg" height="64">
    </picture>
  </a>
</p>

# Stochastic Extension for DuckDB by [Query.Farm](https://query.farm)

[![DuckDB](https://img.shields.io/badge/DuckDB-community_extension-fdf1e0?logo=duckdb&logoColor=fff000)](https://duckdb.org/community_extensions/extensions/stochastic.html)
[![v1.5 build](https://github.com/Query-farm/stochastic/actions/workflows/MainDistributionPipeline.yml/badge.svg?branch=v1.5)](https://github.com/Query-farm/stochastic/actions/workflows/MainDistributionPipeline.yml?query=branch%3Av1.5)

The **`stochastic`** extension, developed by **[Query.Farm](https://query.farm)**, adds comprehensive statistical distribution functions to DuckDB, enabling advanced statistical analysis, probability calculations, and random sampling directly within SQL queries.

## Documentation

Full documentation, including installation, usage, the function reference, and cookbook examples, is available at:

**[https://query.farm/products/extensions/stochastic](https://query.farm/products/extensions/stochastic)**

## Installation

```sql
INSTALL stochastic FROM community;
LOAD stochastic;
```

## October 7, 2026 compatibility fixes (v1.5 source)

Integer-uniform analytics now describe the inclusive discrete support used by sampling. Mean, variance, standard deviation, and median return DOUBLE rather than truncated BIGINT values. Bounds keep their DOUBLE signature but must be finite BIGINT-valued integers with min <= max; equal bounds are supported. Quantiles use the exact binary DOUBLE probability without boundary snapping: for example, quantile(1,10,0.1) returns 2 because binary 0.1 is slightly above one tenth. Discrete hazard uses P(X=x)/P(X>=x), and standardized moments are undefined at zero variance. Pareto parameter labels now correctly read (scale, shape); positional behavior is unchanged. Community packages may lag these source fixes.
