# Stochastic Extension for DuckDB by [Query.Farm](https://query.farm)

The **`stochastic`** developed by **[Query.Farm](https://query.farm)** extension adds comprehensive statistical distribution functions to DuckDB, enabling advanced statistical analysis, probability calculations, and random sampling directly within SQL queries.


## Installation

**`stochastic` is a [DuckDB Community Extension](https://github.com/duckdb/community-extensions).**

You can install and use it in DuckDB SQL:

```sql
INSTALL stochastic FROM community;
LOAD stochastic;
```

## What are statistical distributions?

Statistical distributions are mathematical functions that describe the probability of different outcomes in a dataset. They are fundamental to statistics, data science, machine learning, and scientific computing. This extension provides functions to:

- Calculate probability density and mass functions (PDF/PMF)
- Compute cumulative distribution functions (CDF)
- Generate quantiles (inverse CDF)
- Sample random values from distributions
- Access distribution properties (mean, variance, etc.)

## Available Distributions

The extension supports a comprehensive set of probability distributions:

### Continuous Distributions
- **Beta** - `dist_beta_*` functions
- **Cauchy** - `dist_cauchy_*` functions
- **Chi-squared** - `dist_chi_squared_*` functions
- **Exponential** - `dist_exponential_*` functions
- **Extreme Value** - `dist_extreme_value_*` functions
- **Fisher F** - `dist_fisher_f_*` functions
- **Gamma** - `dist_gamma_*` functions
- **Log-normal** - `dist_lognormal_*` functions
- **Logistic** - `dist_logistic_*` functions
- **Normal (Gaussian)** - `dist_normal_*` functions
- **Pareto** - `dist_pareto_*` functions
- **Rayleigh** - `dist_rayleigh_*` functions
- **Student's t** - `dist_students_t_*` functions
- **Uniform (Real)** - `dist_uniform_real_*` functions
- **Weibull** - `dist_weibull_*` functions

### Discrete Distributions
- **Bernoulli** - `dist_bernoulli_*` functions
- **Binomial** - `dist_binomial_*` functions
- **Negative Binomial** - `dist_negative_binomial_*` functions
- **Poisson** - `dist_poisson_*` functions
- **Uniform (Integer)** - `dist_uniform_int_*` functions

## Function Categories

Each distribution provides the following function types:

### Sampling Functions
- `dist_{distribution}_sample(params...)` - Generate random samples

### Density/Mass Functions
- `dist_{distribution}_pdf(params..., x)` - Probability density function
- `dist_{distribution}_log_pdf(params..., x)` - Log probability density function

### Cumulative Functions
- `dist_{distribution}_cdf(params..., x)` - Cumulative distribution function
- `dist_{distribution}_log_cdf(params..., x)` - Log cumulative distribution function
- `dist_{distribution}_cdf_complement(params..., x)` - Survival function (1 - CDF)
- `dist_{distribution}_log_cdf_complement(params..., x)` - Log survival function

### Quantile Functions
- `dist_{distribution}_quantile(params..., p)` - Quantile function (inverse CDF)
- `dist_{distribution}_quantile_complement(params..., p)` - Complementary quantile function

### Hazard Functions
- `dist_{distribution}_hazard(params..., x)` - Hazard function
- `dist_{distribution}_chf(params..., x)` - Cumulative hazard function

### Distribution Properties
- `dist_{distribution}_kurtosis_excess(params...)` - Excess kurtosis
- `dist_{distribution}_kurtosis(params...)` - Kurtosis
- `dist_{distribution}_mean(params...)` - Expected value
- `dist_{distribution}_median(params...)` - Median (50th percentile)
- `dist_{distribution}_mode(params...)` - Mode (most likely value)
- `dist_{distribution}_range(params...)` - Support range
- `dist_{distribution}_skewness(params...)` - Skewness
- `dist_{distribution}_stddev(params...)` - Standard deviation
- `dist_{distribution}_support(params...)` - Distribution support
- `dist_{distribution}_variance(params...)` - Variance

## Distribution Parameters

Below are the parameters for each supported distribution. Use these as arguments for sampling, PDF, CDF, and other functions.

### Continuous Distributions

#### Beta
| Parameter | Description |
|-----------|-------------|
| `alpha`   | Shape parameter α (> 0) |
| `beta`    | Shape parameter β (> 0) |

#### Cauchy
| Parameter | Description |
|-----------|-------------|
| `location`| Location parameter x₀ |
| `scale`   | Scale parameter γ (> 0) |

#### Chi-squared
| Parameter | Description |
|-----------|-------------|
| `df`      | Degrees of freedom (> 0) |

#### Exponential
| Parameter | Description |
|-----------|-------------|
| `rate`    | Rate parameter λ (> 0) |

#### Extreme Value
| Parameter | Description |
|-----------|-------------|
| `location`| Location parameter |
| `scale`   | Scale parameter (> 0) |

#### Fisher F
| Parameter | Description |
|-----------|-------------|
| `df1`     | Numerator degrees of freedom (> 0) |
| `df2`     | Denominator degrees of freedom (> 0) |

#### Gamma
| Parameter | Description |
|-----------|-------------|
| `shape`   | Shape parameter k (> 0) |
| `scale`   | Scale parameter θ (> 0) |

#### Log-normal
| Parameter | Description |
|-----------|-------------|
| `meanlog` | Mean of log values |
| `sdlog`   | Standard deviation of log values (> 0) |

#### Logistic
| Parameter | Description |
|-----------|-------------|
| `location`| Location parameter |
| `scale`   | Scale parameter (> 0) |

#### Normal (Gaussian)
| Parameter | Description |
|-----------|-------------|
| `mean`    | Mean μ |
| `stddev`  | Standard deviation σ (> 0) |

#### Pareto
| Parameter | Description |
|-----------|-------------|
| `scale`   | Scale parameter xₘ (> 0) |
| `shape`   | Shape parameter α (> 0) |

#### Rayleigh
| Parameter | Description |
|-----------|-------------|
| `scale`   | Scale parameter σ (> 0) |

#### Student's t
| Parameter | Description |
|-----------|-------------|
| `df`      | Degrees of freedom (> 0) |

#### Uniform (Real)
| Parameter | Description |
|-----------|-------------|
| `min`     | Lower bound |
| `max`     | Upper bound (must be > min) |

#### Weibull
| Parameter | Description |
|-----------|-------------|
| `shape`   | Shape parameter k (> 0) |
| `scale`   | Scale parameter λ (> 0) |

### Discrete Distributions

#### Bernoulli
| Parameter | Description |
|-----------|-------------|
| `p`       | Probability of success (0 ≤ p ≤ 1) |

#### Binomial
| Parameter | Description |
|-----------|-------------|
| `n`       | Number of trials (integer ≥ 0) |
| `p`       | Probability of success (0 ≤ p ≤ 1) |

#### Negative Binomial
| Parameter | Description |
|-----------|-------------|
| `r`       | Number of successes (integer > 0) |
| `p`       | Probability of success (0 ≤ p ≤ 1) |

#### Poisson
| Parameter | Description |
|-----------|-------------|
| `rate`    | Rate parameter λ (> 0) |

#### Uniform (Integer)
| Parameter | Description |
|-----------|-------------|
| `min`     | Lower bound (integer) |
| `max`     | Upper bound (integer, must be ≥ min) |

## Usage Examples

### Normal Distribution

```sql
-- Generate random samples from N(0, 1)
SELECT dist_normal_sample(0.0, 1.0) AS random_value;

-- Calculate PDF at x = 0.5 for N(0, 1)
SELECT dist_normal_pdf(0.0, 1.0, 0.5) AS density;

-- Calculate CDF (probability that X ≤ 1.96)
SELECT dist_normal_cdf(0.0, 1.0, 1.96) AS probability;

-- Find 95th percentile
SELECT dist_normal_quantile(0.0, 1.0, 0.95) AS percentile_95;

-- Get distribution properties
SELECT
    dist_normal_mean(0.0, 1.0) AS mean,
    dist_normal_variance(0.0, 1.0) AS variance,
    dist_normal_skewness(0.0, 1.0) AS skewness;
```

### Binomial Distribution

```sql
-- Probability mass function for 10 trials, p=0.3
SELECT dist_binomial_pdf(10, 0.3, 7) AS prob_exactly_7;

-- Cumulative probability (≤ 5 successes)
SELECT dist_binomial_cdf(10, 0.3, 5) AS prob_at_most_5;

-- Generate random binomial samples
SELECT dist_binomial_sample(10, 0.3) AS random_successes;
```

### Working with Data Tables

```sql
-- Generate synthetic dataset
CREATE TABLE synthetic_data AS
SELECT
    i,
    dist_normal_sample(100, 15) AS height_cm,
    dist_normal_sample(70, 10) AS weight_kg,
    dist_binomial_sample(1, 0.5) AS gender  -- 0 or 1
FROM range(1000) t(i);

-- Calculate z-scores
SELECT
    height_cm,
    (height_cm - dist_normal_mean(100, 15)) / dist_normal_stddev(100, 15) AS height_zscore
FROM synthetic_data;

-- Probability calculations
SELECT
    weight_kg,
    dist_normal_cdf(70, 10, weight_kg) AS percentile
FROM synthetic_data
LIMIT 10;
```

## Real-World Applications

### A/B Testing and Statistical Significance
**Common Task**: Determine if there's a statistically significant difference between conversion rates.
**Relevant Functions**: `dist_normal_cdf`, `dist_normal_cdf_complement`, `dist_normal_pdf`

### Financial Risk Assessment and VaR Calculation
**Common Task**: Calculate Value at Risk (VaR) for portfolio management.
**Relevant Functions**: `dist_normal_sample`, `dist_normal_quantile`, `dist_normal_cdf`

### Quality Control and Process Monitoring
**Common Task**: Monitor manufacturing processes and detect out-of-control conditions.
**Relevant Functions**: `dist_normal_sample`, `dist_normal_cdf`, `dist_normal_pdf`

### Predictive Analytics and Confidence Intervals
**Common Task**: Build prediction intervals for forecasting models.
**Relevant Functions**: `dist_normal_quantile`, `dist_normal_cdf`, `dist_normal_sample`

### Customer Analytics and CLV Modeling
**Common Task**: Model customer lifetime value with uncertainty quantification.
**Relevant Functions**: `dist_normal_sample`, `dist_exponential_sample`, `dist_normal_quantile`, `dist_normal_cdf`

### Anomaly Detection and Outlier Analysis
**Common Task**: Detect anomalies in time series data using statistical methods.
**Relevant Functions**: `dist_normal_pdf`, `dist_normal_cdf`, `dist_normal_cdf_complement`

### Monte Carlo Simulations
**Common Task**: Run Monte Carlo simulations for risk analysis, optimization, or modeling.
**Relevant Functions**: `dist_normal_sample`, `dist_uniform_real_sample`, `dist_gamma_sample`, `dist_beta_sample`

### Hypothesis Testing
**Common Task**: Perform statistical hypothesis tests (t-tests, chi-square tests, etc.).
**Relevant Functions**: `dist_students_t_cdf`, `dist_chi_squared_cdf`, `dist_normal_cdf`, `dist_fisher_f_cdf`

### Bayesian Analysis
**Common Task**: Implement Bayesian statistical models and posterior analysis.
**Relevant Functions**: `dist_beta_pdf`, `dist_gamma_pdf`, `dist_normal_pdf`, `dist_beta_sample`

### Survival Analysis
**Common Task**: Analyze time-to-event data in medical research or reliability engineering.
**Relevant Functions**: `dist_exponential_pdf`, `dist_weibull_pdf`, `dist_gamma_pdf`, `dist_exponential_cdf`

## Why Use DuckDB + Stochastic vs Python/R?

### ✅ **Advantages**
- **No Data Movement**: Analysis happens where your data lives
- **SQL Familiarity**: Use existing SQL skills instead of learning specialized libraries
- **Performance**: Columnar processing with vectorized statistical operations
- **Integration**: Works seamlessly with existing BI tools and SQL workflows
- **Real-time**: Analyze streaming data without export/import cycles

### 📊 **Performance Benefits**
Statistical operations are vectorized and optimized for DuckDB's columnar engine.

## Parameter Validation

All distribution functions include comprehensive parameter validation:

```sql
-- This will throw an error: standard deviation must be > 0
SELECT dist_normal_pdf(0.0, -1.0, 0.5);
-- Error: normal: Standard deviation must be > 0 was: -1.000000

-- This will throw an error: probability must be between 0 and 1
SELECT dist_binomial_pdf(10, 1.5, 5);
-- Error: binomial: Probability must be between 0 and 1 was: 1.500000
```

## License

MIT Licensed
