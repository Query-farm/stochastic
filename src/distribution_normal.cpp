/*
 * Normal Distribution Functions for DuckDB
 *
 * This file implements all normal distribution-related scalar functions including:
 * - Probability density function (PDF) and log-PDF
 * - Cumulative distribution function (CDF) and log-CDF
 * - Complementary CDF functions (survival functions)
 * - Quantile functions (inverse CDF)
 * - Distribution properties (mean, variance, skewness, etc.)
 * - Random sampling
 *
 * The normal distribution is parameterized by:
 * - mean (μ): location parameter, can be any real number
 * - stddev (σ): scale parameter, must be positive
 */

#include "utils.hpp"
#include "rng_utils.hpp"

namespace duckdb {

namespace {

// Macro for distribution functions that take no additional parameters (e.g., mean, variance)
// These functions only require the distribution parameters (mean, stddev)
#define NONE_FUNC(FuncName, BoostFunc)                                                                                 \
	inline void FuncName(DataChunk &args, ExpressionState &state, Vector &result) {                                    \
		DistributionCallBinaryNone<boost::math::normal_distribution<double>, double, double, double>(                  \
		    args, state, result, [](const auto &dist) { return BoostFunc; });                                          \
	}

// Macro for distribution functions that take one additional parameter (e.g., PDF, CDF, quantile)
// These functions require distribution parameters (mean, stddev) plus one evaluation point
#define UNARY_FUNC(FuncName, BoostFunc)                                                                                \
	inline void FuncName(DataChunk &args, ExpressionState &state, Vector &result) {                                    \
		DistributionCallBinaryUnary<boost::math::normal_distribution<double>, double, double, double, double>(         \
		    args, state, result, [](const auto &dist, auto x) { return BoostFunc; });                                  \
	}

UNARY_FUNC(NormalPdf, boost::math::pdf(dist, x));
UNARY_FUNC(NormalLogPdf, boost::math::logpdf(dist, x));
UNARY_FUNC(NormalCdf, boost::math::cdf(dist, x));
UNARY_FUNC(NormalLogCdf, boost::math::logcdf(dist, x));
UNARY_FUNC(NormalCdfComplement, boost::math::cdf(boost::math::complement(dist, x)));
UNARY_FUNC(NormalLogCdfComplement, boost::math::logcdf(boost::math::complement(dist, x)));
UNARY_FUNC(NormalQuantile, boost::math::quantile(dist, x));
UNARY_FUNC(NormalQuantileComplement, boost::math::quantile(boost::math::complement(dist, x)));

NONE_FUNC(NormalMean, dist.mean());
NONE_FUNC(NormalStdDev, dist.standard_deviation());
NONE_FUNC(NormalVariance, boost::math::variance(dist));
NONE_FUNC(NormalMode, boost::math::mode(dist));
NONE_FUNC(NormalMedian, boost::math::median(dist));
NONE_FUNC(NormalSkewness, boost::math::skewness(dist));
NONE_FUNC(NormalKurtosis, boost::math::kurtosis(dist));
NONE_FUNC(NormalKurtosisExcess, boost::math::kurtosis_excess(dist));

// Random sampling function for normal distribution
// Uses thread-local RNG for reproducible results across parallel execution
inline void NormalRand(DataChunk &args, ExpressionState &state, Vector &result) {
	DistributionSampleBinary<boost::random::normal_distribution<double>, double, double, double>(args, state, result);
}

} // namespace

// Registers all normal distribution-related functions (PDF, CDF, quantile, statistics, sampling) with the database
// instance. The normal distribution is parameterized by mean (μ) and standard deviation (σ > 0).
void LoadDistributionNormal(DatabaseInstance &instance) {
	// Parameter types for functions that take an additional evaluation point (x or p)
	const duckdb::vector<LogicalType> param_types_unary = {LogicalType::DOUBLE, LogicalType::DOUBLE,
	                                                       LogicalType::DOUBLE};
	// Parameter types for functions that only need distribution parameters
	const duckdb::vector<LogicalType> param_types_none = {LogicalType::DOUBLE, LogicalType::DOUBLE};
	// Parameter types for quantile functions (probability p instead of x)
	const duckdb::vector<LogicalType> param_types_quantile = {LogicalType::DOUBLE, LogicalType::DOUBLE,
	                                                          LogicalType::DOUBLE};

	const duckdb::vector<string> param_names_none = {"mean", "stddev"};
	const duckdb::vector<string> param_names_unary = {"mean", "stddev", "x"};
	const duckdb::vector<string> param_names_quantile = {"mean", "stddev", "p"};

	// === SAMPLING FUNCTIONS ===
	// Used to sample from a distribution
	RegisterFunction(
	    instance, "normal_sample", FunctionStability::VOLATILE, param_types_none, LogicalType::DOUBLE, NormalRand,
	    "Generates random samples from the normal distribution with specified mean and standard deviation.",
	    "normal_sample(0.0, 1.0)", param_names_none);

	// === PROBABILITY DENSITY FUNCTIONS ===
	RegisterFunction(
	    instance, "normal_pdf", FunctionStability::CONSISTENT, param_types_unary, LogicalType::DOUBLE, NormalPdf,
	    "Computes the probability density function (PDF) of the normal distribution. Returns the probability density "
	    "at point x for a normal distribution with specified mean and standard deviation.",
	    "normal_pdf(0, 1.0, 0.5)", param_names_unary);

	RegisterFunction(instance, "normal_log_pdf", FunctionStability::CONSISTENT, param_types_unary, LogicalType::DOUBLE,
	                 NormalLogPdf,
	                 "Computes the natural logarithm of the probability density function (log-PDF) of the normal "
	                 "distribution. Useful for numerical stability when dealing with very small probabilities.",
	                 "normal_log_pdf(0, 1.0, 0.5)", param_names_unary);

	// === CUMULATIVE DISTRIBUTION FUNCTIONS ===
	RegisterFunction(instance, "normal_cdf", FunctionStability::CONSISTENT, param_types_unary, LogicalType::DOUBLE,
	                 NormalCdf,
	                 "Computes the cumulative distribution function (CDF) of the normal distribution. Returns the "
	                 "probability that a random variable X is less than or equal to x.",
	                 "normal_cdf(0, 1.0, 0.5)", param_names_unary);

	RegisterFunction(instance, "normal_cdf_complement", FunctionStability::CONSISTENT, param_types_unary,
	                 LogicalType::DOUBLE, NormalCdfComplement,
	                 "Computes the complementary cumulative distribution function (1 - CDF) of the normal "
	                 "distribution. Returns the probability that X > x, equivalent to the survival function.",
	                 "normal_cdf_complement(0, 1.0, 0.5)", param_names_unary);

	RegisterFunction(
	    instance, "normal_log_cdf", FunctionStability::CONSISTENT, param_types_unary, LogicalType::DOUBLE, NormalLogCdf,
	    "Computes the natural logarithm of the cumulative distribution function (CDF) of the normal distribution. "
	    "Returns the logarithm of the probability that a random variable X is less than or equal to x.",
	    "normal_log_cdf(0, 1.0, 0.5)", param_names_unary);

	RegisterFunction(
	    instance, "normal_log_cdf_complement", FunctionStability::CONSISTENT, param_types_unary, LogicalType::DOUBLE,
	    NormalLogCdfComplement,
	    "Computes the natural logarithm of the complementary cumulative distribution function (1 - CDF) of the normal "
	    "distribution. Returns the logarithm of the probability that X > x, equivalent to the survival function.",
	    "normal_log_cdf_complement(0, 1.0, 0.5)", param_names_unary);

	// === QUANTILE FUNCTIONS ===

	// === QUANTILE FUNCTIONS ===
	RegisterFunction(instance, "normal_quantile", FunctionStability::CONSISTENT, param_types_quantile,
	                 LogicalType::DOUBLE, NormalQuantile,
	                 "Computes the quantile function (inverse CDF) of the normal distribution. Returns the value x "
	                 "such that P(X ≤ x) = p, where p is the cumulative probability.",
	                 "normal_quantile(0, 1.0, 0.95)", param_names_quantile);

	RegisterFunction(instance, "normal_quantile_complement", FunctionStability::CONSISTENT, param_types_quantile,
	                 LogicalType::DOUBLE, NormalQuantileComplement,
	                 "Computes the complementary quantile function of the normal distribution. Returns the value x "
	                 "such that P(X > x) = p, useful for computing upper tail quantiles.",
	                 "normal_quantile_complement(0, 1.0, 0.95)", param_names_quantile);

	// === DISTRIBUTION PROPERTIES ===

	RegisterFunction(instance, "normal_mean", FunctionStability::CONSISTENT, param_types_none, LogicalType::DOUBLE,
	                 NormalMean, "Returns the mean (μ) of the normal distribution, which is the first moment.",
	                 "normal_mean(0.0, 1.0)", param_names_none);

	RegisterFunction(instance, "normal_stddev", FunctionStability::CONSISTENT, param_types_none, LogicalType::DOUBLE,
	                 NormalStdDev, "Returns the standard deviation (σ) of the normal distribution.",
	                 "normal_stddev(0.0, 1.0)", param_names_none);

	RegisterFunction(instance, "normal_variance", FunctionStability::CONSISTENT, param_types_none, LogicalType::DOUBLE,
	                 NormalVariance, "Returns the variance (σ²) of the normal distribution.",
	                 "normal_variance(0.0, 1.0)", param_names_none);

	RegisterFunction(instance, "normal_mode", FunctionStability::CONSISTENT, param_types_none, LogicalType::DOUBLE,
	                 NormalMode,
	                 "Returns the mode (most likely value) of the normal distribution, which equals the mean.",
	                 "normal_mode(0.0, 1.0)", param_names_none);

	RegisterFunction(instance, "normal_median", FunctionStability::CONSISTENT, param_types_none, LogicalType::DOUBLE,
	                 NormalMedian,
	                 "Returns the median (50th percentile) of the normal distribution, which equals the mean.",
	                 "normal_median(0.0, 1.0)", param_names_none);

	RegisterFunction(instance, "normal_skewness", FunctionStability::CONSISTENT, param_types_none, LogicalType::DOUBLE,
	                 NormalSkewness, "Returns the skewness of the normal distribution, which is always 0.",
	                 "normal_skewness(0.0, 1.0)", param_names_none);

	RegisterFunction(instance, "normal_kurtosis", FunctionStability::CONSISTENT, param_types_none, LogicalType::DOUBLE,
	                 NormalKurtosis, "Returns the kurtosis of the normal distribution, which is always 3.",
	                 "normal_kurtosis(0.0, 1.0)", param_names_none);

	RegisterFunction(instance, "normal_kurtosis_excess", FunctionStability::CONSISTENT, param_types_none,
	                 LogicalType::DOUBLE, NormalKurtosisExcess,
	                 "Returns the excess kurtosis of the normal distribution, which is always 0 (kurtosis - 3).",
	                 "normal_kurtosis_excess(0.0, 1.0)", param_names_none);
}
} // end namespace duckdb
