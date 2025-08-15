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
#include "distribution_traits.hpp"

namespace duckdb {

namespace {

// Macro for distribution functions that take no additional parameters (e.g., mean, variance)
// These functions only require the distribution parameters (mean, stddev)
#define NONE_FUNC(FuncName, Func)                                                                                      \
	inline void FuncName(DataChunk &args, ExpressionState &state, Vector &result) {                                    \
		DistributionCallBinaryNone<boost::math::normal_distribution<double>>(                                          \
		    args, state, result, [](Vector &result, const auto &dist) { return Func; });                               \
	}

// Macro for distribution functions that take one additional parameter (e.g., PDF, CDF, quantile)
// These functions require distribution parameters (mean, stddev) plus one evaluation point
#define UNARY_FUNC(FuncName, Func)                                                                                     \
	inline void FuncName(DataChunk &args, ExpressionState &state, Vector &result) {                                    \
		DistributionCallBinaryUnary<boost::math::normal_distribution<double>, double>(                                 \
		    args, state, result,                                                                                       \
		    [](const auto &dist, auto x) -> boost::math::normal_distribution<double>::value_type { return Func; });    \
	}

// UNARY_FUNC(NormalPdf, boost::math::pdf(dist, x));
// UNARY_FUNC(NormalLogPdf, boost::math::logpdf(dist, x));
UNARY_FUNC(NormalCdf, boost::math::cdf(dist, x));
UNARY_FUNC(NormalLogCdf, boost::math::logcdf(dist, x));
UNARY_FUNC(NormalCdfComplement, boost::math::cdf(boost::math::complement(dist, x)));
UNARY_FUNC(NormalLogCdfComplement, boost::math::logcdf(boost::math::complement(dist, x)));
UNARY_FUNC(NormalQuantile, boost::math::quantile(dist, x));
UNARY_FUNC(NormalQuantileComplement, boost::math::quantile(boost::math::complement(dist, x)));
UNARY_FUNC(NormalHazard, boost::math::hazard(dist, x));
UNARY_FUNC(NormalChf, boost::math::chf(dist, x));
NONE_FUNC(NormalMean, dist.mean());
NONE_FUNC(NormalStdDev, dist.standard_deviation());
NONE_FUNC(NormalVariance, boost::math::variance(dist));
NONE_FUNC(NormalMode, boost::math::mode(dist));
NONE_FUNC(NormalMedian, boost::math::median(dist));
NONE_FUNC(NormalSkewness, boost::math::skewness(dist));
NONE_FUNC(NormalKurtosis, boost::math::kurtosis(dist));
NONE_FUNC(NormalKurtosisExcess, boost::math::kurtosis_excess(dist));
NONE_FUNC(NormalRange, boost::math::range(dist));
NONE_FUNC(NormalSupport, boost::math::support(dist));

} // namespace

void LoadDistributionNormal(DatabaseInstance &instance) {
#define REGISTER RegisterFunction<boost::math::normal_distribution<double>>
	vector<std::pair<string, LogicalType>> param_names_quantile = {{"p", LogicalType::DOUBLE}};
	vector<std::pair<string, LogicalType>> param_names_unary = {{"x", LogicalType::DOUBLE}};

	auto make_unary = [](auto func) {
		return [func](DataChunk &args, ExpressionState &state, Vector &result) {
			DistributionCallBinaryUnary<boost::math::normal_distribution<double>, double>(args, state, result, func);
		};
	};

	// === SAMPLING FUNCTIONS ===
	// Used to sample from a distribution
	REGISTER(
	    instance, "sample", FunctionStability::VOLATILE, LogicalType::DOUBLE,
	    [](DataChunk &args, ExpressionState &state, Vector &result) {
		    DistributionSampleBinary<boost::random::normal_distribution<double>, double>(args, state, result);
	    },
	    "Generates random samples from the normal distribution with specified mean and standard deviation.",
	    "sample(0.0, 1.0)");

	REGISTER(
	    instance, "pdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> boost::math::normal_distribution<double>::value_type {
		    return boost::math::pdf(dist, x);
	    }),
	    "Computes the probability density function (PDF) of the normal distribution. Returns the probability density"
	    "at point x for a normal distribution with specified mean and standard deviation.",
	    "pdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "log_pdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> boost::math::normal_distribution<double>::value_type {
		         return boost::math::logpdf(dist, x);
	         }),
	         "Computes the natural logarithm of the probability density function (log-PDF) of the normal "
	         "distribution. Useful for numerical stability when dealing with very small probabilities.",
	         "log_pdf(0, 1.0, 0.5)", param_names_unary);

	// === CUMULATIVE DISTRIBUTION FUNCTIONS ===
	REGISTER(instance, "cdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalCdf,
	         "Computes the cumulative distribution function (CDF) of the normal distribution. Returns the "
	         "probability that a random variable X is less than or equal to x.",
	         "cdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "cdf_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalCdfComplement,
	         "Computes the complementary cumulative distribution function (1 - CDF) of the normal "
	         "distribution. Returns the probability that X > x, equivalent to the survival function.",
	         "cdf_complement(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "log_cdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalLogCdf,
	         "Computes the natural logarithm of the cumulative distribution function (CDF) of the normal distribution. "
	         "Returns the logarithm of the probability that a random variable X is less than or equal to x.",
	         "log_cdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(
	    instance, "log_cdf_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalLogCdfComplement,
	    "Computes the natural logarithm of the complementary cumulative distribution function (1 - CDF) of the normal"
	    " distribution. Returns the logarithm of the probability that X > x, equivalent to the survival function.",
	    "log_cdf_complement(0, 1.0, 0.5)", param_names_unary);

	// === QUANTILE FUNCTIONS ===
	REGISTER(instance, "quantile", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalQuantile,
	         "Computes the quantile function (inverse CDF) of the normal distribution. Returns the value x "
	         "such that P(X ≤ x) = p, where p is the cumulative probability.",
	         "quantile(0, 1.0, 0.95)", param_names_quantile);

	REGISTER(instance, "quantile_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         NormalQuantileComplement,
	         "Computes the complementary quantile function of the normal distribution. Returns the value x "
	         "such that P(X > x) = p, useful for computing upper tail quantiles.",
	         "quantile_complement(0, 1.0, 0.95)", param_names_quantile);

	REGISTER(instance, "hazard", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalHazard,
	         "Computes the hazard function of the normal distribution.", "hazard(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "chf", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalChf,
	         "Computes the cumulative hazard function of the normal distribution.", "chf(0, 1.0, 0.5)",
	         param_names_unary);

	// === DISTRIBUTION PROPERTIES ===

	REGISTER(instance, "mean", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalMean,
	         "Returns the mean (μ) of the normal distribution, which is the first moment.", "mean(0.0, 1.0)");

	REGISTER(instance, "stddev", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalStdDev,
	         "Returns the standard deviation (σ) of the normal distribution.", "stddev(0.0, 1.0)");

	REGISTER(instance, "variance", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalVariance,
	         "Returns the variance (σ²) of the normal distribution.", "variance(0.0, 1.0)");

	REGISTER(instance, "mode", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalMode,
	         "Returns the mode (most likely value) of the normal distribution, which equals the mean.",
	         "mode(0.0, 1.0)");

	REGISTER(instance, "median", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalMedian,
	         "Returns the median (50th percentile) of the normal distribution, which equals the mean.",
	         "median(0.0, 1.0)");

	REGISTER(instance, "skewness", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalSkewness,
	         "Returns the skewness of the normal distribution, which is always 0.", "skewness(0.0, 1.0)");

	REGISTER(instance, "kurtosis", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalKurtosis,
	         "Returns the kurtosis of the normal distribution, which is always 3.", "kurtosis(0.0, 1.0)");

	REGISTER(instance, "kurtosis_excess", FunctionStability::CONSISTENT, LogicalType::DOUBLE, NormalKurtosisExcess,
	         "Returns the excess kurtosis of the normal distribution, which is always 0 (kurtosis - 3).",
	         "kurtosis_excess(0.0, 1.0)");

	REGISTER(instance, "range", FunctionStability::CONSISTENT, LogicalType::ARRAY(LogicalType::DOUBLE, 2), NormalRange,
	         "Returns the range of the normal distribution, which is (-∞, +∞).", "range(0.0, 1.0)");

	REGISTER(instance, "support", FunctionStability::CONSISTENT, LogicalType::ARRAY(LogicalType::DOUBLE, 2),
	         NormalSupport, "Returns the support of the normal distribution, which is (-∞, +∞).", "support(0.0, 1.0)");
}
} // end namespace duckdb
