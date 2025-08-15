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

#define DISTRIBUTION boost::math::normal_distribution<double>
#define REGISTER     RegisterFunction<DISTRIBUTION>

void LoadDistributionNormal(DatabaseInstance &instance) {
	vector<std::pair<string, LogicalType>> param_names_quantile = {{"p", LogicalType::DOUBLE}};
	vector<std::pair<string, LogicalType>> param_names_unary = {{"x", LogicalType::DOUBLE}};

	auto make_unary = [](auto func) {
		return [func](DataChunk &args, ExpressionState &state, Vector &result) {
			DistributionCallBinaryUnary<DISTRIBUTION, double>(args, state, result, func);
		};
	};

	auto make_none = [](auto func) {
		return [func](DataChunk &args, ExpressionState &state, Vector &result) {
			DistributionCallBinaryNone<DISTRIBUTION>(args, state, result,
			                                         [func](Vector &result, const auto &dist) { return func(dist); });
		};
	};

	REGISTER(
	    instance, "sample", FunctionStability::VOLATILE, LogicalType::DOUBLE,
	    [](DataChunk &args, ExpressionState &state, Vector &result) {
		    DistributionSampleBinary<boost::random::normal_distribution<double>, double>(args, state, result);
	    },
	    "Generates random samples from the normal distribution with specified mean and standard deviation.",
	    "sample(0.0, 1.0)");

	REGISTER(
	    instance, "pdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::pdf(dist, x); }),
	    "Computes the probability density function (PDF) of the normal distribution. Returns the probability density"
	    "at point x for a normal distribution with specified mean and standard deviation.",
	    "pdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(
	    instance, "log_pdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::logpdf(dist, x); }),
	    "Computes the natural logarithm of the probability density function (log-PDF) of the normal "
	    "distribution. Useful for numerical stability when dealing with very small probabilities.",
	    "log_pdf(0, 1.0, 0.5)", param_names_unary);

	// === CUMULATIVE DISTRIBUTION FUNCTIONS ===
	REGISTER(instance, "cdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::cdf(dist, x); }),
	         "Computes the cumulative distribution function (CDF) of the normal distribution. Returns the "
	         "probability that a random variable X is less than or equal to x.",
	         "cdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "cdf_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type {
		         return boost::math::cdf(boost::math::complement(dist, x));
	         }),
	         "Computes the complementary cumulative distribution function (1 - CDF) of the normal "
	         "distribution. Returns the probability that X > x, equivalent to the survival function.",
	         "cdf_complement(0, 1.0, 0.5)", param_names_unary);

	REGISTER(
	    instance, "log_cdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::logcdf(dist, x); }),
	    "Computes the natural logarithm of the cumulative distribution function (CDF) of the normal distribution. "
	    "Returns the logarithm of the probability that a random variable X is less than or equal to x.",
	    "log_cdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(
	    instance, "log_cdf_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type {
		    return boost::math::logcdf(boost::math::complement(dist, x));
	    }),
	    "Computes the natural logarithm of the complementary cumulative distribution function (1 - CDF) of the normal"
	    " distribution. Returns the logarithm of the probability that X > x, equivalent to the survival function.",
	    "log_cdf_complement(0, 1.0, 0.5)", param_names_unary);

	// === QUANTILE FUNCTIONS ===
	REGISTER(
	    instance, "quantile", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto p) -> DISTRIBUTION::value_type { return boost::math::quantile(dist, p); }),
	    "Computes the quantile function (inverse CDF) of the normal distribution. Returns the value x "
	    "such that P(X ≤ x) = p, where p is the cumulative probability.",
	    "quantile(0, 1.0, 0.95)", param_names_quantile);

	REGISTER(instance, "quantile_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto p) -> DISTRIBUTION::value_type {
		         return boost::math::quantile(boost::math::complement(dist, p));
	         }),
	         "Computes the complementary quantile function of the normal distribution. Returns the value x "
	         "such that P(X > x) = p, useful for computing upper tail quantiles.",
	         "quantile_complement(0, 1.0, 0.95)", param_names_quantile);

	REGISTER(
	    instance, "hazard", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::hazard(dist, x); }),
	    "Computes the hazard function of the normal distribution.", "hazard(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "chf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::chf(dist, x); }),
	         "Computes the cumulative hazard function of the normal distribution.", "chf(0, 1.0, 0.5)",
	         param_names_unary);

	// === DISTRIBUTION PROPERTIES ===

	REGISTER(instance, "mean", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return dist.mean(); }),
	         "Returns the mean (μ) of the normal distribution, which is the first moment.", "mean(0.0, 1.0)");

	REGISTER(instance, "stddev", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return dist.standard_deviation(); }),
	         "Returns the standard deviation (σ) of the normal distribution.", "stddev(0.0, 1.0)");

	REGISTER(instance, "variance", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::variance(dist); }),
	         "Returns the variance (σ²) of the normal distribution.", "variance(0.0, 1.0)");

	REGISTER(instance, "mode", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::mode(dist); }),
	         "Returns the mode (most likely value) of the normal distribution, which equals the mean.",
	         "mode(0.0, 1.0)");

	REGISTER(instance, "median", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::median(dist); }),
	         "Returns the median (50th percentile) of the normal distribution, which equals the mean.",
	         "median(0.0, 1.0)");

	REGISTER(instance, "skewness", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::skewness(dist); }),
	         "Returns the skewness of the normal distribution, which is always 0.", "skewness(0.0, 1.0)");

	REGISTER(instance, "kurtosis", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::kurtosis(dist); }),
	         "Returns the kurtosis of the normal distribution, which is always 3.", "kurtosis(0.0, 1.0)");

	REGISTER(instance, "kurtosis_excess", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::kurtosis_excess(dist); }),
	         "Returns the excess kurtosis of the normal distribution, which is always 0 (kurtosis - 3).",
	         "kurtosis_excess(0.0, 1.0)");

	REGISTER(instance, "range", FunctionStability::CONSISTENT, LogicalType::ARRAY(LogicalType::DOUBLE, 2),
	         make_none([](const auto &dist) { return boost::math::range(dist); }),
	         "Returns the range of the normal distribution, which is (-∞, +∞).", "range(0.0, 1.0)");

	REGISTER(instance, "support", FunctionStability::CONSISTENT, LogicalType::ARRAY(LogicalType::DOUBLE, 2),
	         make_none([](const auto &dist) { return boost::math::support(dist); }),
	         "Returns the support of the normal distribution, which is (-∞, +∞).", "support(0.0, 1.0)");
}
} // end namespace duckdb
