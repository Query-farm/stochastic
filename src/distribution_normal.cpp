#include "utils.hpp"
#include "rng_utils.hpp"

namespace duckdb {

namespace {

// Generate all normal distribution functions using the macro
DEFINE_DISTRIBUTION_FUNC(NormalPdfFunc, boost::math::normal_distribution<double>, boost::math::pdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(NormalLogPdfFunc, boost::math::normal_distribution<double>, boost::math::logpdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(NormalCdfFunc, boost::math::normal_distribution<double>, boost::math::cdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(NormalLogCdfFunc, boost::math::normal_distribution<double>, boost::math::logcdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(NormalQuantileFunc, boost::math::normal_distribution<double>, boost::math::quantile(dist, x))

DEFINE_DISTRIBUTION_FUNC(NormalCdfComplementFunc, boost::math::normal_distribution<double>,
                         boost::math::cdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(NormalLogCdfComplementFunc, boost::math::normal_distribution<double>,
                         boost::math::logcdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(NormalQuantileComplementFunc, boost::math::normal_distribution<double>,
                         boost::math::quantile(boost::math::complement(dist, x)))

inline void NormalRandFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	GenericTwoParamSampleFunc<boost::random::normal_distribution<double>, double>(args, state, result);
}

} // namespace

void LoadDistributionNormal(DatabaseInstance &instance) {
	const duckdb::vector<LogicalType> normal_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE,
	                                                        LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> normal_sample_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};

	REGISTER_NORMAL_FUNC(
	    instance, "normal_pdf", NormalPdfFunc,
	    "Computes the probability density function (PDF) of the normal distribution. Returns the probability density "
	    "at point x for a normal distribution with specified mean and standard deviation.",
	    "normal_pdf(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(instance, "normal_log_pdf", NormalLogPdfFunc,
	                     "Computes the natural logarithm of the probability density function (log-PDF) of the normal "
	                     "distribution. Useful for numerical stability when dealing with very small probabilities.",
	                     "normal_log_pdf(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(instance, "normal_cdf", NormalCdfFunc,
	                     "Computes the cumulative distribution function (CDF) of the normal distribution. Returns the "
	                     "probability that a random variable X is less than or equal to x.",
	                     "normal_cdf(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(instance, "normal_cdf_complement", NormalCdfComplementFunc,
	                     "Computes the complementary cumulative distribution function (1 - CDF) of the normal "
	                     "distribution. Returns the probability that X > x, equivalent to the survival function.",
	                     "normal_cdf_complement(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(
	    instance, "normal_log_cdf", NormalLogCdfFunc,
	    "Computes the natural logarithm of the complementary cumulative distribution function (1 - CDF) of the normal "
	    "distribution. Returns the logarithm of the probability that X > x, equivalent to the survival function.",
	    "normal_log_cdf(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(
	    instance, "normal_log_cdf_complement", NormalLogCdfComplementFunc,
	    "Computes the natural logarithm of the complementary cumulative distribution function (1 - CDF) of the normal "
	    "distribution. Returns the logarithm of the probability that X > x, equivalent to the survival function.",
	    "normal_log_cdf_complement(0.0, 1.0, 0.5)", "x");

	REGISTER_NORMAL_FUNC(instance, "normal_quantile", NormalQuantileFunc,
	                     "Computes the quantile function (inverse CDF) of the normal distribution. Returns the value x "
	                     "such that P(X ≤ x) = p, where p is the cumulative probability.",
	                     "normal_quantile(0.0, 1.0, 0.5)", "p");

	REGISTER_NORMAL_FUNC(instance, "normal_quantile_complement", NormalQuantileComplementFunc,
	                     "Computes the complementary quantile function of the normal distribution. Returns the value x "
	                     "such that P(X > x) = p, useful for computing upper tail quantiles.",
	                     "normal_quantile_complement(0.0, 1.0, 0.5)", "p");

	// Sampling function
	RegisterSamplingFunction(
	    instance, "normal_sample", NormalRandFunc,
	    "Generates random samples from the normal distribution with specified mean and standard deviation.",
	    "normal_sample(0.0, 1.0)", {"mean", "stddev"}, normal_sample_param_types);
}

} // namespace duckdb
