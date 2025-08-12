#include "utils.hpp"
#include "rng_utils.hpp"

namespace duckdb {

namespace {
// Generate Log Normal distribution functions
DEFINE_DISTRIBUTION_FUNC(LogNormalPdfFunc, boost::math::lognormal_distribution<double>, boost::math::pdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(LogNormalLogPdfFunc, boost::math::lognormal_distribution<double>, boost::math::logpdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(LogNormalCdfFunc, boost::math::lognormal_distribution<double>, boost::math::cdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(LogNormalLogCdfFunc, boost::math::lognormal_distribution<double>, boost::math::logcdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(LogNormalQuantileFunc, boost::math::lognormal_distribution<double>,
                         boost::math::quantile(dist, x))

DEFINE_DISTRIBUTION_FUNC(LogNormalCdfComplementFunc, boost::math::lognormal_distribution<double>,
                         boost::math::cdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(LogNormalLogCdfComplementFunc, boost::math::lognormal_distribution<double>,
                         boost::math::logcdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(LogNormalQuantileComplementFunc, boost::math::lognormal_distribution<double>,
                         boost::math::quantile(boost::math::complement(dist, x)))

// Statistical moment functions for log normal distribution
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogNormalMeanFunc, boost::math::lognormal_distribution<double>,
                                   boost::math::mean(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogNormalStdDevFunc, boost::math::lognormal_distribution<double>,
                                   boost::math::standard_deviation(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogNormalVarianceFunc, boost::math::lognormal_distribution<double>,
                                   boost::math::variance(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogNormalModeFunc, boost::math::lognormal_distribution<double>,
                                   boost::math::mode(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogNormalMedianFunc, boost::math::lognormal_distribution<double>,
                                   boost::math::median(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogNormalSkewnessFunc, boost::math::lognormal_distribution<double>,
                                   boost::math::skewness(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogNormalKurtosisFunc, boost::math::lognormal_distribution<double>,
                                   boost::math::kurtosis(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogNormalKurtosisExcessFunc, boost::math::lognormal_distribution<double>,
                                   boost::math::kurtosis_excess(dist))

// Specialized Log Normal sampling function
inline void LogNormalSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	GenericTwoParamSampleFunc<boost::random::lognormal_distribution<double>, double>(args, state, result);
}
} // namespace

void LoadDistributionLogNormal(DatabaseInstance &instance) {
	// Log Normal distribution parameter types
	const duckdb::vector<LogicalType> lognormal_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE,
	                                                           LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> lognormal_sample_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> lognormal_statistic_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};

	// Register Log Normal distribution functions
	REGISTER_LOGNORMAL_FUNC(
	    instance, "lognormal_pdf", LogNormalPdfFunc,
	    "Computes the probability density function (PDF) of the log normal distribution. Returns the "
	    "probability density at point x for a log normal distribution with specified mean and standard deviation "
	    "of the underlying normal distribution.",
	    "lognormal_pdf(0.0, 1.0, 2.0)", "x");

	REGISTER_LOGNORMAL_FUNC(
	    instance, "lognormal_cdf", LogNormalCdfFunc,
	    "Computes the cumulative distribution function (CDF) of the log normal distribution. Returns the probability "
	    "that X ≤ x for a log normal distribution with specified mean and standard deviation of the underlying normal "
	    "distribution.",
	    "lognormal_cdf(0.0, 1.0, 2.0)", "x");

	REGISTER_LOGNORMAL_FUNC(instance, "lognormal_quantile", LogNormalQuantileFunc,
	                        "Computes the quantile function (inverse CDF) of the log normal distribution. Returns the "
	                        "value x such that P(X ≤ x) = p for a log normal distribution.",
	                        "lognormal_quantile(0.0, 1.0, 0.5)", "p");

	// Log functions
	REGISTER_LOGNORMAL_FUNC(
	    instance, "lognormal_log_pdf", LogNormalLogPdfFunc,
	    "Computes the natural logarithm of the probability density function (log-PDF) of the log normal distribution. "
	    "Useful for numerical stability when dealing with very small probabilities.",
	    "lognormal_log_pdf(0.0, 1.0, 2.0)", "x");

	REGISTER_LOGNORMAL_FUNC(instance, "lognormal_log_cdf", LogNormalLogCdfFunc,
	                        "Computes the natural logarithm of the cumulative distribution function (log-CDF) of the "
	                        "log normal distribution. Useful for numerical stability in extreme tail calculations.",
	                        "lognormal_log_cdf(0.0, 1.0, 2.0)", "x");

	// Complement functions
	REGISTER_LOGNORMAL_FUNC(instance, "lognormal_cdf_complement", LogNormalCdfComplementFunc,
	                        "Computes the complementary cumulative distribution function (1 - CDF) of the log normal "
	                        "distribution. Returns the probability that X > x, equivalent to the survival function.",
	                        "lognormal_cdf_complement(0.0, 1.0, 2.0)", "x");

	REGISTER_LOGNORMAL_FUNC(
	    instance, "lognormal_log_cdf_complement", LogNormalLogCdfComplementFunc,
	    "Computes the natural logarithm of the complementary cumulative distribution function of the log normal "
	    "distribution. Useful for numerical stability in tail probability calculations.",
	    "lognormal_log_cdf_complement(0.0, 1.0, 2.0)", "x");

	REGISTER_LOGNORMAL_FUNC(instance, "lognormal_quantile_complement", LogNormalQuantileComplementFunc,
	                        "Computes the complementary quantile function of the log normal distribution. Returns the "
	                        "value x such that P(X > x) = p, useful for computing upper tail quantiles.",
	                        "lognormal_quantile_complement(0.0, 1.0, 0.3)", "p");

	// Log Normal sampling function
	RegisterSamplingFunction(
	    instance, "lognormal_sample", LogNormalSampleFunc,
	    "Generates random samples from the log normal distribution with specified mean and standard deviation "
	    "of the underlying normal distribution. Returns random values following a log normal distribution.",
	    "lognormal_sample(0.0, 1.0)", {"mean", "stddev"}, lognormal_sample_param_types);

	// Statistical moment functions
	REGISTER_LOGNORMAL_STATISTIC_FUNC(instance, "lognormal_mean", LogNormalMeanFunc,
	                                  "Returns the mean of the log normal distribution.", "lognormal_mean(0.0, 1.0)");

	REGISTER_LOGNORMAL_STATISTIC_FUNC(instance, "lognormal_stddev", LogNormalStdDevFunc,
	                                  "Returns the standard deviation of the log normal distribution.",
	                                  "lognormal_stddev(0.0, 1.0)");

	REGISTER_LOGNORMAL_STATISTIC_FUNC(instance, "lognormal_variance", LogNormalVarianceFunc,
	                                  "Returns the variance of the log normal distribution.",
	                                  "lognormal_variance(0.0, 1.0)");

	REGISTER_LOGNORMAL_STATISTIC_FUNC(instance, "lognormal_mode", LogNormalModeFunc,
	                                  "Returns the mode (most likely value) of the log normal distribution.",
	                                  "lognormal_mode(0.0, 1.0)");

	REGISTER_LOGNORMAL_STATISTIC_FUNC(instance, "lognormal_median", LogNormalMedianFunc,
	                                  "Returns the median (50th percentile) of the log normal distribution.",
	                                  "lognormal_median(0.0, 1.0)");

	REGISTER_LOGNORMAL_STATISTIC_FUNC(instance, "lognormal_skewness", LogNormalSkewnessFunc,
	                                  "Returns the skewness of the log normal distribution.",
	                                  "lognormal_skewness(0.0, 1.0)");

	REGISTER_LOGNORMAL_STATISTIC_FUNC(instance, "lognormal_kurtosis", LogNormalKurtosisFunc,
	                                  "Returns the kurtosis of the log normal distribution.",
	                                  "lognormal_kurtosis(0.0, 1.0)");

	REGISTER_LOGNORMAL_STATISTIC_FUNC(instance, "lognormal_kurtosis_excess", LogNormalKurtosisExcessFunc,
	                                  "Returns the excess kurtosis of the log normal distribution (kurtosis - 3).",
	                                  "lognormal_kurtosis_excess(0.0, 1.0)");
}

} // namespace duckdb
