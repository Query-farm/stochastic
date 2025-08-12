#include "utils.hpp"
#include "rng_utils.hpp"
#include <cmath>

namespace duckdb {

namespace {
// Generate Logistic distribution functions
DEFINE_DISTRIBUTION_FUNC(LogisticPdfFunc, boost::math::logistic_distribution<double>, boost::math::pdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(LogisticLogPdfFunc, boost::math::logistic_distribution<double>, boost::math::logpdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(LogisticCdfFunc, boost::math::logistic_distribution<double>, boost::math::cdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(LogisticLogCdfFunc, boost::math::logistic_distribution<double>, boost::math::logcdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(LogisticQuantileFunc, boost::math::logistic_distribution<double>,
                         boost::math::quantile(dist, x))

DEFINE_DISTRIBUTION_FUNC(LogisticCdfComplementFunc, boost::math::logistic_distribution<double>,
                         boost::math::cdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(LogisticLogCdfComplementFunc, boost::math::logistic_distribution<double>,
                         boost::math::logcdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(LogisticQuantileComplementFunc, boost::math::logistic_distribution<double>,
                         boost::math::quantile(boost::math::complement(dist, x)))

// Statistical moment functions for logistic distribution
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogisticMeanFunc, boost::math::logistic_distribution<double>,
                                   boost::math::mean(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogisticStdDevFunc, boost::math::logistic_distribution<double>,
                                   boost::math::standard_deviation(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogisticVarianceFunc, boost::math::logistic_distribution<double>,
                                   boost::math::variance(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogisticModeFunc, boost::math::logistic_distribution<double>,
                                   boost::math::mode(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogisticMedianFunc, boost::math::logistic_distribution<double>,
                                   boost::math::median(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogisticSkewnessFunc, boost::math::logistic_distribution<double>,
                                   boost::math::skewness(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogisticKurtosisFunc, boost::math::logistic_distribution<double>,
                                   boost::math::kurtosis(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(LogisticKurtosisExcessFunc, boost::math::logistic_distribution<double>,
                                   boost::math::kurtosis_excess(dist))

} // namespace
void LoadDistributionLogistic(DatabaseInstance &instance) {
	// Logistic distribution parameter types
	const duckdb::vector<LogicalType> logistic_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE,
	                                                          LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> logistic_sample_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> logistic_statistic_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};

	// Register Logistic distribution functions
	REGISTER_LOGISTIC_FUNC(
	    instance, "logistic_pdf", LogisticPdfFunc,
	    "Computes the probability density function (PDF) of the logistic distribution. Returns the "
	    "probability density at point x for a logistic distribution with specified location and scale parameters.",
	    "logistic_pdf(0.0, 1.0, 0.5)", "x");

	REGISTER_LOGISTIC_FUNC(
	    instance, "logistic_cdf", LogisticCdfFunc,
	    "Computes the cumulative distribution function (CDF) of the logistic distribution. Returns the probability "
	    "that X ≤ x for a logistic distribution with specified location and scale parameters.",
	    "logistic_cdf(0.0, 1.0, 0.5)", "x");

	REGISTER_LOGISTIC_FUNC(instance, "logistic_quantile", LogisticQuantileFunc,
	                       "Computes the quantile function (inverse CDF) of the logistic distribution. Returns the "
	                       "value x such that P(X ≤ x) = p for a logistic distribution.",
	                       "logistic_quantile(0.0, 1.0, 0.5)", "p");

	// Log functions
	REGISTER_LOGISTIC_FUNC(
	    instance, "logistic_log_pdf", LogisticLogPdfFunc,
	    "Computes the natural logarithm of the probability density function (log-PDF) of the logistic distribution. "
	    "Useful for numerical stability when dealing with very small probabilities.",
	    "logistic_log_pdf(0.0, 1.0, 0.5)", "x");

	REGISTER_LOGISTIC_FUNC(instance, "logistic_log_cdf", LogisticLogCdfFunc,
	                       "Computes the natural logarithm of the cumulative distribution function (log-CDF) of the "
	                       "logistic distribution. Useful for numerical stability in extreme tail calculations.",
	                       "logistic_log_cdf(0.0, 1.0, 0.5)", "x");

	// Complement functions
	REGISTER_LOGISTIC_FUNC(instance, "logistic_cdf_complement", LogisticCdfComplementFunc,
	                       "Computes the complementary cumulative distribution function (1 - CDF) of the logistic "
	                       "distribution. Returns the probability that X > x, equivalent to the survival function.",
	                       "logistic_cdf_complement(0.0, 1.0, 0.5)", "x");

	REGISTER_LOGISTIC_FUNC(
	    instance, "logistic_log_cdf_complement", LogisticLogCdfComplementFunc,
	    "Computes the natural logarithm of the complementary cumulative distribution function of the logistic "
	    "distribution. Useful for numerical stability in tail probability calculations.",
	    "logistic_log_cdf_complement(0.0, 1.0, 0.5)", "x");

	REGISTER_LOGISTIC_FUNC(instance, "logistic_quantile_complement", LogisticQuantileComplementFunc,
	                       "Computes the complementary quantile function of the logistic distribution. Returns the "
	                       "value x such that P(X > x) = p, useful for computing upper tail quantiles.",
	                       "logistic_quantile_complement(0.0, 1.0, 0.3)", "p");

	// Logistic sampling function
	// RegisterSamplingFunction(
	//     instance, "logistic_sample", LogisticSampleFunc,
	//     "Generates random samples from the logistic distribution with specified location and scale parameters. "
	//     "Returns random values following a logistic distribution.",
	//     "logistic_sample(0.0, 1.0)", {"location", "scale"}, logistic_sample_param_types);

	// Statistical moment functions
	REGISTER_LOGISTIC_STATISTIC_FUNC(instance, "logistic_mean", LogisticMeanFunc,
	                                 "Returns the mean (location parameter) of the logistic distribution.",
	                                 "logistic_mean(0.0, 1.0)");

	REGISTER_LOGISTIC_STATISTIC_FUNC(instance, "logistic_stddev", LogisticStdDevFunc,
	                                 "Returns the standard deviation of the logistic distribution.",
	                                 "logistic_stddev(0.0, 1.0)");

	REGISTER_LOGISTIC_STATISTIC_FUNC(instance, "logistic_variance", LogisticVarianceFunc,
	                                 "Returns the variance of the logistic distribution.",
	                                 "logistic_variance(0.0, 1.0)");

	REGISTER_LOGISTIC_STATISTIC_FUNC(instance, "logistic_mode", LogisticModeFunc,
	                                 "Returns the mode (location parameter) of the logistic distribution.",
	                                 "logistic_mode(0.0, 1.0)");

	REGISTER_LOGISTIC_STATISTIC_FUNC(instance, "logistic_median", LogisticMedianFunc,
	                                 "Returns the median (location parameter) of the logistic distribution.",
	                                 "logistic_median(0.0, 1.0)");

	REGISTER_LOGISTIC_STATISTIC_FUNC(instance, "logistic_skewness", LogisticSkewnessFunc,
	                                 "Returns the skewness (0) of the logistic distribution.",
	                                 "logistic_skewness(0.0, 1.0)");

	REGISTER_LOGISTIC_STATISTIC_FUNC(instance, "logistic_kurtosis", LogisticKurtosisFunc,
	                                 "Returns the kurtosis (4.2) of the logistic distribution.",
	                                 "logistic_kurtosis(0.0, 1.0)");

	REGISTER_LOGISTIC_STATISTIC_FUNC(instance, "logistic_kurtosis_excess", LogisticKurtosisExcessFunc,
	                                 "Returns the excess kurtosis (1.2) of the logistic distribution (kurtosis - 3).",
	                                 "logistic_kurtosis_excess(0.0, 1.0)");
}

} // namespace duckdb
