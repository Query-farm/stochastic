#include "utils.hpp"
#include "rng_utils.hpp"

namespace duckdb {

namespace {
// Generate Binomial distribution functions
DEFINE_DISTRIBUTION_FUNC(BinomialPdfFunc, boost::math::binomial_distribution<double>, boost::math::pdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(BinomialLogPdfFunc, boost::math::binomial_distribution<double>, boost::math::logpdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(BinomialCdfFunc, boost::math::binomial_distribution<double>, boost::math::cdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(BinomialLogCdfFunc, boost::math::binomial_distribution<double>, boost::math::logcdf(dist, x))
DEFINE_DISTRIBUTION_FUNC(BinomialQuantileFunc, boost::math::binomial_distribution<double>,
                         boost::math::quantile(dist, x))

DEFINE_DISTRIBUTION_FUNC(BinomialCdfComplementFunc, boost::math::binomial_distribution<double>,
                         boost::math::cdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(BinomialLogCdfComplementFunc, boost::math::binomial_distribution<double>,
                         boost::math::logcdf(boost::math::complement(dist, x)))
DEFINE_DISTRIBUTION_FUNC(BinomialQuantileComplementFunc, boost::math::binomial_distribution<double>,
                         boost::math::quantile(boost::math::complement(dist, x)))

// Statistical moment functions for binomial distribution
DEFINE_DISTRIBUTION_STATISTIC_FUNC(BinomialMeanFunc, boost::math::binomial_distribution<double>,
                                   boost::math::mean(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(BinomialStdDevFunc, boost::math::binomial_distribution<double>,
                                   boost::math::standard_deviation(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(BinomialVarianceFunc, boost::math::binomial_distribution<double>,
                                   boost::math::variance(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(BinomialModeFunc, boost::math::binomial_distribution<double>,
                                   boost::math::mode(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(BinomialSkewnessFunc, boost::math::binomial_distribution<double>,
                                   boost::math::skewness(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(BinomialKurtosisFunc, boost::math::binomial_distribution<double>,
                                   boost::math::kurtosis(dist))
DEFINE_DISTRIBUTION_STATISTIC_FUNC(BinomialKurtosisExcessFunc, boost::math::binomial_distribution<double>,
                                   boost::math::kurtosis_excess(dist))

// Specialized Binomial sampling function
inline void BinomialSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	GenericTwoParamSampleFunc<boost::random::binomial_distribution<int>, int>(args, state, result);
}
} // namespace

void LoadDistributionBinomial(DatabaseInstance &instance) {
	// Binomial distribution parameter types
	const duckdb::vector<LogicalType> binomial_param_types = {LogicalType::INTEGER, LogicalType::DOUBLE,
	                                                          LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> binomial_sample_param_types = {LogicalType::INTEGER, LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> binomial_statistic_param_types = {LogicalType::INTEGER, LogicalType::DOUBLE};

	// Register Binomial distribution functions
	REGISTER_BINOMIAL_FUNC(
	    instance, "binomial_pdf", BinomialPdfFunc,
	    "Computes the probability mass function (PMF) of the binomial distribution. Returns the "
	    "probability of getting exactly x successes in n independent Bernoulli trials with success probability p.",
	    "binomial_pdf(10, 0.3, 3)", "x");

	REGISTER_BINOMIAL_FUNC(
	    instance, "binomial_cdf", BinomialCdfFunc,
	    "Computes the cumulative distribution function (CDF) of the binomial distribution. Returns the probability "
	    "that X ≤ x for a binomial distribution with n trials and success probability p.",
	    "binomial_cdf(10, 0.3, 3)", "x");

	REGISTER_BINOMIAL_FUNC(instance, "binomial_quantile", BinomialQuantileFunc,
	                       "Computes the quantile function (inverse CDF) of the binomial distribution. Returns the "
	                       "smallest value x such that P(X ≤ x) ≥ p.",
	                       "binomial_quantile(10, 0.3, 0.5)", "p");

	// Log functions
	REGISTER_BINOMIAL_FUNC(
	    instance, "binomial_log_pdf", BinomialLogPdfFunc,
	    "Computes the natural logarithm of the probability mass function (log-PMF) of the binomial distribution. "
	    "Useful for numerical stability when dealing with very small probabilities.",
	    "binomial_log_pdf(10, 0.3, 3)", "x");

	REGISTER_BINOMIAL_FUNC(instance, "binomial_log_cdf", BinomialLogCdfFunc,
	                       "Computes the natural logarithm of the cumulative distribution function (log-CDF) of the "
	                       "binomial distribution. Useful for numerical stability in extreme tail calculations.",
	                       "binomial_log_cdf(10, 0.3, 3)", "x");

	// Complement functions
	REGISTER_BINOMIAL_FUNC(instance, "binomial_cdf_complement", BinomialCdfComplementFunc,
	                       "Computes the complementary cumulative distribution function (1 - CDF) of the binomial "
	                       "distribution. Returns the probability that X > x, equivalent to the survival function.",
	                       "binomial_cdf_complement(10, 0.3, 3)", "x");

	REGISTER_BINOMIAL_FUNC(
	    instance, "binomial_log_cdf_complement", BinomialLogCdfComplementFunc,
	    "Computes the natural logarithm of the complementary cumulative distribution function of the binomial "
	    "distribution. Useful for numerical stability in tail probability calculations.",
	    "binomial_log_cdf_complement(10, 0.3, 3)", "x");

	REGISTER_BINOMIAL_FUNC(instance, "binomial_quantile_complement", BinomialQuantileComplementFunc,
	                       "Computes the complementary quantile function of the binomial distribution. Returns the "
	                       "value x such that P(X > x) = p, useful for computing upper tail quantiles.",
	                       "binomial_quantile_complement(10, 0.3, 0.7)", "p");

	// Binomial sampling function
	RegisterSamplingFunction(instance, "binomial_sample", BinomialSampleFunc,
	                         "Generates random samples from the binomial distribution with n trials and success "
	                         "probability p. Returns the number of successes out of n trials.",
	                         "binomial_sample(10, 0.3)", {"n", "p"}, binomial_sample_param_types);

	// Statistical moment functions
	REGISTER_BINOMIAL_STATISTIC_FUNC(instance, "binomial_mean", BinomialMeanFunc,
	                                 "Returns the mean (n*p) of the binomial distribution.", "binomial_mean(10, 0.3)");

	REGISTER_BINOMIAL_STATISTIC_FUNC(instance, "binomial_stddev", BinomialStdDevFunc,
	                                 "Returns the standard deviation (√(n*p*(1-p))) of the binomial distribution.",
	                                 "binomial_stddev(10, 0.3)");

	REGISTER_BINOMIAL_STATISTIC_FUNC(instance, "binomial_variance", BinomialVarianceFunc,
	                                 "Returns the variance (n*p*(1-p)) of the binomial distribution.",
	                                 "binomial_variance(10, 0.3)");

	REGISTER_BINOMIAL_STATISTIC_FUNC(instance, "binomial_mode", BinomialModeFunc,
	                                 "Returns the mode (most likely value) of the binomial distribution.",
	                                 "binomial_mode(10, 0.3)");

	REGISTER_BINOMIAL_STATISTIC_FUNC(instance, "binomial_skewness", BinomialSkewnessFunc,
	                                 "Returns the skewness of the binomial distribution.",
	                                 "binomial_skewness(10, 0.3)");

	REGISTER_BINOMIAL_STATISTIC_FUNC(instance, "binomial_kurtosis", BinomialKurtosisFunc,
	                                 "Returns the kurtosis of the binomial distribution.",
	                                 "binomial_kurtosis(10, 0.3)");

	REGISTER_BINOMIAL_STATISTIC_FUNC(instance, "binomial_kurtosis_excess", BinomialKurtosisExcessFunc,
	                                 "Returns the excess kurtosis of the binomial distribution (kurtosis - 3).",
	                                 "binomial_kurtosis_excess(10, 0.3)");
}

} // namespace duckdb
