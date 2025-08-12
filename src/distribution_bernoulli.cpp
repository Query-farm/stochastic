#include "utils.hpp"
#include "rng_utils.hpp"

namespace duckdb {

namespace {
// Generate Bernoulli distribution functions
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliPdfFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::pdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliLogPdfFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::logpdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliCdfFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::cdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliLogCdfFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::logcdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliQuantileFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::quantile(dist, x))

DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliCdfComplementFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::cdf(boost::math::complement(dist, x)))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliLogCdfComplementFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::logcdf(boost::math::complement(dist, x)))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(BernoulliQuantileComplementFunc, boost::math::bernoulli_distribution<double>,
                                      boost::math::quantile(boost::math::complement(dist, x)))

// Specialized Bernoulli sampling function with validation
inline void BernoulliSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	GenericSingleParamSampleFunc<boost::random::bernoulli_distribution<double>, int8_t>(args, state, result);
}
} // namespace

void LoadDistributionBernoulli(DatabaseInstance &instance) {
	// Bernoulli distribution parameter types
	const duckdb::vector<LogicalType> bernoulli_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> bernoulli_sample_param_types = {LogicalType::DOUBLE};

	// Register Bernoulli distribution functions
	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_pdf", BernoulliPdfFunc,
	                        "Computes the probability mass function (PMF) of the Bernoulli distribution. Returns the "
	                        "probability of getting value x (0 or 1) for a Bernoulli trial with success probability p.",
	                        "bernoulli_pdf(0.3, 1)", "x");

	REGISTER_BERNOULLI_FUNC(
	    instance, "bernoulli_cdf", BernoulliCdfFunc,
	    "Computes the cumulative distribution function (CDF) of the Bernoulli distribution. Returns the probability "
	    "that X ≤ x for a Bernoulli distribution with success probability p.",
	    "bernoulli_cdf(0.3, 1)", "x");

	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_quantile", BernoulliQuantileFunc,
	                        "Computes the quantile function (inverse CDF) of the Bernoulli distribution. Returns the "
	                        "smallest value x such that P(X ≤ x) ≥ p.",
	                        "bernoulli_quantile(0.3, 0.5)", "p");

	// Log functions
	REGISTER_BERNOULLI_FUNC(
	    instance, "bernoulli_log_pdf", BernoulliLogPdfFunc,
	    "Computes the natural logarithm of the probability mass function (log-PMF) of the Bernoulli distribution. "
	    "Useful for numerical stability when dealing with very small probabilities.",
	    "bernoulli_log_pdf(0.3, 1)", "x");

	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_log_cdf", BernoulliLogCdfFunc,
	                        "Computes the natural logarithm of the cumulative distribution function (log-CDF) of the "
	                        "Bernoulli distribution. Useful for numerical stability in extreme tail calculations.",
	                        "bernoulli_log_cdf(0.3, 1)", "x");

	// Complement functions
	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_cdf_complement", BernoulliCdfComplementFunc,
	                        "Computes the complementary cumulative distribution function (1 - CDF) of the Bernoulli "
	                        "distribution. Returns the probability that X > x, equivalent to the survival function.",
	                        "bernoulli_cdf_complement(0.3, 0)", "x");

	REGISTER_BERNOULLI_FUNC(
	    instance, "bernoulli_log_cdf_complement", BernoulliLogCdfComplementFunc,
	    "Computes the natural logarithm of the complementary cumulative distribution function of the Bernoulli "
	    "distribution. Useful for numerical stability in tail probability calculations.",
	    "bernoulli_log_cdf_complement(0.3, 0)", "x");

	REGISTER_BERNOULLI_FUNC(instance, "bernoulli_quantile_complement", BernoulliQuantileComplementFunc,
	                        "Computes the complementary quantile function of the Bernoulli distribution. Returns the "
	                        "value x such that P(X > x) = p, useful for computing upper tail quantiles.",
	                        "bernoulli_quantile_complement(0.3, 0.3)", "p");

	// Bernoulli sampling function
	RegisterSingleParamSamplingFunction(instance, "bernoulli_sample", LogicalType::TINYINT, BernoulliSampleFunc,
	                                    "Generates random samples from the Bernoulli distribution with success "
	                                    "probability p. Returns 1 with probability p, 0 with probability (1-p).",
	                                    "bernoulli_sample(0.3)", {"p"}, bernoulli_sample_param_types);
}

} // namespace duckdb