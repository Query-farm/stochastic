#include "utils.hpp"
#include "rng_utils.hpp"

namespace duckdb {

namespace {
// Generate Exponential distribution functions
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(ExponentialPdfFunc, boost::math::exponential_distribution<double>,
                                      boost::math::pdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(ExponentialLogPdfFunc, boost::math::exponential_distribution<double>,
                                      boost::math::logpdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(ExponentialCdfFunc, boost::math::exponential_distribution<double>,
                                      boost::math::cdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(ExponentialLogCdfFunc, boost::math::exponential_distribution<double>,
                                      boost::math::logcdf(dist, x))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(ExponentialQuantileFunc, boost::math::exponential_distribution<double>,
                                      boost::math::quantile(dist, x))

DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(ExponentialCdfComplementFunc, boost::math::exponential_distribution<double>,
                                      boost::math::cdf(boost::math::complement(dist, x)))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(ExponentialLogCdfComplementFunc, boost::math::exponential_distribution<double>,
                                      boost::math::logcdf(boost::math::complement(dist, x)))
DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(ExponentialQuantileComplementFunc, boost::math::exponential_distribution<double>,
                                      boost::math::quantile(boost::math::complement(dist, x)))

// Statistical moment functions for exponential distribution
template <typename Distribution, typename Operation>
inline void SingleParameterStatisticFunc(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &param_vector = args.data[0];

	// If parameter is constant, optimize the computation
	if (param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		const auto param = ConstantVector::GetData<double>(param_vector)[0];
		Distribution dist(param);
		auto stat_value = op(dist);

		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<double>(result);
		result_data[0] = stat_value;
		return;
	}

	UnaryExecutor::Execute<double, double>(param_vector, result, args.size(), [&](double param) {
		Distribution dist(param);
		return op(dist);
	});
}

// Generic macro to define single-parameter distribution statistic functions
#define DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(func_name, distribution_type, boost_operation)                 \
	inline void func_name(DataChunk &args, ExpressionState &state, Vector &result) {                                   \
		SingleParameterStatisticFunc<distribution_type>(args, state, result,                                           \
		                                                [](const auto &dist) { return boost_operation; });             \
	}

DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(ExponentialMeanFunc, boost::math::exponential_distribution<double>,
                                                boost::math::mean(dist))
DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(ExponentialStdDevFunc, boost::math::exponential_distribution<double>,
                                                boost::math::standard_deviation(dist))
DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(ExponentialVarianceFunc, boost::math::exponential_distribution<double>,
                                                boost::math::variance(dist))
DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(ExponentialModeFunc, boost::math::exponential_distribution<double>,
                                                boost::math::mode(dist))
DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(ExponentialMedianFunc, boost::math::exponential_distribution<double>,
                                                boost::math::median(dist))
DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(ExponentialSkewnessFunc, boost::math::exponential_distribution<double>,
                                                boost::math::skewness(dist))
DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(ExponentialKurtosisFunc, boost::math::exponential_distribution<double>,
                                                boost::math::kurtosis(dist))
DEFINE_SINGLE_PARAM_DISTRIBUTION_STATISTIC_FUNC(ExponentialKurtosisExcessFunc,
                                                boost::math::exponential_distribution<double>,
                                                boost::math::kurtosis_excess(dist))

// Specialized Exponential sampling function
inline void ExponentialSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	GenericSingleParamSampleFunc<boost::random::exponential_distribution<double>, double>(args, state, result);
}
} // namespace

void LoadDistributionExponential(DatabaseInstance &instance) {
	// Exponential distribution parameter types
	const duckdb::vector<LogicalType> exponential_param_types = {LogicalType::DOUBLE, LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> exponential_sample_param_types = {LogicalType::DOUBLE};
	const duckdb::vector<LogicalType> exponential_statistic_param_types = {LogicalType::DOUBLE};

	// Register Exponential distribution functions
	REGISTER_EXPONENTIAL_FUNC(
	    instance, "exponential_pdf", ExponentialPdfFunc,
	    "Computes the probability density function (PDF) of the exponential distribution. Returns the "
	    "probability density at point x for an exponential distribution with rate parameter lambda.",
	    "exponential_pdf(1.5, 2.0)", "x");

	REGISTER_EXPONENTIAL_FUNC(
	    instance, "exponential_cdf", ExponentialCdfFunc,
	    "Computes the cumulative distribution function (CDF) of the exponential distribution. Returns the probability "
	    "that X ≤ x for an exponential distribution with rate parameter lambda.",
	    "exponential_cdf(1.5, 2.0)", "x");

	REGISTER_EXPONENTIAL_FUNC(
	    instance, "exponential_quantile", ExponentialQuantileFunc,
	    "Computes the quantile function (inverse CDF) of the exponential distribution. Returns the "
	    "value x such that P(X ≤ x) = p for an exponential distribution with rate parameter lambda.",
	    "exponential_quantile(1.5, 0.5)", "p");

	// Log functions
	REGISTER_EXPONENTIAL_FUNC(
	    instance, "exponential_log_pdf", ExponentialLogPdfFunc,
	    "Computes the natural logarithm of the probability density function (log-PDF) of the exponential distribution. "
	    "Useful for numerical stability when dealing with very small probabilities.",
	    "exponential_log_pdf(1.5, 2.0)", "x");

	REGISTER_EXPONENTIAL_FUNC(instance, "exponential_log_cdf", ExponentialLogCdfFunc,
	                          "Computes the natural logarithm of the cumulative distribution function (log-CDF) of the "
	                          "exponential distribution. Useful for numerical stability in extreme tail calculations.",
	                          "exponential_log_cdf(1.5, 2.0)", "x");

	// Complement functions
	REGISTER_EXPONENTIAL_FUNC(
	    instance, "exponential_cdf_complement", ExponentialCdfComplementFunc,
	    "Computes the complementary cumulative distribution function (1 - CDF) of the exponential "
	    "distribution. Returns the probability that X > x, equivalent to the survival function.",
	    "exponential_cdf_complement(1.5, 2.0)", "x");

	REGISTER_EXPONENTIAL_FUNC(
	    instance, "exponential_log_cdf_complement", ExponentialLogCdfComplementFunc,
	    "Computes the natural logarithm of the complementary cumulative distribution function of the exponential "
	    "distribution. Useful for numerical stability in tail probability calculations.",
	    "exponential_log_cdf_complement(1.5, 2.0)", "x");

	REGISTER_EXPONENTIAL_FUNC(
	    instance, "exponential_quantile_complement", ExponentialQuantileComplementFunc,
	    "Computes the complementary quantile function of the exponential distribution. Returns the "
	    "value x such that P(X > x) = p, useful for computing upper tail quantiles.",
	    "exponential_quantile_complement(1.5, 0.3)", "p");

	// Exponential sampling function
	RegisterSingleParamSamplingFunction(
	    instance, "exponential_sample", LogicalType::DOUBLE, ExponentialSampleFunc,
	    "Generates random samples from the exponential distribution with rate parameter lambda. "
	    "Returns random values following an exponential distribution.",
	    "exponential_sample(1.5)", {"lambda"}, exponential_sample_param_types);

	// Statistical moment functions
	RegisterSingleParamDistributionFunction(instance, "exponential_mean", ExponentialMeanFunc,
	                                        "Returns the mean (1/lambda) of the exponential distribution.",
	                                        "exponential_mean(1.5)", {"lambda"}, exponential_statistic_param_types);

	RegisterSingleParamDistributionFunction(
	    instance, "exponential_stddev", ExponentialStdDevFunc,
	    "Returns the standard deviation (1/lambda) of the exponential distribution.", "exponential_stddev(1.5)",
	    {"lambda"}, exponential_statistic_param_types);

	RegisterSingleParamDistributionFunction(instance, "exponential_variance", ExponentialVarianceFunc,
	                                        "Returns the variance (1/lambda²) of the exponential distribution.",
	                                        "exponential_variance(1.5)", {"lambda"}, exponential_statistic_param_types);

	RegisterSingleParamDistributionFunction(instance, "exponential_mode", ExponentialModeFunc,
	                                        "Returns the mode (0) of the exponential distribution.",
	                                        "exponential_mode(1.5)", {"lambda"}, exponential_statistic_param_types);

	RegisterSingleParamDistributionFunction(instance, "exponential_median", ExponentialMedianFunc,
	                                        "Returns the median (ln(2)/lambda) of the exponential distribution.",
	                                        "exponential_median(1.5)", {"lambda"}, exponential_statistic_param_types);

	RegisterSingleParamDistributionFunction(instance, "exponential_skewness", ExponentialSkewnessFunc,
	                                        "Returns the skewness (2) of the exponential distribution.",
	                                        "exponential_skewness(1.5)", {"lambda"}, exponential_statistic_param_types);

	RegisterSingleParamDistributionFunction(instance, "exponential_kurtosis", ExponentialKurtosisFunc,
	                                        "Returns the kurtosis (9) of the exponential distribution.",
	                                        "exponential_kurtosis(1.5)", {"lambda"}, exponential_statistic_param_types);

	RegisterSingleParamDistributionFunction(
	    instance, "exponential_kurtosis_excess", ExponentialKurtosisExcessFunc,
	    "Returns the excess kurtosis (6) of the exponential distribution (kurtosis - 3).",
	    "exponential_kurtosis_excess(1.5)", {"lambda"}, exponential_statistic_param_types);
}

} // namespace duckdb
