#include "utils.hpp"
#include "rng_utils.hpp"
#include "distribution_traits.hpp"

namespace duckdb {

#define DISTRIBUTION_SHORT_NAME "gamma"
#define DISTRIBUTION_TEXT       string(string(DISTRIBUTION_SHORT_NAME) + " distribution")
#define DISTRIBUTION_NAME       gamma_distribution
// Specialization for boost::random::normal_distribution<double>
template <>
struct distribution_traits<boost::math::DISTRIBUTION_NAME<double>> {
	using param1_t = double;
	using param2_t = double;

	static constexpr std::array<const char *, 2> param_names = {"alpha", "beta"};
	static constexpr string prefix = DISTRIBUTION_SHORT_NAME;

	static std::vector<LogicalType> LogicalParamTypes() {
		return {logical_type_map<param1_t>::Get(), logical_type_map<param2_t>::Get()};
	}
};

template <>
struct distribution_traits<boost::random::DISTRIBUTION_NAME<double>> {
	using param1_t = double;
	using param2_t = double;

	static constexpr std::array<const char *, 2> param_names = {"alpha", "beta"};

	static constexpr string prefix = DISTRIBUTION_SHORT_NAME;

	static std::vector<LogicalType> LogicalParamTypes() {
		return {logical_type_map<param1_t>::Get(), logical_type_map<param2_t>::Get()};
	}
};

#define DISTRIBUTION        boost::math::DISTRIBUTION_NAME<double>
#define SAMPLE_DISTRIBUTION boost::random::DISTRIBUTION_NAME<double>
#define REGISTER            RegisterFunction<DISTRIBUTION>

#define CONCAT(a, b)            a##b
#define EXPAND_AND_CONCAT(a, b) CONCAT(a, b)

#define LOAD_DISTRIBUTION_FN void EXPAND_AND_CONCAT(Load_, DISTRIBUTION_NAME)(DatabaseInstance & instance)

LOAD_DISTRIBUTION_FN {
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
		    DistributionSampleBinary<SAMPLE_DISTRIBUTION, double>(args, state, result);
	    },
	    "Generates random samples from the " + DISTRIBUTION_TEXT + " with specified parameters.", "sample(0.0, 1.0)");

	REGISTER(instance, "pdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::pdf(dist, x); }),
	         "Computes the probability density function (PDF) of the " + DISTRIBUTION_TEXT +
	             ". Returns the probability density"
	             "at point x for a " +
	             DISTRIBUTION_TEXT + " with specified parameters.",
	         "pdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(
	    instance, "log_pdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::logpdf(dist, x); }),
	    "Computes the natural logarithm of the probability density function (log-PDF) of the " + DISTRIBUTION_TEXT +
	        ". Useful for numerical stability when dealing with very small probabilities.",
	    "log_pdf(0, 1.0, 0.5)", param_names_unary);

	// === CUMULATIVE DISTRIBUTION FUNCTIONS ===
	REGISTER(instance, "cdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::cdf(dist, x); }),
	         "Computes the cumulative distribution function (CDF) of the " + DISTRIBUTION_TEXT +
	             ". Returns the "
	             "probability that a random variable X is less than or equal to x.",
	         "cdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "cdf_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type {
		         return boost::math::cdf(boost::math::complement(dist, x));
	         }),
	         "Computes the complementary cumulative distribution function (1 - CDF) of the " + DISTRIBUTION_TEXT +
	             ". Returns the probability that X > x, equivalent to the survival function.",
	         "cdf_complement(0, 1.0, 0.5)", param_names_unary);

	REGISTER(
	    instance, "log_cdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::logcdf(dist, x); }),
	    "Computes the natural logarithm of the cumulative distribution function (CDF) of the " + DISTRIBUTION_TEXT +
	        ". "
	        "Returns the logarithm of the probability that a random variable X is less than or equal to x.",
	    "log_cdf(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "log_cdf_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type {
		         return boost::math::logcdf(boost::math::complement(dist, x));
	         }),
	         "Computes the natural logarithm of the complementary cumulative distribution function (1 - CDF) of the " +
	             DISTRIBUTION_TEXT +
	             ". Returns the logarithm of the probability that X > x, equivalent to the survival function.",
	         "log_cdf_complement(0, 1.0, 0.5)", param_names_unary);

	// === QUANTILE FUNCTIONS ===
	REGISTER(
	    instance, "quantile", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto p) -> DISTRIBUTION::value_type { return boost::math::quantile(dist, p); }),
	    "Computes the quantile function (inverse CDF) of the " + DISTRIBUTION_TEXT +
	        ". Returns the value x "
	        "such that P(X ≤ x) = p, where p is the cumulative probability.",
	    "quantile(0, 1.0, 0.95)", param_names_quantile);

	REGISTER(instance, "quantile_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto p) -> DISTRIBUTION::value_type {
		         return boost::math::quantile(boost::math::complement(dist, p));
	         }),
	         "Computes the complementary quantile function of the " + DISTRIBUTION_TEXT +
	             ". Returns the value x "
	             "such that P(X > x) = p, useful for computing upper tail quantiles.",
	         "quantile_complement(0, 1.0, 0.95)", param_names_quantile);

	REGISTER(
	    instance, "hazard", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::hazard(dist, x); }),
	    "Computes the hazard function of the " + DISTRIBUTION_TEXT + ".", "hazard(0, 1.0, 0.5)", param_names_unary);

	REGISTER(instance, "chf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return boost::math::chf(dist, x); }),
	         "Computes the cumulative hazard function of the " + DISTRIBUTION_TEXT + ".", "chf(0, 1.0, 0.5)",
	         param_names_unary);

	// === DISTRIBUTION PROPERTIES ===

	REGISTER(instance, "mean", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::mean(dist); }),
	         "Returns the mean (μ) of the " + DISTRIBUTION_TEXT + ", which is the first moment.", "mean(0.0, 1.0)");

	REGISTER(instance, "stddev", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::standard_deviation(dist); }),
	         "Returns the standard deviation (σ) of the " + DISTRIBUTION_TEXT + ".", "stddev(0.0, 1.0)");

	REGISTER(instance, "variance", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::variance(dist); }),
	         "Returns the variance (σ²) of the " + DISTRIBUTION_TEXT + ".", "variance(0.0, 1.0)");

	REGISTER(instance, "mode", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::mode(dist); }),
	         "Returns the mode (most likely value) of the " + DISTRIBUTION_TEXT + ", which equals the mean.",
	         "mode(0.0, 1.0)");

	REGISTER(instance, "median", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::median(dist); }),
	         "Returns the median (50th percentile) of the " + DISTRIBUTION_TEXT + ", which equals the mean.",
	         "median(0.0, 1.0)");

	REGISTER(instance, "skewness", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::skewness(dist); }),
	         "Returns the skewness of the " + DISTRIBUTION_TEXT + ".", "skewness(0.0, 1.0)");

	REGISTER(instance, "kurtosis", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::kurtosis(dist); }),
	         "Returns the kurtosis of the " + DISTRIBUTION_TEXT + ".", "kurtosis(0.0, 1.0)");

	REGISTER(instance, "kurtosis_excess", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return boost::math::kurtosis_excess(dist); }),
	         "Returns the excess kurtosis of the " + DISTRIBUTION_TEXT + ".", "kurtosis_excess(0.0, 1.0)");

	REGISTER(instance, "range", FunctionStability::CONSISTENT, LogicalType::ARRAY(LogicalType::DOUBLE, 2),
	         make_none([](const auto &dist) { return boost::math::range(dist); }),
	         "Returns the range of the " + DISTRIBUTION_TEXT + ".", "range(0.0, 1.0)");

	REGISTER(instance, "support", FunctionStability::CONSISTENT, LogicalType::ARRAY(LogicalType::DOUBLE, 2),
	         make_none([](const auto &dist) { return boost::math::support(dist); }),
	         "Returns the support of the " + DISTRIBUTION_TEXT + ".", "support(0.0, 1.0)");
}
} // end namespace duckdb
