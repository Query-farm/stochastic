#include "utils.hpp"

#include "distribution_traits.hpp"
#include <cmath>

namespace duckdb {

#define DISTRIBUTION_SHORT_NAME "uniform_int"
#define DISTRIBUTION_TEXT       string(string(DISTRIBUTION_SHORT_NAME) + " distribution")
#define DISTRIBUTION_NAME       uniform_int_distribution
#define DISTRIBUTION            DiscreteUniformDistribution
#define SAMPLE_DISTRIBUTION     std::uniform_int_distribution<int64_t>
#define REGISTER                RegisterFunction<DISTRIBUTION>

// Analytics must describe the same inclusive integer support as the sampler.
// Keep the existing DOUBLE bound arguments, but reject values that cannot be
// converted to BIGINT without truncation or undefined behavior.
class DiscreteUniformDistribution {
public:
	using value_type = double;

	static void Validate(double min, double max) {
		const double int64_limit = std::ldexp(1.0, 63);
		if (!std::isfinite(min) || !std::isfinite(max) || std::trunc(min) != min || std::trunc(max) != max ||
		    min < -int64_limit || max >= int64_limit || min > max) {
			throw InvalidInputException("uniform_int: bounds must be finite BIGINT integers with min <= max");
		}
	}

	DiscreteUniformDistribution(double min, double max) {
		Validate(min, max);
		lower = static_cast<int64_t>(min);
		upper = static_cast<int64_t>(max);
		count = Distance(upper, lower) + 1.0;
	}

	double Pdf(int64_t x) const {
		return x < lower || x > upper ? 0.0 : 1.0 / count;
	}

	double Cdf(int64_t x) const {
		return x < lower ? 0.0 : x >= upper ? 1.0 : (Distance(x, lower) + 1.0) / count;
	}

	double Survival(int64_t x) const {
		return x < lower ? 1.0 : x >= upper ? 0.0 : Distance(upper, x) / count;
	}

	int64_t Quantile(double p, bool complement = false) const {
		if (!std::isfinite(p) || p < 0 || p > 1) {
			throw InvalidInputException("uniform_int: probability must be finite and between 0 and 1");
		}
		if (p == 0) {
			return complement ? upper : lower;
		}
		if (p == 1) {
			return complement ? lower : upper;
		}
		// Compute the integer rank using the exact binary DOUBLE probability.
		// A rounded DOUBLE support count loses endpoints above 2^53, and a
		// rounded CDF can make several adjacent integers appear to have p=0.5.
		const auto size = static_cast<uint64_t>(upper) - static_cast<uint64_t>(lower) + 1;
		int exponent;
		const auto fraction = std::frexp(p, &exponent);
		const auto mantissa = static_cast<uint64_t>(std::ldexp(fraction, 53));
		const auto product = uhugeint_t(size) * uhugeint_t(mantissa);
		const auto shift = static_cast<uint64_t>(53 - exponent);
		// The product has at most 117 bits. Subnormal probabilities need no
		// oversized shift: their positive product is strictly between 0 and 1.
		const auto integral = shift >= 128 ? uhugeint_t(0) : product >> uhugeint_t(shift);
		const bool has_fraction = shift >= 128 || product != (integral << uhugeint_t(shift));
		const auto rank = integral.lower;
		const auto offset = complement ? size - rank - 1 : rank + (has_fraction ? 1 : 0) - 1;
		return AddOffset(lower, offset);
	}

	double Mean() const {
		return static_cast<double>(lower) / 2.0 + static_cast<double>(upper) / 2.0;
	}

	double Variance() const {
		return (count * count - 1.0) / 12.0;
	}

	double Hazard(int64_t x) const {
		if (x > upper) {
			throw InvalidInputException("uniform_int: hazard is undefined above the maximum");
		}
		return x < lower ? 0.0 : 1.0 / (Distance(upper, x) + 1.0);
	}

	double Skewness() const {
		CheckNonzeroVariance();
		return 0;
	}

	double KurtosisExcess() const {
		CheckNonzeroVariance();
		return -1.2 - 2.4 / (count * count - 1.0);
	}

	int64_t lower;
	int64_t upper;

private:
	static int64_t AddOffset(int64_t base, uint64_t offset) {
		if (base < 0) {
			const auto distance_to_zero = static_cast<uint64_t>(-(base + 1)) + 1;
			if (offset >= distance_to_zero) {
				return static_cast<int64_t>(offset - distance_to_zero);
			}
		}
		return base + static_cast<int64_t>(offset);
	}

	static double Distance(int64_t hi, int64_t lo) {
		return static_cast<double>(static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo));
	}

	void CheckNonzeroVariance() const {
		if (lower == upper) {
			throw InvalidInputException(
			    "uniform_int: standardized moments are undefined for a single-point distribution");
		}
	}

	double count;
};

template <typename DistType>
struct distribution_traits_base {
	using param1_t = double;
	using param2_t = double;
	using return_t = double;

	static constexpr std::array<const char *, 2> param_names = {"min", "max"};
	static constexpr const char *prefix = DISTRIBUTION_SHORT_NAME;

	static std::vector<LogicalType> LogicalParamTypes() {
		return {logical_type_map<param1_t>::Get(), logical_type_map<param2_t>::Get()};
	}

	static void ValidateParameters(param1_t min, param2_t max) {
		DiscreteUniformDistribution::Validate(min, max);
	}
};

#define DEFINE_DIST_TRAITS(DIST)                                                                                       \
	template <>                                                                                                        \
	struct distribution_traits<DIST> : public distribution_traits_base<DIST> {};

DEFINE_DIST_TRAITS(DISTRIBUTION);
DEFINE_DIST_TRAITS(SAMPLE_DISTRIBUTION);

#define CONCAT(a, b)            a##b
#define EXPAND_AND_CONCAT(a, b) CONCAT(a, b)

#define LOAD_DISTRIBUTION_FN void EXPAND_AND_CONCAT(Load_, DISTRIBUTION_NAME)(ExtensionLoader & loader)

LOAD_DISTRIBUTION_FN {
	vector<std::pair<string, LogicalType>> param_names_quantile = {{"p", LogicalType::DOUBLE}};
	vector<std::pair<string, LogicalType>> param_names_unary = {{"x", LogicalType::BIGINT}};

	// x-parameter functions (pdf, cdf, etc.) where x is BIGINT
	auto make_unary = [](auto func) {
		return [func](DataChunk &args, ExpressionState &state, Vector &result) {
			DistributionCallBinaryUnary<DISTRIBUTION, int64_t>(args, state, result, func);
		};
	};

	// p-parameter functions (quantile) where p is DOUBLE
	auto make_unary_p = [](auto func) {
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
	    loader, "sample", FunctionStability::VOLATILE, LogicalType::BIGINT,
	    [](DataChunk &args, ExpressionState &state, Vector &result) {
		    DistributionSampleBinary<SAMPLE_DISTRIBUTION, int64_t>(args, state, result);
	    },
	    "Generates random samples from the " + DISTRIBUTION_TEXT + " with specified parameters.", "sample(1, 6)");

	REGISTER(loader, "pdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return dist.Pdf(x); }),
	         "Computes the probability mass function (PMF) of the " + DISTRIBUTION_TEXT +
	             ". Returns P(X = x) on the inclusive integer support for a " + DISTRIBUTION_TEXT +
	             " with specified parameters.",
	         "pdf(1, 6, 3)", param_names_unary);

	REGISTER(loader, "log_pdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return std::log(dist.Pdf(x)); }),
	         "Computes the natural logarithm of the probability mass function (log-PMF) of the " + DISTRIBUTION_TEXT +
	             ". Useful for numerical stability when dealing with very small probabilities.",
	         "log_pdf(1, 6, 3)", param_names_unary);

	// === CUMULATIVE DISTRIBUTION FUNCTIONS ===
	REGISTER(loader, "cdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return dist.Cdf(x); }),
	         "Computes the cumulative distribution function (CDF) of the " + DISTRIBUTION_TEXT +
	             ". Returns the "
	             "probability that a random variable X is less than or equal to x.",
	         "cdf(1, 6, 3)", param_names_unary);

	REGISTER(loader, "cdf_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return dist.Survival(x); }),
	         "Computes the complementary cumulative distribution function (1 - CDF) of the " + DISTRIBUTION_TEXT +
	             ". Returns the probability that X > x, equivalent to the survival function.",
	         "cdf_complement(1, 6, 3)", param_names_unary);

	REGISTER(loader, "log_cdf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return std::log(dist.Cdf(x)); }),
	         "Computes the natural logarithm of the cumulative distribution function (CDF) of the " +
	             DISTRIBUTION_TEXT +
	             ". "
	             "Returns the logarithm of the probability that a random variable X is less than or equal to x.",
	         "log_cdf(1, 6, 3)", param_names_unary);

	REGISTER(
	    loader, "log_cdf_complement", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return std::log(dist.Survival(x)); }),
	    "Computes the natural logarithm of the complementary cumulative distribution function (1 - CDF) of the " +
	        DISTRIBUTION_TEXT +
	        ". Returns the logarithm of the probability that X > x, equivalent to the survival function.",
	    "log_cdf_complement(1, 6, 3)", param_names_unary);

	// === QUANTILE FUNCTIONS ===
	REGISTER(loader, "quantile", FunctionStability::CONSISTENT, LogicalType::BIGINT,
	         make_unary_p([](const auto &dist, auto p) -> int64_t { return dist.Quantile(p); }),
	         "Computes the quantile function (inverse CDF) of the " + DISTRIBUTION_TEXT +
	             ". Returns the value x "
	             "with P(X <= x) >= p, using the exact binary DOUBLE probability. At p = 0, returns min.",
	         "quantile(1, 6, 0.95)", param_names_quantile);

	REGISTER(loader, "quantile_complement", FunctionStability::CONSISTENT, LogicalType::BIGINT,
	         make_unary_p([](const auto &dist, auto p) -> int64_t { return dist.Quantile(p, true); }),
	         "Computes the complementary quantile function of the " + DISTRIBUTION_TEXT +
	             ". Returns the value x "
	             "with P(X > x) <= p, using the exact binary DOUBLE probability. At p = 1, returns min.",
	         "quantile_complement(1, 6, 0.05)", param_names_quantile);

	REGISTER(loader, "hazard", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return dist.Hazard(x); }),
	         "Computes the discrete hazard P(X = x) / P(X >= x). Undefined above max.", "hazard(1, 6, 3)",
	         param_names_unary);

	REGISTER(
	    loader, "chf", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	    make_unary([](const auto &dist, auto x) -> DISTRIBUTION::value_type { return -std::log(dist.Survival(x)); }),
	    "Computes the cumulative hazard function of the " + DISTRIBUTION_TEXT + ".", "chf(1, 6, 3)", param_names_unary);

	// === DISTRIBUTION PROPERTIES ===

	REGISTER(loader, "mean", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) -> double { return dist.Mean(); }),
	         "Returns the mean (μ) of the " + DISTRIBUTION_TEXT + ", which is the first moment.", "mean(1, 6)");

	REGISTER(loader, "stddev", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) -> double { return std::sqrt(dist.Variance()); }),
	         "Returns the standard deviation (σ) of the " + DISTRIBUTION_TEXT + ".", "stddev(1, 6)");

	REGISTER(loader, "variance", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) -> double { return dist.Variance(); }),
	         "Returns the variance (σ²) of the " + DISTRIBUTION_TEXT + ".", "variance(1, 6)");

	REGISTER(loader, "mode", FunctionStability::CONSISTENT, LogicalType::BIGINT,
	         make_none([](const auto &dist) -> int64_t { return dist.lower; }),
	         "Returns min as a representative mode; every integer in the support has equal mass.", "mode(1, 6)");

	REGISTER(loader, "median", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) -> double { return dist.Mean(); }),
	         "Returns the median (50th percentile) of the " + DISTRIBUTION_TEXT + ", which equals the mean.",
	         "median(1, 6)");

	REGISTER(loader, "skewness", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return dist.Skewness(); }),
	         "Returns the skewness of the " + DISTRIBUTION_TEXT + ".", "skewness(1, 6)");

	REGISTER(loader, "kurtosis", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return dist.KurtosisExcess() + 3.0; }),
	         "Returns the kurtosis of the " + DISTRIBUTION_TEXT + ".", "kurtosis(1, 6)");

	REGISTER(loader, "kurtosis_excess", FunctionStability::CONSISTENT, LogicalType::DOUBLE,
	         make_none([](const auto &dist) { return dist.KurtosisExcess(); }),
	         "Returns the excess kurtosis of the " + DISTRIBUTION_TEXT + ".", "kurtosis_excess(1, 6)");

	REGISTER(loader, "range", FunctionStability::CONSISTENT, LogicalType::ARRAY(LogicalType::BIGINT, 2),
	         make_none([](const auto &dist) -> std::pair<int64_t, int64_t> {
		         return {dist.lower, dist.upper};
	         }),
	         "Returns the range of the " + DISTRIBUTION_TEXT + ".", "range(1, 6)");

	REGISTER(loader, "support", FunctionStability::CONSISTENT, LogicalType::ARRAY(LogicalType::BIGINT, 2),
	         make_none([](const auto &dist) -> std::pair<int64_t, int64_t> {
		         return {dist.lower, dist.upper};
	         }),
	         "Returns the support of the " + DISTRIBUTION_TEXT + ".", "support(1, 6)");
}
} // end namespace duckdb
