#include "utils.hpp"

#include <cmath>

namespace duckdb {

namespace {

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

struct UniformIntSpec : DistributionSpecBase {
	static constexpr const char *name = "uniform_int";
	static constexpr const char *display_name = "discrete uniform";
	static constexpr bool discrete = true;
	static constexpr bool check_sample_range = false;
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"min", "max"};
	using boost_type = DiscreteUniformDistribution;
	using sampler_type = std::uniform_int_distribution<int64_t>;
	using sample_t = int64_t;
	using quantile_t = int64_t;
	static constexpr const char *example_params = "1, 6";
	static constexpr const char *example_x = "3";

	static void Validate(double min, double max) {
		DiscreteUniformDistribution::Validate(min, max);
	}
};

} // namespace

void Load_uniform_int_distribution(ExtensionLoader &loader) {
	using Spec = UniformIntSpec;
	using Dist = DiscreteUniformDistribution;
	using X = std::tuple<int64_t>;
	using P = std::tuple<double>;
	const vector<std::pair<string, LogicalType>> x_param = {{"x", LogicalType::BIGINT}};
	const vector<std::pair<string, LogicalType>> p_param = {{"p", LogicalType::DOUBLE}};
	const string text = "discrete uniform distribution";

	RegisterSampling<Spec>(loader);

	auto pmf = [](const Dist &dist, int64_t x) {
		return dist.Pdf(x);
	};
	auto log_pmf = [](const Dist &dist, int64_t x) {
		return std::log(dist.Pdf(x));
	};
	for (auto name : {"pdf", "pmf"}) {
		RegisterEvaluation<Spec, double, X>(loader, name, pmf,
		                                    {"Computes the probability mass function (PMF) of the " + text +
		                                         ", P(X = x) on the inclusive integer support [min, max].",
		                                     "1, 6, 3", "density"},
		                                    x_param);
	}
	for (auto name : {"log_pdf", "log_pmf"}) {
		RegisterEvaluation<Spec, double, X>(
		    loader, name, log_pmf,
		    {"Computes the natural logarithm of the probability mass function (PMF) of the " + text + ".", "1, 6, 3",
		     "density"},
		    x_param);
	}

	RegisterEvaluation<Spec, double, X>(
	    loader, "cdf", [](const Dist &dist, int64_t x) { return dist.Cdf(x); },
	    {"Computes the cumulative distribution function (CDF) of the " + text + ", P(X <= x).", "1, 6, 3",
	     "cumulative"},
	    x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "cdf_complement", [](const Dist &dist, int64_t x) { return dist.Survival(x); },
	    {"Computes the survival function of the " + text + ", P(X > x).", "1, 6, 3", "cumulative"}, x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "log_cdf", [](const Dist &dist, int64_t x) { return std::log(dist.Cdf(x)); },
	    {"Computes the natural logarithm of the cumulative distribution function (CDF) of the " + text + ".", "1, 6, 3",
	     "cumulative"},
	    x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "log_cdf_complement", [](const Dist &dist, int64_t x) { return std::log(dist.Survival(x)); },
	    {"Computes the natural logarithm of the survival function of the " + text + ", log P(X > x).", "1, 6, 3",
	     "cumulative"},
	    x_param);

	RegisterEvaluation<Spec, int64_t, P>(
	    loader, "quantile", [](const Dist &dist, double p) { return dist.Quantile(p); },
	    {"Computes the quantile function (inverse CDF) of the " + text +
	         ". Returns the smallest x with P(X <= x) >= p, using the exact binary DOUBLE probability. At p = 0, "
	         "returns min.",
	     "1, 6, 0.95", "quantile"},
	    p_param);
	RegisterEvaluation<Spec, int64_t, P>(
	    loader, "quantile_complement", [](const Dist &dist, double p) { return dist.Quantile(p, true); },
	    {"Computes the complementary quantile function of the " + text +
	         ". Returns the smallest x with P(X > x) <= p, using the exact binary DOUBLE probability. At p = 1, "
	         "returns min.",
	     "1, 6, 0.05", "quantile"},
	    p_param);

	RegisterEvaluation<Spec, double, X>(
	    loader, "hazard", [](const Dist &dist, int64_t x) { return dist.Hazard(x); },
	    {"Computes the discrete hazard P(X = x) / P(X >= x) of the " + text + ". Undefined above max.", "1, 6, 3",
	     "hazard"},
	    x_param);
	RegisterEvaluation<Spec, double, X>(
	    loader, "chf", [](const Dist &dist, int64_t x) { return -std::log(dist.Survival(x)); },
	    {"Computes the cumulative hazard function of the " + text + ".", "1, 6, 3", "hazard"}, x_param);

	RegisterEvaluation<Spec, double>(loader, "mean", [](const Dist &dist) { return dist.Mean(); },
	                                 {"Returns the mean of the " + text + ".", "1, 6", "property"});
	RegisterEvaluation<Spec, double>(loader, "stddev", [](const Dist &dist) { return std::sqrt(dist.Variance()); },
	                                 {"Returns the standard deviation of the " + text + ".", "1, 6", "property"});
	RegisterEvaluation<Spec, double>(loader, "variance", [](const Dist &dist) { return dist.Variance(); },
	                                 {"Returns the variance of the " + text + ".", "1, 6", "property"});
	RegisterEvaluation<Spec, int64_t>(
	    loader, "mode", [](const Dist &dist) { return dist.lower; },
	    {"Returns min as a representative mode; every integer in the support has equal mass.", "1, 6", "property"});
	RegisterEvaluation<Spec, double>(
	    loader, "median", [](const Dist &dist) { return dist.Mean(); },
	    {"Returns the median of the " + text + ", which equals its mean.", "1, 6", "property"});
	RegisterEvaluation<Spec, double>(loader, "skewness", [](const Dist &dist) { return dist.Skewness(); },
	                                 {"Returns the skewness of the " + text + ".", "1, 6", "property"});
	RegisterEvaluation<Spec, double>(loader, "kurtosis", [](const Dist &dist) { return dist.KurtosisExcess() + 3.0; },
	                                 {"Returns the kurtosis of the " + text + ".", "1, 6", "property"});
	RegisterEvaluation<Spec, double>(loader, "kurtosis_excess", [](const Dist &dist) { return dist.KurtosisExcess(); },
	                                 {"Returns the excess kurtosis of the " + text + ".", "1, 6", "property"});

	using Interval = std::pair<int64_t, int64_t>;
	auto interval = [](const Dist &dist) -> Interval {
		return {dist.lower, dist.upper};
	};
	RegisterEvaluation<Spec, Interval>(loader, "range", interval,
	                                   {"Returns the range of the " + text + " as [min, max].", "1, 6", "property"});
	RegisterEvaluation<Spec, Interval>(loader, "support", interval,
	                                   {"Returns the support of the " + text + " as [min, max].", "1, 6", "property"});
}

} // namespace duckdb
