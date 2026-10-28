#include "utils.hpp"

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Hypothesis test aggregates. Each returns a STRUCT with the test statistic, the degrees of freedom and the
// two-sided (or upper-tail, for chi-squared) p-value. NULL inputs are ignored.
//===--------------------------------------------------------------------===//

struct NoDestroy {
	template <class STATE>
	static void Destroy(STATE &) {
	}
};

using TestResult = std::array<double, 3>;

double StudentTTwoSidedPValue(double t, double degrees_of_freedom) {
	const boost::math::students_t_distribution<double, StochasticPolicy> dist(degrees_of_freedom);
	return 2 * boost::math::cdf(boost::math::complement(dist, std::abs(t)));
}

void RequireFiniteValue(const char *name, double x) {
	if (!std::isfinite(x)) {
		throw InvalidInputException("%s: values must be finite, got %s", name, FormatParameter(x));
	}
}

//! Two independent samples identified by sample_index 0 and 1.
template <class Derived>
struct TwoSampleTTest : NoDestroy {
	using inputs = std::tuple<double, int64_t>;
	static constexpr std::array<const char *, 3> result_names = {"t_statistic", "degrees_of_freedom", "p_value"};
	struct State {
		RunningMoments samples[2];
	};

	static void Update(State &state, double x, int64_t sample_index) {
		RequireFiniteValue(Derived::name, x);
		if (sample_index != 0 && sample_index != 1) {
			throw InvalidInputException("%s: sample_index must be 0 or 1, got %s", Derived::name,
			                            std::to_string(sample_index));
		}
		state.samples[sample_index].Add(x);
	}
	static void Combine(const State &source, State &target) {
		target.samples[0].Merge(source.samples[0]);
		target.samples[1].Merge(source.samples[1]);
	}
	static std::optional<TestResult> Finalize(State &state) {
		const auto &a = state.samples[0];
		const auto &b = state.samples[1];
		if (a.count < 2 || b.count < 2) {
			return std::nullopt;
		}
		return Derived::Compute(a, b);
	}
};

struct WelchTTest : TwoSampleTTest<WelchTTest> {
	static constexpr const char *name = "welch_t_test";
	static std::optional<TestResult> Compute(const RunningMoments &a, const RunningMoments &b) {
		const double va = a.SampleVariance() / a.count;
		const double vb = b.SampleVariance() / b.count;
		if (va + vb == 0) {
			return std::nullopt;
		}
		const double t = (a.mean - b.mean) / std::sqrt(va + vb);
		const double df = (va + vb) * (va + vb) / (va * va / (a.count - 1) + vb * vb / (b.count - 1));
		return TestResult {t, df, StudentTTwoSidedPValue(t, df)};
	}
};

struct StudentTTest : TwoSampleTTest<StudentTTest> {
	static constexpr const char *name = "student_t_test";
	static std::optional<TestResult> Compute(const RunningMoments &a, const RunningMoments &b) {
		const double df = a.count + b.count - 2;
		const double pooled = (a.m2 + b.m2) / df;
		if (pooled == 0) {
			return std::nullopt;
		}
		const double t = (a.mean - b.mean) / std::sqrt(pooled * (1 / a.count + 1 / b.count));
		return TestResult {t, df, StudentTTwoSidedPValue(t, df)};
	}
};

struct OneSampleTTest : NoDestroy {
	static constexpr const char *name = "one_sample_t_test";
	using inputs = std::tuple<double, double>;
	static constexpr std::array<const char *, 3> result_names = {"t_statistic", "degrees_of_freedom", "p_value"};
	struct State {
		RunningMoments moments;
		GroupParameters<1> mu;
	};

	static void Update(State &state, double x, double mu) {
		RequireFiniteValue(name, x);
		RequireFiniteValue(name, mu);
		state.mu.Check(name, mu);
		state.moments.Add(x);
	}
	static void Combine(const State &source, State &target) {
		target.mu.Merge(name, source.mu);
		target.moments.Merge(source.moments);
	}
	static std::optional<TestResult> Finalize(State &state) {
		const auto &m = state.moments;
		if (m.count < 2 || m.m2 == 0) {
			return std::nullopt;
		}
		const double df = m.count - 1;
		const double t = (m.mean - state.mu.values[0]) / std::sqrt(m.SampleVariance() / m.count);
		return TestResult {t, df, StudentTTwoSidedPValue(t, df)};
	}
};

//! Pearson's chi-squared goodness-of-fit test over categories (one row per category). Expected values are
//! rescaled to the observed total, so they may be given as counts or as probabilities. With c = O / E,
//! sum((o - c e)^2 / (c e)) = (E / O) sum(o^2 / e) - O, which keeps the state mergeable.
struct ChiSquaredTest : NoDestroy {
	static constexpr const char *name = "chi_squared_test";
	using inputs = std::tuple<double, double>;
	static constexpr std::array<const char *, 3> result_names = {"statistic", "degrees_of_freedom", "p_value"};
	struct State {
		double categories;
		double observed;
		double expected;
		double ratio_sum;
	};

	static void Update(State &state, double observed, double expected) {
		RequireFiniteValue(name, observed);
		RequireFiniteValue(name, expected);
		if (observed < 0) {
			throw InvalidInputException("%s: observed counts must be >= 0, got %s", name, FormatParameter(observed));
		}
		if (!(expected > 0)) {
			throw InvalidInputException("%s: expected values must be > 0, got %s", name, FormatParameter(expected));
		}
		state.categories += 1;
		state.observed += observed;
		state.expected += expected;
		state.ratio_sum += observed * observed / expected;
	}
	static void Combine(const State &source, State &target) {
		target.categories += source.categories;
		target.observed += source.observed;
		target.expected += source.expected;
		target.ratio_sum += source.ratio_sum;
	}
	static std::optional<TestResult> Finalize(State &state) {
		if (state.categories < 2 || state.observed == 0) {
			return std::nullopt;
		}
		const double statistic = std::max(0.0, state.expected / state.observed * state.ratio_sum - state.observed);
		const double df = state.categories - 1;
		const boost::math::chi_squared_distribution<double, StochasticPolicy> dist(df);
		return TestResult {statistic, df, boost::math::cdf(boost::math::complement(dist, statistic))};
	}
};

} // namespace

void Load_hypothesis_tests(ExtensionLoader &loader) {
	RegisterStructAggregate<WelchTTest>(
	    loader, WelchTTest::name, {{"x", LogicalType::DOUBLE}, {"sample_index", LogicalType::BIGINT}},
	    {"Welch's unequal-variance t-test of whether two independent samples have the same mean. sample_index (0 or "
	     "1) assigns each row to a sample. Returns the t statistic, Welch-Satterthwaite degrees of freedom and the "
	     "two-sided p-value.",
	     "SELECT welch_t_test(value, CASE WHEN treated THEN 1 ELSE 0 END) FROM t", "hypothesis_test", ""});
	RegisterStructAggregate<StudentTTest>(
	    loader, StudentTTest::name, {{"x", LogicalType::DOUBLE}, {"sample_index", LogicalType::BIGINT}},
	    {"Student's two-sample t-test with pooled variance of whether two independent samples have the same mean. "
	     "sample_index (0 or 1) assigns each row to a sample. Returns the t statistic, degrees of freedom and the "
	     "two-sided p-value.",
	     "SELECT student_t_test(value, CASE WHEN treated THEN 1 ELSE 0 END) FROM t", "hypothesis_test", ""});
	RegisterStructAggregate<OneSampleTTest>(
	    loader, OneSampleTTest::name, {{"x", LogicalType::DOUBLE}, {"mu", LogicalType::DOUBLE}},
	    {"One-sample t-test of whether the mean of x equals mu. Returns the t statistic, degrees of freedom and the "
	     "two-sided p-value.",
	     "SELECT one_sample_t_test(value, 0.0) FROM t", "hypothesis_test", ""});
	RegisterStructAggregate<ChiSquaredTest>(
	    loader, ChiSquaredTest::name, {{"observed", LogicalType::DOUBLE}, {"expected", LogicalType::DOUBLE}},
	    {"Pearson's chi-squared goodness-of-fit test with one row per category. Expected values are rescaled to the "
	     "observed total, so they can be counts or probabilities. Returns the statistic, degrees of freedom "
	     "(categories - 1) and the p-value.",
	     "SELECT chi_squared_test(observed, expected) FROM categories", "hypothesis_test", ""});
}

} // namespace duckdb
