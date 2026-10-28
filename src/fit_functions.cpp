#include "utils.hpp"

#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Maximum likelihood fits: dist_<name>_fit(x) returns the fitted parameters as a STRUCT whose fields are named
// like the distribution's parameters. NULL inputs are ignored and an empty or degenerate sample returns NULL.
//===--------------------------------------------------------------------===//

template <class T>
void RequireFiniteSample(const char *name, T x) {
	if (!std::isfinite(x)) {
		throw InvalidInputException("%s: fit values must be finite, got %s", name, FormatParameter(x));
	}
}

void RequireFitDomain(const char *name, bool valid, const char *domain, double x) {
	if (!valid) {
		throw InvalidInputException("%s: fit values must be %s, got %s", name, domain, FormatParameter(x));
	}
}

struct NoDestroy {
	template <class STATE>
	static void Destroy(STATE &) {
	}
};

//! Fits using the mean and variance of g(x), where g is the identity or a transform such as log.
template <class Derived>
struct MomentsFit : NoDestroy {
	using inputs = std::tuple<double>;
	struct State {
		RunningMoments moments;
	};

	static void Update(State &state, double x) {
		RequireFiniteSample(Derived::name, x);
		state.moments.Add(Derived::Transform(x));
	}
	static void Combine(const State &source, State &target) {
		target.moments.Merge(source.moments);
	}
};

struct NormalFit : MomentsFit<NormalFit> {
	static constexpr const char *name = "normal";
	static constexpr std::array<const char *, 2> result_names = {"mean", "stddev"};
	static double Transform(double x) {
		return x;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		const auto &m = state.moments;
		if (m.count < 2 || m.m2 <= 0) {
			return std::nullopt;
		}
		return std::array<double, 2> {m.mean, std::sqrt(m.m2 / m.count)};
	}
};

struct LognormalFit : MomentsFit<LognormalFit> {
	static constexpr const char *name = "lognormal";
	static constexpr std::array<const char *, 2> result_names = {"mean", "stddev"};
	static double Transform(double x) {
		RequireFitDomain(name, x > 0, "> 0", x);
		return std::log(x);
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		const auto &m = state.moments;
		if (m.count < 2 || m.m2 <= 0) {
			return std::nullopt;
		}
		return std::array<double, 2> {m.mean, std::sqrt(m.m2 / m.count)};
	}
};

//! Fits whose estimate is a function of the sample mean (x must lie in the given domain).
template <class Derived>
struct MeanFit : NoDestroy {
	using inputs = std::tuple<double>;
	struct State {
		double count;
		double sum;
	};

	static void Update(State &state, double x) {
		RequireFiniteSample(Derived::name, x);
		Derived::CheckDomain(x);
		state.count += 1;
		state.sum += x;
	}
	static void Combine(const State &source, State &target) {
		target.count += source.count;
		target.sum += source.sum;
	}
	static std::optional<std::array<double, 1>> Finalize(State &state) {
		if (state.count == 0) {
			return std::nullopt;
		}
		const auto estimate = Derived::Estimate(state.sum / state.count);
		if (!estimate) {
			return std::nullopt;
		}
		return std::array<double, 1> {*estimate};
	}
};

struct ExponentialFit : MeanFit<ExponentialFit> {
	static constexpr const char *name = "exponential";
	static constexpr std::array<const char *, 1> result_names = {"rate"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, x >= 0, ">= 0", x);
	}
	static std::optional<double> Estimate(double mean) {
		return mean > 0 ? std::optional<double>(1 / mean) : std::nullopt;
	}
};

struct PoissonFit : MeanFit<PoissonFit> {
	static constexpr const char *name = "poisson";
	static constexpr std::array<const char *, 1> result_names = {"rate"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, x >= 0 && std::trunc(x) == x, "non-negative integers", x);
	}
	static std::optional<double> Estimate(double mean) {
		return mean > 0 ? std::optional<double>(mean) : std::nullopt;
	}
};

struct BernoulliFit : MeanFit<BernoulliFit> {
	static constexpr const char *name = "bernoulli";
	static constexpr std::array<const char *, 1> result_names = {"prob"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, x == 0 || x == 1, "0 or 1", x);
	}
	static std::optional<double> Estimate(double mean) {
		return mean;
	}
};

//! Geometric distribution counting failures before the first success: p = 1 / (1 + mean).
struct GeometricFit : MeanFit<GeometricFit> {
	static constexpr const char *name = "geometric";
	static constexpr std::array<const char *, 1> result_names = {"prob"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, x >= 0 && std::trunc(x) == x, "non-negative integers", x);
	}
	static std::optional<double> Estimate(double mean) {
		return 1 / (1 + mean);
	}
};

struct RayleighFit : NoDestroy {
	static constexpr const char *name = "rayleigh";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 1> result_names = {"scale"};
	struct State {
		double count;
		double sum_squares;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(name, x);
		RequireFitDomain(name, x >= 0, ">= 0", x);
		state.count += 1;
		state.sum_squares += x * x;
	}
	static void Combine(const State &source, State &target) {
		target.count += source.count;
		target.sum_squares += source.sum_squares;
	}
	static std::optional<std::array<double, 1>> Finalize(State &state) {
		if (state.count == 0 || state.sum_squares == 0) {
			return std::nullopt;
		}
		return std::array<double, 1> {std::sqrt(state.sum_squares / (2 * state.count))};
	}
};

struct UniformRealFit : NoDestroy {
	static constexpr const char *name = "uniform_real";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> result_names = {"min", "max"};
	struct State {
		bool set;
		double min;
		double max;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(name, x);
		if (!state.set) {
			state = {true, x, x};
			return;
		}
		state.min = std::min(state.min, x);
		state.max = std::max(state.max, x);
	}
	static void Combine(const State &source, State &target) {
		if (!source.set) {
			return;
		}
		if (!target.set) {
			target = source;
			return;
		}
		target.min = std::min(target.min, source.min);
		target.max = std::max(target.max, source.max);
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (!state.set || !(state.min < state.max)) {
			return std::nullopt;
		}
		return std::array<double, 2> {state.min, state.max};
	}
};

//! scale = min(x), shape = n / sum(log(x / scale)).
struct ParetoFit : NoDestroy {
	static constexpr const char *name = "pareto";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> result_names = {"scale", "shape"};
	struct State {
		double count;
		double sum_log;
		double min;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(name, x);
		RequireFitDomain(name, x > 0, "> 0", x);
		state.min = state.count == 0 ? x : std::min(state.min, x);
		state.count += 1;
		state.sum_log += std::log(x);
	}
	static void Combine(const State &source, State &target) {
		if (source.count == 0) {
			return;
		}
		target.min = target.count == 0 ? source.min : std::min(target.min, source.min);
		target.count += source.count;
		target.sum_log += source.sum_log;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (state.count == 0) {
			return std::nullopt;
		}
		const double denominator = state.sum_log - state.count * std::log(state.min);
		if (!(denominator > 0)) {
			return std::nullopt;
		}
		return std::array<double, 2> {state.min, state.count / denominator};
	}
};

//! Gamma MLE: solve log(k) - digamma(k) = log(mean) - mean(log x) by Newton's method from Minka's starting point.
struct GammaFit : NoDestroy {
	static constexpr const char *name = "gamma";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> result_names = {"shape", "scale"};
	struct State {
		double count;
		double sum;
		double sum_log;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(name, x);
		RequireFitDomain(name, x > 0, "> 0", x);
		state.count += 1;
		state.sum += x;
		state.sum_log += std::log(x);
	}
	static void Combine(const State &source, State &target) {
		target.count += source.count;
		target.sum += source.sum;
		target.sum_log += source.sum_log;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (state.count < 2) {
			return std::nullopt;
		}
		const double mean = state.sum / state.count;
		const double s = std::log(mean) - state.sum_log / state.count;
		if (!(s > 1e-12)) {
			return std::nullopt;
		}
		double shape = (3 - s + std::sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s);
		for (idx_t i = 0; i < 100; i++) {
			const double f = std::log(shape) - boost::math::digamma(shape, StochasticPolicy()) - s;
			const double df = 1 / shape - boost::math::trigamma(shape, StochasticPolicy());
			const double next = shape - f / df;
			const bool converged = std::abs(next - shape) <= 1e-14 * shape;
			shape = next > 0 ? next : shape / 2;
			if (converged) {
				break;
			}
		}
		return std::array<double, 2> {shape, mean / shape};
	}
};

//! Beta MLE: Newton's method on the digamma equations, started from the method of moments estimate.
struct BetaFit : NoDestroy {
	static constexpr const char *name = "beta";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> result_names = {"alpha", "beta"};
	struct State {
		RunningMoments moments;
		double sum_log;
		double sum_log1m;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(name, x);
		RequireFitDomain(name, x > 0 && x < 1, "between 0 and 1 (exclusive)", x);
		state.moments.Add(x);
		state.sum_log += std::log(x);
		state.sum_log1m += std::log1p(-x);
	}
	static void Combine(const State &source, State &target) {
		target.moments.Merge(source.moments);
		target.sum_log += source.sum_log;
		target.sum_log1m += source.sum_log1m;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		const auto &m = state.moments;
		if (m.count < 2 || m.m2 <= 0) {
			return std::nullopt;
		}
		const double mean = m.mean;
		const double variance = m.m2 / m.count;
		const double common = mean * (1 - mean) / variance - 1;
		double a = common > 0 ? mean * common : 1;
		double b = common > 0 ? (1 - mean) * common : 1;
		const double target_a = state.sum_log / m.count;
		const double target_b = state.sum_log1m / m.count;
		const StochasticPolicy policy;
		for (idx_t i = 0; i < 200; i++) {
			const double psi_ab = boost::math::digamma(a + b, policy);
			const double g1 = boost::math::digamma(a, policy) - psi_ab - target_a;
			const double g2 = boost::math::digamma(b, policy) - psi_ab - target_b;
			const double t_ab = boost::math::trigamma(a + b, policy);
			const double j11 = boost::math::trigamma(a, policy) - t_ab;
			const double j22 = boost::math::trigamma(b, policy) - t_ab;
			const double j12 = -t_ab;
			const double det = j11 * j22 - j12 * j12;
			double step_a = (j22 * g1 - j12 * g2) / det;
			double step_b = (j11 * g2 - j12 * g1) / det;
			// Halve the step until both parameters stay positive
			while (a - step_a <= 0 || b - step_b <= 0) {
				step_a /= 2;
				step_b /= 2;
			}
			a -= step_a;
			b -= step_b;
			if (std::abs(step_a) <= 1e-14 * a && std::abs(step_b) <= 1e-14 * b) {
				break;
			}
		}
		return std::array<double, 2> {a, b};
	}
};

template <class OP>
void RegisterFit(ExtensionLoader &loader, const string &display_name, const string &method) {
	const string name = string("dist_") + OP::name + "_fit";
	string fields;
	for (auto field : OP::result_names) {
		fields += fields.empty() ? field : string(", ") + field;
	}
	RegisterStructAggregate<OP>(loader, name, {{"x", LogicalType::DOUBLE}},
	                            {"Fits a " + display_name + " distribution to x by " + method +
	                                 ". Returns a STRUCT with the fitted parameters (" + fields +
	                                 "), or NULL when there are too few or only identical values.",
	                             "SELECT " + name + "(x) FROM t", "fit", OP::name});
}

} // namespace

void Load_fit_functions(ExtensionLoader &loader) {
	const string mle = "maximum likelihood";
	RegisterFit<NormalFit>(loader, "normal", mle);
	RegisterFit<LognormalFit>(loader, "log-normal", mle);
	RegisterFit<ExponentialFit>(loader, "exponential", mle);
	RegisterFit<PoissonFit>(loader, "Poisson", mle);
	RegisterFit<BernoulliFit>(loader, "Bernoulli", mle);
	RegisterFit<GeometricFit>(loader, "geometric (failures before the first success)", mle);
	RegisterFit<RayleighFit>(loader, "Rayleigh", mle);
	RegisterFit<UniformRealFit>(loader, "continuous uniform", mle + " (the sample minimum and maximum)");
	RegisterFit<ParetoFit>(loader, "Pareto", mle);
	RegisterFit<GammaFit>(loader, "gamma", mle + " (Newton's method)");
	RegisterFit<BetaFit>(loader, "beta", mle + " (Newton's method)");
}

} // namespace duckdb
