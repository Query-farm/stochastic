#include "utils.hpp"

#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Maximum likelihood fits: dist_<name>_fit(x) returns the fitted parameters as a STRUCT whose fields are named
// like the distribution's parameters, followed by the log_likelihood, aic and bic of the fit. NULL inputs are
// ignored and an empty or degenerate sample returns NULL.
//
// A fit operation provides what RegisterStructAggregate needs (see aggregate_utils.hpp), with param_names in
// place of result_names and a Finalize that returns only the parameters, plus:
//
//   static double Count(const State &state);                                   // number of values
//   static double LogLikelihood(const State &state, const std::array<double, K> &params);
//===--------------------------------------------------------------------===//

constexpr double PI = boost::math::constants::pi<double>();
constexpr double EULER = boost::math::constants::euler<double>();

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

bool IsNonNegativeInteger(double x) {
	return x >= 0 && std::trunc(x) == x;
}

double LogGamma(double x) {
	return boost::math::lgamma(x, StochasticPolicy());
}

double Digamma(double x) {
	return boost::math::digamma(x, StochasticPolicy());
}

double Trigamma(double x) {
	return boost::math::trigamma(x, StochasticPolicy());
}

//! x * log(y), taking 0 * log(0) as 0.
double XLogY(double x, double y) {
	return x == 0 ? 0 : x * std::log(y);
}

struct FitBase {
	//! Parameters in the result that were given rather than estimated (not counted by the AIC and BIC)
	static constexpr idx_t fixed_parameters = 0;

	template <class STATE>
	static void Destroy(STATE &) {
	}
};

template <std::size_t K>
constexpr std::array<const char *, K + 3> FitResultNames(const std::array<const char *, K> &params) {
	std::array<const char *, K + 3> names {};
	for (std::size_t i = 0; i < K; i++) {
		names[i] = params[i];
	}
	names[K] = "log_likelihood";
	names[K + 1] = "aic";
	names[K + 2] = "bic";
	return names;
}

//! Adds the log-likelihood, AIC and BIC to the parameters of a fit operation.
template <class OP>
struct FitAggregate : OP {
	static constexpr std::size_t param_count = OP::param_names.size();
	static constexpr std::array<const char *, param_count + 3> result_names = FitResultNames(OP::param_names);

	static std::optional<std::array<double, param_count + 3>> Finalize(typename OP::State &state) {
		auto params = OP::Finalize(state);
		if (!params) {
			return std::nullopt;
		}
		const double log_likelihood = OP::LogLikelihood(state, *params);
		const double k = static_cast<double>(param_count - OP::fixed_parameters);
		std::array<double, param_count + 3> result;
		std::copy(params->begin(), params->end(), result.begin());
		result[param_count] = log_likelihood;
		result[param_count + 1] = 2 * k - 2 * log_likelihood;
		result[param_count + 2] = k * std::log(OP::Count(state)) - 2 * log_likelihood;
		for (auto value : result) {
			if (!std::isfinite(value)) {
				throw MathError(MathError::Kind::EVALUATION, "the fit is not finite; the values may be too large or "
				                                             "too small in magnitude");
			}
		}
		return result;
	}
};

//===--------------------------------------------------------------------===//
// Fits from sufficient statistics
//===--------------------------------------------------------------------===//

//! Fits using the mean and variance of g(x), where g is the identity or a transform such as log.
template <class Derived>
struct MomentsFit : FitBase {
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> param_names = {"mean", "stddev"};
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
	static double Count(const State &state) {
		return state.moments.count;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		const auto &m = state.moments;
		if (m.count < 2 || m.m2 <= 0) {
			return std::nullopt;
		}
		return std::array<double, 2> {m.mean, std::sqrt(m.m2 / m.count)};
	}
	//! Log-likelihood of the transformed values under the fitted normal distribution.
	static double NormalLogLikelihood(const State &state) {
		const auto &m = state.moments;
		return -m.count / 2 * (std::log(2 * PI * m.m2 / m.count) + 1);
	}
};

struct NormalFit : MomentsFit<NormalFit> {
	static constexpr const char *name = "normal";
	static double Transform(double x) {
		return x;
	}
	static double LogLikelihood(const State &state, const std::array<double, 2> &) {
		return NormalLogLikelihood(state);
	}
};

struct LognormalFit : MomentsFit<LognormalFit> {
	static constexpr const char *name = "lognormal";
	static double Transform(double x) {
		RequireFitDomain(name, x > 0, "> 0", x);
		return std::log(x);
	}
	static double LogLikelihood(const State &state, const std::array<double, 2> &) {
		// The density of x is the normal density of log(x) divided by x
		return NormalLogLikelihood(state) - state.moments.count * state.moments.mean;
	}
};

//! Fits whose estimate is a function of the sample mean (x must lie in the given domain). Aux(x) is summed
//! alongside x for the log-likelihood.
template <class Derived>
struct MeanFit : FitBase {
	using inputs = std::tuple<double>;
	struct State {
		double count;
		double sum;
		double sum_aux;
	};

	static double Aux(double) {
		return 0;
	}
	static void Update(State &state, double x) {
		RequireFiniteSample(Derived::name, x);
		Derived::CheckDomain(x);
		state.count += 1;
		state.sum += x;
		state.sum_aux += Derived::Aux(x);
	}
	static void Combine(const State &source, State &target) {
		target.count += source.count;
		target.sum += source.sum;
		target.sum_aux += source.sum_aux;
	}
	static double Count(const State &state) {
		return state.count;
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
	static constexpr std::array<const char *, 1> param_names = {"rate"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, x >= 0, ">= 0", x);
	}
	static std::optional<double> Estimate(double mean) {
		return mean > 0 ? std::optional<double>(1 / mean) : std::nullopt;
	}
	static double LogLikelihood(const State &state, const std::array<double, 1> &params) {
		return state.count * std::log(params[0]) - params[0] * state.sum;
	}
};

struct PoissonFit : MeanFit<PoissonFit> {
	static constexpr const char *name = "poisson";
	static constexpr std::array<const char *, 1> param_names = {"rate"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, IsNonNegativeInteger(x), "non-negative integers", x);
	}
	//! log(x!)
	static double Aux(double x) {
		return LogGamma(x + 1);
	}
	static std::optional<double> Estimate(double mean) {
		return mean > 0 ? std::optional<double>(mean) : std::nullopt;
	}
	static double LogLikelihood(const State &state, const std::array<double, 1> &params) {
		return state.sum * std::log(params[0]) - state.count * params[0] - state.sum_aux;
	}
};

struct BernoulliFit : MeanFit<BernoulliFit> {
	static constexpr const char *name = "bernoulli";
	static constexpr std::array<const char *, 1> param_names = {"prob"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, x == 0 || x == 1, "0 or 1", x);
	}
	static std::optional<double> Estimate(double mean) {
		return mean;
	}
	static double LogLikelihood(const State &state, const std::array<double, 1> &params) {
		return XLogY(state.sum, params[0]) + XLogY(state.count - state.sum, 1 - params[0]);
	}
};

//! Geometric distribution counting failures before the first success: p = 1 / (1 + mean).
struct GeometricFit : MeanFit<GeometricFit> {
	static constexpr const char *name = "geometric";
	static constexpr std::array<const char *, 1> param_names = {"prob"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, IsNonNegativeInteger(x), "non-negative integers", x);
	}
	static std::optional<double> Estimate(double mean) {
		return 1 / (1 + mean);
	}
	static double LogLikelihood(const State &state, const std::array<double, 1> &params) {
		return state.count * std::log(params[0]) + XLogY(state.sum, 1 - params[0]);
	}
};

//! Binomial with a known number of trials: prob = mean(x) / trials.
struct BinomialFit : FitBase {
	static constexpr const char *name = "binomial";
	using inputs = std::tuple<double, int64_t>;
	static constexpr std::array<const char *, 1> param_names = {"prob"};
	struct State {
		GroupParameters<1> trials;
		double count;
		double sum;
		//! Sum of log(trials choose x)
		double sum_log_choose;
	};
	static void Update(State &state, double x, int64_t trials) {
		RequireFiniteSample(name, x);
		RequireNonNegative(name, "trials", trials);
		state.trials.Check(name, trials);
		const double t = static_cast<double>(trials);
		RequireFitDomain(name, IsNonNegativeInteger(x) && x <= t, "integers between 0 and trials", x);
		state.count += 1;
		state.sum += x;
		state.sum_log_choose += LogGamma(t + 1) - LogGamma(x + 1) - LogGamma(t - x + 1);
	}
	static void Combine(const State &source, State &target) {
		target.trials.Merge(name, source.trials);
		target.count += source.count;
		target.sum += source.sum;
		target.sum_log_choose += source.sum_log_choose;
	}
	static double Count(const State &state) {
		return state.count;
	}
	static std::optional<std::array<double, 1>> Finalize(State &state) {
		if (state.count == 0 || state.trials.values[0] == 0) {
			return std::nullopt;
		}
		return std::array<double, 1> {state.sum / (state.count * state.trials.values[0])};
	}
	static double LogLikelihood(const State &state, const std::array<double, 1> &params) {
		const double failures = state.count * state.trials.values[0] - state.sum;
		return state.sum_log_choose + XLogY(state.sum, params[0]) + XLogY(failures, 1 - params[0]);
	}
};

struct RayleighFit : FitBase {
	static constexpr const char *name = "rayleigh";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"scale"};
	struct State {
		double count;
		double sum_squares;
		double sum_log;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(name, x);
		RequireFitDomain(name, x >= 0, ">= 0", x);
		state.count += 1;
		state.sum_squares += x * x;
		state.sum_log += std::log(x);
	}
	static void Combine(const State &source, State &target) {
		target.count += source.count;
		target.sum_squares += source.sum_squares;
		target.sum_log += source.sum_log;
	}
	static double Count(const State &state) {
		return state.count;
	}
	static std::optional<std::array<double, 1>> Finalize(State &state) {
		if (state.count == 0 || state.sum_squares == 0) {
			return std::nullopt;
		}
		return std::array<double, 1> {std::sqrt(state.sum_squares / (2 * state.count))};
	}
	static double LogLikelihood(const State &state, const std::array<double, 1> &params) {
		const double variance = params[0] * params[0];
		return state.sum_log - state.count * std::log(variance) - state.sum_squares / (2 * variance);
	}
};

struct UniformRealFit : FitBase {
	static constexpr const char *name = "uniform_real";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> param_names = {"min", "max"};
	struct State {
		double count;
		double min;
		double max;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(name, x);
		if (state.count == 0) {
			state = {1, x, x};
			return;
		}
		state.count += 1;
		state.min = std::min(state.min, x);
		state.max = std::max(state.max, x);
	}
	static void Combine(const State &source, State &target) {
		if (source.count == 0) {
			return;
		}
		if (target.count == 0) {
			target = source;
			return;
		}
		target.count += source.count;
		target.min = std::min(target.min, source.min);
		target.max = std::max(target.max, source.max);
	}
	static double Count(const State &state) {
		return state.count;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (state.count == 0 || !(state.min < state.max)) {
			return std::nullopt;
		}
		return std::array<double, 2> {state.min, state.max};
	}
	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		return -state.count * std::log(params[1] - params[0]);
	}
};

//! scale = min(x), shape = n / sum(log(x / scale)).
struct ParetoFit : FitBase {
	static constexpr const char *name = "pareto";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> param_names = {"scale", "shape"};
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
	static double Count(const State &state) {
		return state.count;
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
	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		const double scale = params[0];
		const double shape = params[1];
		return state.count * (std::log(shape) + shape * std::log(scale)) - (shape + 1) * state.sum_log;
	}
};

//! Solves log(k) - digamma(k) = s for the gamma shape k by Newton's method from Minka's starting point.
double SolveGammaShape(double s) {
	double shape = (3 - s + std::sqrt((s - 3) * (s - 3) + 24 * s)) / (12 * s);
	for (idx_t i = 0; i < 100; i++) {
		const double f = std::log(shape) - Digamma(shape) - s;
		const double df = 1 / shape - Trigamma(shape);
		const double next = shape - f / df;
		const bool converged = std::abs(next - shape) <= 1e-14 * shape;
		shape = next > 0 ? next : shape / 2;
		if (converged) {
			break;
		}
	}
	return shape;
}

double GammaLogLikelihood(double count, double sum, double sum_log, double shape, double scale) {
	return (shape - 1) * sum_log - sum / scale - count * (shape * std::log(scale) + LogGamma(shape));
}

//! Sums of y and log(y), where y is x for the gamma fit and 1 / x for the inverse gamma fit.
template <class Derived>
struct GammaFamilyFit : FitBase {
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> param_names = {"shape", "scale"};
	struct State {
		double count;
		double sum;
		double sum_log;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(Derived::name, x);
		RequireFitDomain(Derived::name, x > 0, "> 0", x);
		const double y = Derived::Transform(x);
		state.count += 1;
		state.sum += y;
		state.sum_log += std::log(y);
	}
	static void Combine(const State &source, State &target) {
		target.count += source.count;
		target.sum += source.sum;
		target.sum_log += source.sum_log;
	}
	static double Count(const State &state) {
		return state.count;
	}
	//! Shape and scale of the gamma distribution of y.
	static std::optional<std::array<double, 2>> FitGamma(const State &state) {
		if (state.count < 2) {
			return std::nullopt;
		}
		const double mean = state.sum / state.count;
		const double s = std::log(mean) - state.sum_log / state.count;
		if (!(s > 1e-12)) {
			return std::nullopt;
		}
		const double shape = SolveGammaShape(s);
		return std::array<double, 2> {shape, mean / shape};
	}
};

struct GammaFit : GammaFamilyFit<GammaFit> {
	static constexpr const char *name = "gamma";
	static double Transform(double x) {
		return x;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		return FitGamma(state);
	}
	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		return GammaLogLikelihood(state.count, state.sum, state.sum_log, params[0], params[1]);
	}
};

//! 1 / x follows a gamma distribution whose rate is the inverse gamma scale.
struct InverseGammaFit : GammaFamilyFit<InverseGammaFit> {
	static constexpr const char *name = "inverse_gamma";
	static double Transform(double x) {
		return 1 / x;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		auto fit = FitGamma(state);
		if (!fit) {
			return std::nullopt;
		}
		return std::array<double, 2> {(*fit)[0], 1 / (*fit)[1]};
	}
	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		// The density of x is the gamma density of 1 / x divided by x^2, and sum_log is the sum of log(1 / x)
		return GammaLogLikelihood(state.count, state.sum, state.sum_log, params[0], 1 / params[1]) + 2 * state.sum_log;
	}
};

//! Chi-squared degrees of freedom k: solve digamma(k / 2) = mean(log x) - log(2).
struct ChiSquaredFit : FitBase {
	static constexpr const char *name = "chi_squared";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"degrees_of_freedom"};
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
	static double Count(const State &state) {
		return state.count;
	}
	//! Inverse of the digamma function by Newton's method from Minka's starting point.
	static double InverseDigamma(double target) {
		double y = target >= -2.22 ? std::exp(target) + 0.5 : -1 / (target + EULER);
		for (idx_t i = 0; i < 100; i++) {
			const double next = y - (Digamma(y) - target) / Trigamma(y);
			const bool converged = std::abs(next - y) <= 1e-14 * y;
			y = next > 0 ? next : y / 2;
			if (converged) {
				break;
			}
		}
		return y;
	}
	static std::optional<std::array<double, 1>> Finalize(State &state) {
		if (state.count == 0) {
			return std::nullopt;
		}
		return std::array<double, 1> {2 * InverseDigamma(state.sum_log / state.count - std::log(2.0))};
	}
	static double LogLikelihood(const State &state, const std::array<double, 1> &params) {
		const double half = params[0] / 2;
		return (half - 1) * state.sum_log - state.sum / 2 - state.count * (half * std::log(2.0) + LogGamma(half));
	}
};

//! mean = mean(x), 1 / shape = mean(1 / x - 1 / mean).
struct InverseGaussianFit : FitBase {
	static constexpr const char *name = "inverse_gaussian";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> param_names = {"mean", "shape"};
	struct State {
		double count;
		double sum;
		double sum_inverse;
		double sum_log;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(name, x);
		RequireFitDomain(name, x > 0, "> 0", x);
		state.count += 1;
		state.sum += x;
		state.sum_inverse += 1 / x;
		state.sum_log += std::log(x);
	}
	static void Combine(const State &source, State &target) {
		target.count += source.count;
		target.sum += source.sum;
		target.sum_inverse += source.sum_inverse;
		target.sum_log += source.sum_log;
	}
	static double Count(const State &state) {
		return state.count;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (state.count < 2) {
			return std::nullopt;
		}
		const double mean = state.sum / state.count;
		const double denominator = state.sum_inverse - state.count / mean;
		if (!(denominator > 1e-14 * state.sum_inverse)) {
			return std::nullopt;
		}
		return std::array<double, 2> {mean, state.count / denominator};
	}
	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		const double mean = params[0];
		const double shape = params[1];
		// sum((x - mean)^2 / x)
		const double deviance = state.sum - 2 * state.count * mean + mean * mean * state.sum_inverse;
		return state.count / 2 * std::log(shape / (2 * PI)) - 1.5 * state.sum_log -
		       shape / (2 * mean * mean) * deviance;
	}
};

//! Beta MLE: Newton's method on the digamma equations, started from the method of moments estimate.
struct BetaFit : FitBase {
	static constexpr const char *name = "beta";
	using inputs = std::tuple<double>;
	static constexpr std::array<const char *, 2> param_names = {"alpha", "beta"};
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
	static double Count(const State &state) {
		return state.moments.count;
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
		for (idx_t i = 0; i < 200; i++) {
			const double psi_ab = Digamma(a + b);
			const double g1 = Digamma(a) - psi_ab - target_a;
			const double g2 = Digamma(b) - psi_ab - target_b;
			const double t_ab = Trigamma(a + b);
			const double j11 = Trigamma(a) - t_ab;
			const double j22 = Trigamma(b) - t_ab;
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
	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		const double a = params[0];
		const double b = params[1];
		const double log_beta = LogGamma(a) + LogGamma(b) - LogGamma(a + b);
		return (a - 1) * state.sum_log + (b - 1) * state.sum_log1m - state.moments.count * log_beta;
	}
};

//===--------------------------------------------------------------------===//
// Fits that need the whole sample. Finalize may reorder the collected values.
//===--------------------------------------------------------------------===//

template <class Derived>
struct CollectedFit : FitBase {
	using inputs = std::tuple<double>;
	struct State {
		CollectedValues values;
	};
	static void Update(State &state, double x) {
		RequireFiniteSample(Derived::name, x);
		Derived::CheckDomain(x);
		state.values.Add(x);
	}
	static void CheckDomain(double) {
	}
	static void Combine(const State &source, State &target) {
		target.values.Merge(source.values);
	}
	static void Destroy(State &state) {
		state.values.Destroy();
	}
	static double Count(const State &state) {
		return static_cast<double>(state.values.size());
	}
};

double SortedMedian(const vector<double> &sorted) {
	const auto n = sorted.size();
	return n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

//! location = median(x), scale = mean(|x - location|).
struct LaplaceFit : CollectedFit<LaplaceFit> {
	static constexpr const char *name = "laplace";
	static constexpr std::array<const char *, 2> param_names = {"location", "scale"};
	static double SumAbsoluteDeviation(const vector<double> &xs, double location) {
		double sum = 0;
		for (auto x : xs) {
			sum += std::abs(x - location);
		}
		return sum;
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (state.values.size() < 2) {
			return std::nullopt;
		}
		auto &xs = *state.values.values;
		std::sort(xs.begin(), xs.end());
		const double location = SortedMedian(xs);
		const double scale = SumAbsoluteDeviation(xs, location) / static_cast<double>(xs.size());
		if (!(scale > 0)) {
			return std::nullopt;
		}
		return std::array<double, 2> {location, scale};
	}
	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		const auto &xs = *state.values.values;
		return -static_cast<double>(xs.size()) * std::log(2 * params[1]) -
		       SumAbsoluteDeviation(xs, params[0]) / params[1];
	}
};

//! log f(z) for a standardized location-scale density and its first two derivatives.
struct StandardLogDensity {
	double value;
	double first;
	double second;
};

//! Location-scale families with density f((x - location) / scale) / scale. Derived provides
//! Evaluate(z) -> StandardLogDensity and Start(sorted x) -> {location, scale}. The likelihood is maximized by
//! Newton's method with a line search, falling back to gradient ascent where the Hessian is not negative definite.
template <class Derived>
struct LocationScaleFit : CollectedFit<Derived> {
	using State = typename CollectedFit<Derived>::State;
	static constexpr std::array<const char *, 2> param_names = {"location", "scale"};

	static double SumLogDensity(const vector<double> &xs, double location, double scale) {
		double sum = -static_cast<double>(xs.size()) * std::log(scale);
		for (auto x : xs) {
			sum += Derived::Evaluate((x - location) / scale).value;
		}
		return sum;
	}

	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (state.values.size() < 2) {
			return std::nullopt;
		}
		auto &xs = *state.values.values;
		std::sort(xs.begin(), xs.end());
		if (xs.front() == xs.back()) {
			return std::nullopt;
		}
		auto start = Derived::Start(xs);
		double location = start[0];
		double scale = start[1];
		const double n = static_cast<double>(xs.size());
		double log_likelihood = SumLogDensity(xs, location, scale);
		for (idx_t iteration = 0; iteration < 500; iteration++) {
			// Gradient and Hessian of the log-likelihood in (location, scale)
			double g1 = 0, g2 = -n, h11 = 0, h12 = 0, h22 = n;
			for (auto x : xs) {
				const double z = (x - location) / scale;
				const auto d = Derived::Evaluate(z);
				g1 -= d.first;
				g2 -= z * d.first;
				h11 += d.second;
				h12 += d.first + z * d.second;
				h22 += 2 * z * d.first + z * z * d.second;
			}
			g1 /= scale;
			g2 /= scale;
			const double s2 = scale * scale;
			h11 /= s2;
			h12 /= s2;
			h22 /= s2;
			const double det = h11 * h22 - h12 * h12;
			double step_location, step_scale;
			if (h11 < 0 && det > 0) {
				step_location = -(h22 * g1 - h12 * g2) / det;
				step_scale = -(h11 * g2 - h12 * g1) / det;
			} else {
				step_location = g1 * s2 / n;
				step_scale = g2 * s2 / n;
			}
			bool improved = false;
			for (idx_t halving = 0; halving < 60; halving++) {
				const double next_location = location + step_location;
				const double next_scale = scale + step_scale;
				if (next_scale > 0) {
					const double next_log_likelihood = SumLogDensity(xs, next_location, next_scale);
					if (next_log_likelihood >= log_likelihood) {
						location = next_location;
						scale = next_scale;
						log_likelihood = next_log_likelihood;
						improved = true;
						break;
					}
				}
				step_location /= 2;
				step_scale /= 2;
			}
			const bool converged = std::abs(step_location) <= 1e-13 * (std::abs(location) + scale) &&
			                       std::abs(step_scale) <= 1e-13 * scale;
			if (!improved || converged) {
				break;
			}
		}
		return std::array<double, 2> {location, scale};
	}

	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		return SumLogDensity(*state.values.values, params[0], params[1]);
	}

	//! Population mean and standard deviation, used for starting values.
	static std::array<double, 2> MeanAndStddev(const vector<double> &xs) {
		RunningMoments moments {};
		for (auto x : xs) {
			moments.Add(x);
		}
		return {moments.mean, std::sqrt(moments.m2 / moments.count)};
	}
};

struct LogisticFit : LocationScaleFit<LogisticFit> {
	static constexpr const char *name = "logistic";
	//! log f(z) = -|z| - 2 log(1 + exp(-|z|))
	static StandardLogDensity Evaluate(double z) {
		const double a = std::abs(z);
		const double t = std::tanh(z / 2);
		return {-a - 2 * std::log1p(std::exp(-a)), -t, -(1 - t * t) / 2};
	}
	static std::array<double, 2> Start(const vector<double> &xs) {
		auto moments = MeanAndStddev(xs);
		return {moments[0], moments[1] * std::sqrt(3.0) / PI};
	}
};

struct CauchyFit : LocationScaleFit<CauchyFit> {
	static constexpr const char *name = "cauchy";
	//! log f(z) = -log(pi) - log(1 + z^2)
	static StandardLogDensity Evaluate(double z) {
		const double q = 1 + z * z;
		return {-std::log(PI) - std::log1p(z * z), -2 * z / q, -2 * (1 - z * z) / (q * q)};
	}
	//! The median and half the interquartile range.
	static std::array<double, 2> Start(const vector<double> &xs) {
		const auto n = xs.size();
		double scale = (xs[(3 * (n - 1)) / 4] - xs[(n - 1) / 4]) / 2;
		if (!(scale > 0)) {
			scale = (xs.back() - xs.front()) / 2;
		}
		return {SortedMedian(xs), scale};
	}
};

//! The (maximum) extreme value or Gumbel distribution.
struct ExtremeValueFit : LocationScaleFit<ExtremeValueFit> {
	static constexpr const char *name = "extreme_value";
	//! log f(z) = -z - exp(-z)
	static StandardLogDensity Evaluate(double z) {
		const double e = std::exp(-z);
		return {-z - e, e - 1, -e};
	}
	static std::array<double, 2> Start(const vector<double> &xs) {
		auto moments = MeanAndStddev(xs);
		const double scale = moments[1] * std::sqrt(6.0) / PI;
		return {moments[0] - EULER * scale, scale};
	}
};

//! Weibull MLE: the shape k solves sum(x^k log x) / sum(x^k) - 1 / k = mean(log x), found by Newton's method
//! safeguarded with bisection; then scale = mean(x^k)^(1 / k).
struct WeibullFit : CollectedFit<WeibullFit> {
	static constexpr const char *name = "weibull";
	static constexpr std::array<const char *, 2> param_names = {"shape", "scale"};
	static void CheckDomain(double x) {
		RequireFitDomain(name, x > 0, "> 0", x);
	}
	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (state.values.size() < 2) {
			return std::nullopt;
		}
		// Work with u = log(x) - max(log(x)) so that exp(k u) cannot overflow
		vector<double> u;
		u.reserve(state.values.size());
		for (auto x : *state.values.values) {
			u.push_back(std::log(x));
		}
		const double offset = *std::max_element(u.begin(), u.end());
		RunningMoments moments {};
		for (auto &value : u) {
			value -= offset;
			moments.Add(value);
		}
		if (!(moments.m2 > 0)) {
			return std::nullopt;
		}
		const double n = moments.count;
		double low = 0;
		double high = std::numeric_limits<double>::infinity();
		double shape = 1.2 / std::sqrt(moments.m2 / n);
		for (idx_t i = 0; i < 200; i++) {
			double s0 = 0, s1 = 0, s2 = 0;
			for (auto value : u) {
				const double w = std::exp(shape * value);
				s0 += w;
				s1 += w * value;
				s2 += w * value * value;
			}
			const double weighted_mean = s1 / s0;
			const double f = weighted_mean - 1 / shape - moments.mean;
			const double df = s2 / s0 - weighted_mean * weighted_mean + 1 / (shape * shape);
			if (f == 0) {
				break;
			}
			if (f < 0) {
				low = shape;
			} else {
				high = shape;
			}
			double next = shape - f / df;
			if (!(next > low && next < high)) {
				next = std::isfinite(high) ? (low + high) / 2 : 2 * shape;
			}
			const bool converged = std::abs(next - shape) <= 1e-14 * shape;
			shape = next;
			if (converged) {
				break;
			}
		}
		double mean_power = 0;
		for (auto value : u) {
			mean_power += std::exp(shape * value);
		}
		mean_power /= n;
		return std::array<double, 2> {shape, std::exp(offset + std::log(mean_power) / shape)};
	}
	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		const double shape = params[0];
		const double log_scale = std::log(params[1]);
		double result = static_cast<double>(state.values.size()) * (std::log(shape) - log_scale);
		for (auto x : *state.values.values) {
			const double z = std::log(x) - log_scale;
			result += (shape - 1) * z - std::exp(shape * z);
		}
		return result;
	}
};

//! Degrees of freedom of the standard Student's t distribution, by golden section search over log(df).
struct StudentsTFit : CollectedFit<StudentsTFit> {
	static constexpr const char *name = "students_t";
	static constexpr std::array<const char *, 1> param_names = {"degrees_of_freedom"};
	static constexpr double MIN_LOG_DF = -7;
	static constexpr double MAX_LOG_DF = 18;

	static double SumLogDensity(const vector<double> &xs, double df) {
		const double n = static_cast<double>(xs.size());
		double sum = n * (LogGamma((df + 1) / 2) - LogGamma(df / 2) - std::log(df * PI) / 2);
		for (auto x : xs) {
			sum -= (df + 1) / 2 * std::log1p(x * x / df);
		}
		return sum;
	}
	static std::optional<std::array<double, 1>> Finalize(State &state) {
		if (state.values.size() < 2) {
			return std::nullopt;
		}
		const auto &xs = *state.values.values;
		const double ratio = (std::sqrt(5.0) - 1) / 2;
		double a = MIN_LOG_DF;
		double b = MAX_LOG_DF;
		double c = b - ratio * (b - a);
		double d = a + ratio * (b - a);
		double fc = SumLogDensity(xs, std::exp(c));
		double fd = SumLogDensity(xs, std::exp(d));
		while (b - a > 1e-10) {
			if (fc > fd) {
				b = d;
				d = c;
				fd = fc;
				c = b - ratio * (b - a);
				fc = SumLogDensity(xs, std::exp(c));
			} else {
				a = c;
				c = d;
				fc = fd;
				d = a + ratio * (b - a);
				fd = SumLogDensity(xs, std::exp(d));
			}
		}
		const double df = std::exp((a + b) / 2);
		// For large df the log-likelihood is that of the standard normal plus sum(x^4 - 2 x^2 - 1) / (4 df). When the
		// sum is not positive it approaches the normal limit from below, so unless an interior maximum beats the
		// limit the degrees of freedom are unbounded.
		double tail_sum = 0;
		double normal_log_likelihood = 0;
		for (auto x : xs) {
			tail_sum += x * x * x * x - 2 * x * x - 1;
			normal_log_likelihood -= x * x / 2 + std::log(2 * PI) / 2;
		}
		if (tail_sum <= 0 && SumLogDensity(xs, df) <= normal_log_likelihood + 1e-9 * std::abs(normal_log_likelihood)) {
			return std::nullopt;
		}
		return std::array<double, 1> {df};
	}
	static double LogLikelihood(const State &state, const std::array<double, 1> &params) {
		return SumLogDensity(*state.values.values, params[0]);
	}
};

//! Negative binomial (failures before the given number of successes) with a whole number of successes, as the
//! distribution functions require: the continuous maximum of the profile likelihood is found by bisection and the
//! better of the neighbouring integers is returned. Samples without overdispersion (variance <= mean) give NULL.
struct NegativeBinomialFit : CollectedFit<NegativeBinomialFit> {
	static constexpr const char *name = "negative_binomial";
	static constexpr std::array<const char *, 2> param_names = {"successes", "prob"};
	static constexpr double MAX_SUCCESSES = 9007199254740992.0;

	static void CheckDomain(double x) {
		RequireFitDomain(name, IsNonNegativeInteger(x), "non-negative integers", x);
	}

	//! The distinct values and how often each occurs.
	struct Counts {
		vector<std::pair<double, double>> values;
		double n;
		double mean;
	};

	static Counts Tabulate(vector<double> &xs) {
		std::sort(xs.begin(), xs.end());
		Counts counts {{}, static_cast<double>(xs.size()), 0};
		for (auto x : xs) {
			if (counts.values.empty() || counts.values.back().first != x) {
				counts.values.emplace_back(x, 0);
			}
			counts.values.back().second += 1;
			counts.mean += x;
		}
		counts.mean /= counts.n;
		return counts;
	}

	//! Derivative of the profile log-likelihood with respect to the number of successes.
	static double Score(const Counts &counts, double successes) {
		double sum = 0;
		for (auto &value : counts.values) {
			sum += value.second * (Digamma(value.first + successes) - Digamma(successes));
		}
		return sum + counts.n * std::log(successes / (successes + counts.mean));
	}

	static double ProfileLogLikelihood(const Counts &counts, double successes) {
		const double prob = successes / (successes + counts.mean);
		double sum = counts.n * successes * std::log(prob) + XLogY(counts.n * counts.mean, 1 - prob);
		for (auto &value : counts.values) {
			sum += value.second * (LogGamma(value.first + successes) - LogGamma(successes) - LogGamma(value.first + 1));
		}
		return sum;
	}

	static std::optional<std::array<double, 2>> Finalize(State &state) {
		if (state.values.size() < 2) {
			return std::nullopt;
		}
		const auto counts = Tabulate(*state.values.values);
		double m2 = 0;
		for (auto &value : counts.values) {
			m2 += value.second * (value.first - counts.mean) * (value.first - counts.mean);
		}
		if (!(m2 / counts.n > counts.mean)) {
			return std::nullopt;
		}
		// The score is positive below the continuous maximum and negative above it
		double low = 1;
		double high = 1;
		if (Score(counts, 1) > 0) {
			while (Score(counts, high) > 0) {
				low = high;
				high *= 2;
				if (high > MAX_SUCCESSES) {
					return std::nullopt;
				}
			}
			while (high - low > 0.5) {
				const double mid = (low + high) / 2;
				(Score(counts, mid) > 0 ? low : high) = mid;
			}
		}
		double successes = std::max(1.0, std::floor(low));
		for (double candidate = successes + 1; candidate <= std::ceil(high); candidate++) {
			if (ProfileLogLikelihood(counts, candidate) > ProfileLogLikelihood(counts, successes)) {
				successes = candidate;
			}
		}
		return std::array<double, 2> {successes, successes / (successes + counts.mean)};
	}

	static double LogLikelihood(const State &state, const std::array<double, 2> &params) {
		auto &xs = *state.values.values;
		return ProfileLogLikelihood(Tabulate(xs), params[0]);
	}
};

template <class OP>
void RegisterFit(ExtensionLoader &loader, const string &display_name, const string &method,
                 const string &notes = string(), const vector<std::pair<string, LogicalType>> &extra_params = {},
                 const string &extra_example = string()) {
	const string name = string("dist_") + OP::name + "_fit";
	string fields;
	for (auto field : OP::param_names) {
		fields += fields.empty() ? field : string(", ") + field;
	}
	vector<std::pair<string, LogicalType>> params {{"x", LogicalType::DOUBLE}};
	params.insert(params.end(), extra_params.begin(), extra_params.end());
	RegisterStructAggregate<FitAggregate<OP>>(
	    loader, name, params,
	    {"Fits a " + display_name + " distribution to x by " + method +
	         ". Returns a STRUCT with the fitted parameters (" + fields +
	         ") followed by the log_likelihood, aic and bic of the fit, or NULL when there are too few or only "
	         "identical values." +
	         (notes.empty() ? "" : " " + notes),
	     "SELECT " + name + "(x" + extra_example + ") FROM t", "fit", OP::name});
}

} // namespace

void Load_fit_functions(ExtensionLoader &loader) {
	const string mle = "maximum likelihood";
	const string newton = mle + " (Newton's method)";
	const string whole_sample = "The whole sample is held in memory.";
	RegisterFit<NormalFit>(loader, "normal", mle);
	RegisterFit<LognormalFit>(loader, "log-normal", mle);
	RegisterFit<ExponentialFit>(loader, "exponential", mle);
	RegisterFit<PoissonFit>(loader, "Poisson", mle);
	RegisterFit<BernoulliFit>(loader, "Bernoulli", mle);
	RegisterFit<GeometricFit>(loader, "geometric (failures before the first success)", mle);
	RegisterFit<BinomialFit>(loader, "binomial", mle,
	                         "trials is the known number of trials and must be the same for every row of a group.",
	                         {{"trials", LogicalType::BIGINT}}, ", 10");
	RegisterFit<RayleighFit>(loader, "Rayleigh", mle);
	RegisterFit<UniformRealFit>(loader, "continuous uniform", mle + " (the sample minimum and maximum)");
	RegisterFit<ParetoFit>(loader, "Pareto", mle);
	RegisterFit<GammaFit>(loader, "gamma", newton);
	RegisterFit<InverseGammaFit>(loader, "inverse gamma", newton);
	RegisterFit<ChiSquaredFit>(loader, "chi-squared", newton);
	RegisterFit<InverseGaussianFit>(loader, "inverse Gaussian", mle);
	RegisterFit<BetaFit>(loader, "beta", newton);
	RegisterFit<LaplaceFit>(loader, "Laplace", mle + " (the median and the mean absolute deviation from it)",
	                        whole_sample);
	RegisterFit<LogisticFit>(loader, "logistic", newton, whole_sample);
	RegisterFit<CauchyFit>(loader, "Cauchy", newton, whole_sample);
	RegisterFit<ExtremeValueFit>(loader, "extreme value (Gumbel)", newton, whole_sample);
	RegisterFit<WeibullFit>(loader, "Weibull", newton, whole_sample);
	RegisterFit<StudentsTFit>(
	    loader, "standard Student's t", mle + " (golden section search)",
	    "Returns NULL when the tails are no heavier than normal, where the degrees of freedom are "
	    "unbounded. " +
	        whole_sample);
	RegisterFit<NegativeBinomialFit>(loader, "negative binomial", mle,
	                                 "successes is restricted to whole numbers, as the distribution functions require; "
	                                 "cast it to BIGINT to use it. Returns NULL when the variance does not exceed the "
	                                 "mean. " +
	                                     whole_sample);
}

} // namespace duckdb
