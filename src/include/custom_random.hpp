#pragma once
#include <random>
#include <cmath>

namespace custom_random {

template <typename T = double>
class beta_distribution {
public:
	using result_type = T;

	beta_distribution(T alpha, T beta) : gamma_a_(alpha, T(1)), gamma_b_(beta, T(1)) {
	}

	template <typename Generator>
	result_type operator()(Generator &g) {
		T x, y;
		do {
			x = gamma_a_(g);
			y = gamma_b_(g);
		} while (x == T(0) && y == T(0));
		return x / (x + y);
	}

private:
	std::gamma_distribution<T> gamma_a_;
	std::gamma_distribution<T> gamma_b_;
};

template <typename T = double>
class laplace_distribution {
public:
	using result_type = T;

	laplace_distribution(T location, T scale) : location_(location), scale_(scale), uniform_(T(0), T(1)) {
	}

	template <typename Generator>
	result_type operator()(Generator &g) {
		T raw;
		do {
			raw = uniform_(g);
		} while (raw == T(0));
		T u = raw - T(0.5);
		return location_ - scale_ * std::copysign(T(1), u) * std::log1p(T(-2) * std::abs(u));
	}

private:
	T location_;
	T scale_;
	std::uniform_real_distribution<T> uniform_;
};

template <typename T = double>
class logistic_distribution {
public:
	using result_type = T;

	logistic_distribution(T location, T scale) : location_(location), scale_(scale), uniform_(T(0), T(1)) {
	}

	template <typename Generator>
	result_type operator()(Generator &g) {
		T u;
		do {
			u = uniform_(g);
		} while (u == T(0) || u == T(1));
		return location_ + scale_ * std::log(u / (T(1) - u));
	}

private:
	T location_;
	T scale_;
	std::uniform_real_distribution<T> uniform_;
};

template <typename T = double>
class pareto_distribution {
public:
	using result_type = T;

	pareto_distribution(T scale, T shape) : scale_(scale), shape_(shape), uniform_(T(0), T(1)) {
	}

	template <typename Generator>
	result_type operator()(Generator &g) {
		T u;
		do {
			u = uniform_(g);
		} while (u == T(0));
		return scale_ / std::pow(u, T(1) / shape_);
	}

private:
	T scale_;
	T shape_;
	std::uniform_real_distribution<T> uniform_;
};

template <typename T = double>
class rayleigh_distribution {
public:
	using result_type = T;

	rayleigh_distribution(T sigma) : sigma_(sigma), uniform_(T(0), T(1)) {
	}

	template <typename Generator>
	result_type operator()(Generator &g) {
		T u;
		do {
			u = uniform_(g);
		} while (u == T(0));
		return sigma_ * std::sqrt(T(-2) * std::log(u));
	}

private:
	T sigma_;
	std::uniform_real_distribution<T> uniform_;
};

} // namespace custom_random
