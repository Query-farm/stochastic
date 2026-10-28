#pragma once

#include "duckdb/common/exception.hpp"
#include <boost/math/distributions/complement.hpp>
#include <cmath>
#include <cstdint>
#include <random>

namespace duckdb {

//! Draws a uniform value in the open interval (0, 1).
template <class Generator>
double OpenUniform(Generator &g) {
	std::uniform_real_distribution<double> uniform(0.0, 1.0);
	double u;
	do {
		u = uniform(g);
	} while (u == 0.0);
	return u;
}

//! Samples any distribution by inverting its CDF: quantile(U) with U ~ Uniform(0, 1). Used where no direct
//! sampling algorithm is available; discrete distributions must round quantiles up to an integer.
template <class Dist, class T = double>
class InverseTransformSampler {
public:
	using result_type = T;

	template <class... Args>
	explicit InverseTransformSampler(Args... args) : dist(args...) {
	}

	template <class Generator>
	T operator()(Generator &g) {
		return static_cast<T>(boost::math::quantile(dist, OpenUniform(g)));
	}

private:
	Dist dist;
};

//! Chi-squared variate with possibly fractional degrees of freedom k: Gamma(k / 2, 2).
template <class Generator>
double ChiSquaredVariate(Generator &g, double degrees_of_freedom) {
	std::gamma_distribution<double> gamma(degrees_of_freedom / 2.0, 2.0);
	return gamma(g);
}

//! Non-central chi-squared as a Poisson mixture of central chi-squared variates: chi2(k + 2J), J ~ Poisson(lambda/2).
template <class Generator>
double NonCentralChiSquaredVariate(Generator &g, double degrees_of_freedom, double noncentrality) {
	if (noncentrality == 0) {
		return ChiSquaredVariate(g, degrees_of_freedom);
	}
	std::poisson_distribution<int64_t> poisson(noncentrality / 2.0);
	return ChiSquaredVariate(g, degrees_of_freedom + 2.0 * static_cast<double>(poisson(g)));
}

//! The Poisson mixture needs a Poisson sample with mean lambda / 2, which must stay well inside the DOUBLE
//! integer range.
inline void CheckNoncentralityForSampling(const char *dist, double noncentrality) {
	if (!(noncentrality <= 1e15)) {
		throw InvalidInputException("%s: noncentrality is too large to sample, must be <= 1e15", dist);
	}
}

class NonCentralChiSquaredSampler {
public:
	using result_type = double;

	NonCentralChiSquaredSampler(double degrees_of_freedom, double noncentrality)
	    : degrees_of_freedom(degrees_of_freedom), noncentrality(noncentrality) {
		CheckNoncentralityForSampling("non_central_chi_squared", noncentrality);
	}

	template <class Generator>
	double operator()(Generator &g) {
		return NonCentralChiSquaredVariate(g, degrees_of_freedom, noncentrality);
	}

private:
	double degrees_of_freedom;
	double noncentrality;
};

//! (Z + delta) / sqrt(V / k) with Z ~ N(0, 1) and V ~ chi2(k).
class NonCentralTSampler {
public:
	using result_type = double;

	NonCentralTSampler(double degrees_of_freedom, double noncentrality)
	    : degrees_of_freedom(degrees_of_freedom), noncentrality(noncentrality) {
	}

	template <class Generator>
	double operator()(Generator &g) {
		std::normal_distribution<double> normal(0.0, 1.0);
		const double z = normal(g);
		return (z + noncentrality) / std::sqrt(ChiSquaredVariate(g, degrees_of_freedom) / degrees_of_freedom);
	}

private:
	double degrees_of_freedom;
	double noncentrality;
};

//! (X / d1) / (Y / d2) with X ~ non-central chi2(d1, lambda) and Y ~ chi2(d2).
class NonCentralFSampler {
public:
	using result_type = double;

	NonCentralFSampler(double d1, double d2, double noncentrality) : d1(d1), d2(d2), noncentrality(noncentrality) {
		CheckNoncentralityForSampling("non_central_f", noncentrality);
	}

	template <class Generator>
	double operator()(Generator &g) {
		const double x = NonCentralChiSquaredVariate(g, d1, noncentrality);
		return (x / d1) / (ChiSquaredVariate(g, d2) / d2);
	}

private:
	double d1;
	double d2;
	double noncentrality;
};

//! X / (X + Y) with X ~ non-central chi2(2 alpha, lambda) and Y ~ chi2(2 beta).
class NonCentralBetaSampler {
public:
	using result_type = double;

	NonCentralBetaSampler(double alpha, double beta, double noncentrality)
	    : alpha(alpha), beta(beta), noncentrality(noncentrality) {
		CheckNoncentralityForSampling("non_central_beta", noncentrality);
	}

	template <class Generator>
	double operator()(Generator &g) {
		double x, y;
		do {
			x = NonCentralChiSquaredVariate(g, 2.0 * alpha, noncentrality);
			y = ChiSquaredVariate(g, 2.0 * beta);
		} while (x == 0.0 && y == 0.0);
		return x / (x + y);
	}

private:
	double alpha;
	double beta;
	double noncentrality;
};

//! Azzalini's method: with delta = shape / sqrt(1 + shape^2) and independent U0, V ~ N(0, 1),
//! U1 = delta U0 + sqrt(1 - delta^2) V, and the sample is location + scale * (U0 >= 0 ? U1 : -U1).
class SkewNormalSampler {
public:
	using result_type = double;

	SkewNormalSampler(double location, double scale, double shape)
	    : location(location), scale(scale), delta(shape / std::sqrt(1.0 + shape * shape)) {
	}

	template <class Generator>
	double operator()(Generator &g) {
		std::normal_distribution<double> normal(0.0, 1.0);
		const double u0 = normal(g);
		const double v = normal(g);
		const double u1 = delta * u0 + std::sqrt(1.0 - delta * delta) * v;
		return location + scale * (u0 >= 0 ? u1 : -u1);
	}

private:
	double location;
	double scale;
	double delta;
};

//! Michael, Schucany and Haas (1976) transformation with multiple roots.
class InverseGaussianSampler {
public:
	using result_type = double;

	InverseGaussianSampler(double mean, double shape) : mean(mean), shape(shape) {
	}

	template <class Generator>
	double operator()(Generator &g) {
		std::normal_distribution<double> normal(0.0, 1.0);
		const double nu = normal(g);
		const double y = nu * nu;
		const double x = mean + mean * mean * y / (2.0 * shape) -
		                 mean / (2.0 * shape) * std::sqrt(4.0 * mean * shape * y + mean * mean * y * y);
		std::uniform_real_distribution<double> uniform(0.0, 1.0);
		return uniform(g) <= mean / (mean + x) ? x : mean * mean / x;
	}

private:
	double mean;
	double shape;
};

//! scale / Y with Y ~ Gamma(shape, 1).
class InverseGammaSampler {
public:
	using result_type = double;

	InverseGammaSampler(double shape, double scale) : gamma(shape, 1.0), scale(scale) {
	}

	template <class Generator>
	double operator()(Generator &g) {
		double y;
		do {
			y = gamma(g);
		} while (y == 0.0);
		return scale / y;
	}

private:
	std::gamma_distribution<double> gamma;
	double scale;
};

//! Scaled inverse chi-squared: degrees_of_freedom * scale / Y with Y ~ chi2(degrees_of_freedom).
class InverseChiSquaredSampler {
public:
	using result_type = double;

	InverseChiSquaredSampler(double degrees_of_freedom, double scale)
	    : degrees_of_freedom(degrees_of_freedom), scale(scale) {
	}

	template <class Generator>
	double operator()(Generator &g) {
		double y;
		do {
			y = ChiSquaredVariate(g, degrees_of_freedom);
		} while (y == 0.0);
		return degrees_of_freedom * scale / y;
	}

private:
	double degrees_of_freedom;
	double scale;
};

} // namespace duckdb
