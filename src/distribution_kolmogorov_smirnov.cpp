#include "utils.hpp"

#include <boost/math/tools/toms748_solve.hpp>

namespace duckdb {

namespace {

struct KolmogorovSmirnovSpec : DistributionSpecBase {
	static constexpr const char *name = "kolmogorov_smirnov";
	static constexpr const char *display_name = "Kolmogorov-Smirnov";
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"sample_size"};
	using boost_type = boost::math::kolmogorov_smirnov_distribution<double, StochasticPolicy>;
	struct Sampler;
	using sampler_type = Sampler;
	using sample_t = double;
	static constexpr const char *example_params = "10.0";
	static constexpr const char *example_x = "0.3";

	static void Validate(double sample_size) {
		RequirePositive(name, "sample_size", sample_size);
	}

	//! Beyond x * sqrt(n) = 40 the density is below the smallest DOUBLE. Boost's series for it never terminates
	//! once 8 x overflows, so the tail is handled here.
	static bool InFarTail(const boost_type &dist, double x) {
		return std::isfinite(x) && x * std::sqrt(dist.number_of_observations()) > 40;
	}
	static double Pdf(const boost_type &dist, double x) {
		return InFarTail(dist, x) ? 0 : boost::math::pdf(dist, x);
	}
	static double LogPdf(const boost_type &dist, double x) {
		return std::log(Pdf(dist, x));
	}
	//! pdf / (1 - cdf); in the far tail both are dominated by their first series terms, whose ratio is 4 x n.
	static double Hazard(const boost_type &dist, double x) {
		return InFarTail(dist, x) ? 4 * x * dist.number_of_observations() : boost::math::hazard(dist, x);
	}

	//! Boost's Newton iteration fails to converge for many arguments, so the CDF is inverted by bracketing the
	//! root and solving with TOMS 748.
	static double Invert(const boost_type &dist, double p, bool complement) {
		CheckProbabilityArgument(p);
		if (p == (complement ? 1 : 0)) {
			return 0;
		}
		if (p == (complement ? 0 : 1)) {
			return std::numeric_limits<double>::infinity();
		}
		// Increasing in x, negative at x = 0
		auto f = [&](double x) {
			return complement ? p - boost::math::cdf(boost::math::complement(dist, x)) : boost::math::cdf(dist, x) - p;
		};
		double low = 0;
		double high = 1 / std::sqrt(dist.number_of_observations());
		double f_high = f(high);
		while (f_high < 0) {
			low = high;
			high *= 2;
			f_high = f(high);
		}
		if (f_high == 0) {
			return high;
		}
		std::uintmax_t iterations = 200;
		auto root = boost::math::tools::toms748_solve(f, low, high, f(low), f_high,
		                                              boost::math::tools::eps_tolerance<double>(52), iterations,
		                                              StochasticPolicy());
		return (root.first + root.second) / 2;
	}
	static double Quantile(const boost_type &dist, double p) {
		return Invert(dist, p, false);
	}
	static double QuantileComplement(const boost_type &dist, double q) {
		return Invert(dist, q, true);
	}
	static double Median(const boost_type &dist) {
		return Invert(dist, 0.5, false);
	}

	struct Sampler {
		using result_type = double;
		explicit Sampler(double sample_size) : dist(sample_size) {
		}
		template <class Generator>
		double operator()(Generator &g) {
			return Quantile(dist, OpenUniform(g));
		}
		boost_type dist;
	};
};

} // namespace

void Load_kolmogorov_smirnov_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<KolmogorovSmirnovSpec>(loader);
}

} // namespace duckdb
