#include "utils.hpp"

#include <boost/math/tools/minima.hpp>

namespace duckdb {

namespace {

struct NonCentralChiSquaredSpec : DistributionSpecBase {
	static constexpr const char *name = "non_central_chi_squared";
	static constexpr const char *display_name = "non-central chi-squared";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"degrees_of_freedom", "noncentrality"};
	using boost_type = boost::math::non_central_chi_squared_distribution<double, StochasticPolicy>;
	using sampler_type = NonCentralChiSquaredSampler;
	using sample_t = double;
	static constexpr const char *example_params = "4.0, 2.0";
	static constexpr const char *example_x = "3.0";

	//! Boost's mode search stops early for large noncentrality (it returns about half the true mode at 1e6).
	//! There the density is unimodal and concentrated around the mean, so maximize it with Brent's method.
	static double Mode(const boost_type &dist) {
		if (dist.non_centrality() < 1000) {
			return boost::math::mode(dist);
		}
		const double mean = boost::math::mean(dist);
		const double stddev = boost::math::standard_deviation(dist);
		auto negative_pdf = [&](double x) {
			return -boost::math::pdf(dist, x);
		};
		std::uintmax_t iterations = 200;
		return boost::math::tools::brent_find_minima(negative_pdf, std::max(0.0, mean - 10 * stddev),
		                                             mean + 10 * stddev, 52, iterations)
		    .first;
	}

	static void Validate(double degrees_of_freedom, double noncentrality) {
		RequirePositive(name, "degrees_of_freedom", degrees_of_freedom);
		RequireNonNegative(name, "noncentrality", noncentrality);
		// Boost's series take too long or do not converge beyond this
		RequireAtMost(name, "noncentrality", noncentrality, MAX_NONCENTRALITY);
	}
};

} // namespace

void Load_non_central_chi_squared_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<NonCentralChiSquaredSpec>(loader);
}

} // namespace duckdb
