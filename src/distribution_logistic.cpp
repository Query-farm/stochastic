#include "utils.hpp"

namespace duckdb {

namespace {

struct LogisticSpec : DistributionSpecBase {
	static constexpr const char *name = "logistic";
	static constexpr const char *display_name = "logistic";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"location", "scale"};
	using boost_type = boost::math::logistic_distribution<double, StochasticPolicy>;
	using sampler_type = custom_random::logistic_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0";
	static constexpr const char *example_x = "0.5";

	//! log(1 / (1 + exp(-z))), computed stably. Boost's logcdf returns the CDF instead for large |z|.
	static double LogStandardCdf(double z) {
		return z > 0 ? -std::log1p(std::exp(-z)) : z - std::log1p(std::exp(z));
	}
	static double LogCdf(const boost_type &dist, double x) {
		// Boost reports the invalid argument for NaN
		return std::isnan(x) ? boost::math::logcdf(dist, x) : LogStandardCdf((x - dist.location()) / dist.scale());
	}
	static double LogCdfComplement(const boost_type &dist, double x) {
		return std::isnan(x) ? boost::math::logcdf(boost::math::complement(dist, x))
		                     : LogStandardCdf((dist.location() - x) / dist.scale());
	}

	static void Validate(double, double scale) {
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_logistic_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<LogisticSpec>(loader);
}

} // namespace duckdb
