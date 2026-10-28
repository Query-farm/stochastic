#include "utils.hpp"

namespace duckdb {

namespace {

struct LognormalSpec : DistributionSpecBase {
	static constexpr const char *name = "lognormal";
	static constexpr const char *display_name = "log-normal";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"mean", "stddev"};
	using boost_type = boost::math::lognormal_distribution<double, StochasticPolicy>;
	using sampler_type = std::lognormal_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double, double stddev) {
		RequirePositive(name, "stddev", stddev);
	}
};

} // namespace

void Load_lognormal_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<LognormalSpec>(loader);
}

} // namespace duckdb
