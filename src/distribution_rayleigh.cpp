#include "utils.hpp"

namespace duckdb {

namespace {

struct RayleighSpec : DistributionSpecBase {
	static constexpr const char *name = "rayleigh";
	static constexpr const char *display_name = "Rayleigh";
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"scale"};
	using boost_type = boost::math::rayleigh_distribution<double, StochasticPolicy>;
	using sampler_type = custom_random::rayleigh_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double scale) {
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_rayleigh_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<RayleighSpec>(loader);
}

} // namespace duckdb
