#include "utils.hpp"

namespace duckdb {

namespace {

struct GammaSpec : DistributionSpecBase {
	static constexpr const char *name = "gamma";
	static constexpr const char *display_name = "gamma";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"shape", "scale"};
	using boost_type = boost::math::gamma_distribution<double, StochasticPolicy>;
	using sampler_type = std::gamma_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "2.0, 1.0";
	static constexpr const char *example_x = "1.5";

	static void Validate(double shape, double scale) {
		RequirePositive(name, "shape", shape);
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_gamma_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<GammaSpec>(loader);
}

} // namespace duckdb
