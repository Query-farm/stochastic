#include "utils.hpp"

namespace duckdb {

namespace {

struct LaplaceSpec : DistributionSpecBase {
	static constexpr const char *name = "laplace";
	static constexpr const char *display_name = "Laplace";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"location", "scale"};
	using boost_type = boost::math::laplace_distribution<double, StochasticPolicy>;
	using sampler_type = custom_random::laplace_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double, double scale) {
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_laplace_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<LaplaceSpec>(loader);
}

} // namespace duckdb
