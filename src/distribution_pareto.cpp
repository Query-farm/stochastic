#include "utils.hpp"

namespace duckdb {

namespace {

struct ParetoSpec : DistributionSpecBase {
	static constexpr const char *name = "pareto";
	static constexpr const char *display_name = "Pareto";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"scale", "shape"};
	using boost_type = boost::math::pareto_distribution<double, StochasticPolicy>;
	using sampler_type = custom_random::pareto_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "1.0, 3.0";
	static constexpr const char *example_x = "1.5";

	static void Validate(double scale, double shape) {
		RequirePositive(name, "scale", scale);
		RequirePositive(name, "shape", shape);
	}
};

} // namespace

void Load_pareto_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<ParetoSpec>(loader);
}

} // namespace duckdb
