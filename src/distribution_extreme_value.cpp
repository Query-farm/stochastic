#include "utils.hpp"

namespace duckdb {

namespace {

struct ExtremeValueSpec : DistributionSpecBase {
	static constexpr const char *name = "extreme_value";
	static constexpr const char *display_name = "extreme value (Gumbel)";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"location", "scale"};
	using boost_type = boost::math::extreme_value_distribution<double, StochasticPolicy>;
	using sampler_type = std::extreme_value_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double, double scale) {
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_extreme_value_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<ExtremeValueSpec>(loader);
}

} // namespace duckdb
