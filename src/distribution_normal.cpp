#include "utils.hpp"

namespace duckdb {

namespace {

struct NormalSpec : DistributionSpecBase {
	static constexpr const char *name = "normal";
	static constexpr const char *display_name = "normal";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"mean", "stddev"};
	using boost_type = boost::math::normal_distribution<double, StochasticPolicy>;
	using sampler_type = std::normal_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double, double stddev) {
		RequirePositive(name, "stddev", stddev);
	}
};

} // namespace

void Load_normal_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<NormalSpec>(loader);
}

} // namespace duckdb
