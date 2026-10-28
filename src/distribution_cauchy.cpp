#include "utils.hpp"

namespace duckdb {

namespace {

struct CauchySpec : DistributionSpecBase {
	static constexpr const char *name = "cauchy";
	static constexpr const char *display_name = "Cauchy";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"location", "scale"};
	using boost_type = boost::math::cauchy_distribution<double, StochasticUndefinedPolicy>;
	using sampler_type = std::cauchy_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double, double scale) {
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_cauchy_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<CauchySpec>(loader);
}

} // namespace duckdb
