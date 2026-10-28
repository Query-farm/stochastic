#include "utils.hpp"

namespace duckdb {

namespace {

struct WeibullSpec : DistributionSpecBase {
	static constexpr const char *name = "weibull";
	static constexpr const char *display_name = "Weibull";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"shape", "scale"};
	using boost_type = boost::math::weibull_distribution<double, StochasticPolicy>;
	using sampler_type = std::weibull_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "1.5, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double shape, double scale) {
		RequirePositive(name, "shape", shape);
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_weibull_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<WeibullSpec>(loader);
}

} // namespace duckdb
