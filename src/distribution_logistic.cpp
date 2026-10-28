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

	static void Validate(double, double scale) {
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_logistic_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<LogisticSpec>(loader);
}

} // namespace duckdb
