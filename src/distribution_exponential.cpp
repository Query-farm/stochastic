#include "utils.hpp"

namespace duckdb {

namespace {

struct ExponentialSpec : DistributionSpecBase {
	static constexpr const char *name = "exponential";
	static constexpr const char *display_name = "exponential";
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"rate"};
	using boost_type = boost::math::exponential_distribution<double, StochasticPolicy>;
	using sampler_type = std::exponential_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double rate) {
		RequirePositive(name, "rate", rate);
	}
};

} // namespace

void Load_exponential_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<ExponentialSpec>(loader);
}

} // namespace duckdb
