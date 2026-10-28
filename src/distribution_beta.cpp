#include "utils.hpp"

namespace duckdb {

namespace {

struct BetaSpec : DistributionSpecBase {
	static constexpr const char *name = "beta";
	static constexpr const char *display_name = "beta";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"alpha", "beta"};
	using boost_type = boost::math::beta_distribution<double, StochasticPolicy>;
	using sampler_type = custom_random::beta_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "2.0, 5.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double alpha, double beta) {
		RequirePositive(name, "alpha", alpha);
		RequirePositive(name, "beta", beta);
	}
};

} // namespace

void Load_beta_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<BetaSpec>(loader);
}

} // namespace duckdb
