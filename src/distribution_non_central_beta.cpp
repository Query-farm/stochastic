#include "utils.hpp"

namespace duckdb {

namespace {

struct NonCentralBetaSpec : DistributionSpecBase {
	static constexpr const char *name = "non_central_beta";
	static constexpr const char *display_name = "non-central beta";
	using params = std::tuple<double, double, double>;
	static constexpr std::array<const char *, 3> param_names = {"alpha", "beta", "noncentrality"};
	using boost_type = boost::math::non_central_beta_distribution<double, StochasticUndefinedPolicy>;
	using sampler_type = NonCentralBetaSampler;
	using sample_t = double;
	static constexpr const char *example_params = "2.0, 3.0, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double alpha, double beta, double noncentrality) {
		RequirePositive(name, "alpha", alpha);
		RequirePositive(name, "beta", beta);
		RequireNonNegative(name, "noncentrality", noncentrality);
		// Boost's series take too long or do not converge beyond this
		RequireAtMost(name, "noncentrality", noncentrality, MAX_NONCENTRALITY);
	}
};

} // namespace

void Load_non_central_beta_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<NonCentralBetaSpec>(loader);
}

} // namespace duckdb
