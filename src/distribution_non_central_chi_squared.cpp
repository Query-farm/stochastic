#include "utils.hpp"

namespace duckdb {

namespace {

struct NonCentralChiSquaredSpec : DistributionSpecBase {
	static constexpr const char *name = "non_central_chi_squared";
	static constexpr const char *display_name = "non-central chi-squared";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"degrees_of_freedom", "noncentrality"};
	using boost_type = boost::math::non_central_chi_squared_distribution<double, StochasticPolicy>;
	using sampler_type = NonCentralChiSquaredSampler;
	using sample_t = double;
	static constexpr const char *example_params = "4.0, 2.0";
	static constexpr const char *example_x = "3.0";

	static void Validate(double degrees_of_freedom, double noncentrality) {
		RequirePositive(name, "degrees_of_freedom", degrees_of_freedom);
		RequireNonNegative(name, "noncentrality", noncentrality);
	}
};

} // namespace

void Load_non_central_chi_squared_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<NonCentralChiSquaredSpec>(loader);
}

} // namespace duckdb
