#include "utils.hpp"

namespace duckdb {

namespace {

struct ChiSquaredSpec : DistributionSpecBase {
	static constexpr const char *name = "chi_squared";
	static constexpr const char *display_name = "chi-squared";
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"degrees_of_freedom"};
	using boost_type = boost::math::chi_squared_distribution<double, StochasticPolicy>;
	using sampler_type = std::chi_squared_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "5.0";
	static constexpr const char *example_x = "3.0";

	static void Validate(double degrees_of_freedom) {
		RequirePositive(name, "degrees_of_freedom", degrees_of_freedom);
	}
};

} // namespace

void Load_chi_squared_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<ChiSquaredSpec>(loader);
}

} // namespace duckdb
