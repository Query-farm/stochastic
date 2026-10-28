#include "utils.hpp"

namespace duckdb {

namespace {

struct InverseChiSquaredSpec : DistributionSpecBase {
	static constexpr const char *name = "inverse_chi_squared";
	static constexpr const char *display_name = "scaled inverse chi-squared";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"degrees_of_freedom", "scale"};
	using boost_type = boost::math::inverse_chi_squared_distribution<double, StochasticPolicy>;
	using sampler_type = InverseChiSquaredSampler;
	using sample_t = double;
	static constexpr const char *example_params = "5.0, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double degrees_of_freedom, double scale) {
		RequirePositive(name, "degrees_of_freedom", degrees_of_freedom);
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_inverse_chi_squared_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<InverseChiSquaredSpec>(loader);
}

} // namespace duckdb
