#include "utils.hpp"

namespace duckdb {

namespace {

struct NonCentralTSpec : DistributionSpecBase {
	static constexpr const char *name = "non_central_t";
	static constexpr const char *display_name = "non-central Student's t";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"degrees_of_freedom", "noncentrality"};
	using boost_type = boost::math::non_central_t_distribution<double, StochasticPolicy>;
	using sampler_type = NonCentralTSampler;
	using sample_t = double;
	static constexpr const char *example_params = "10.0, 1.0";
	static constexpr const char *example_x = "1.5";

	static void Validate(double degrees_of_freedom, double) {
		RequirePositive(name, "degrees_of_freedom", degrees_of_freedom);
	}
};

} // namespace

void Load_non_central_t_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<NonCentralTSpec>(loader);
}

} // namespace duckdb
