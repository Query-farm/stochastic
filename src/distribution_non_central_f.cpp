#include "utils.hpp"

namespace duckdb {

namespace {

struct NonCentralFSpec : DistributionSpecBase {
	static constexpr const char *name = "non_central_f";
	static constexpr const char *display_name = "non-central F";
	using params = std::tuple<double, double, double>;
	static constexpr std::array<const char *, 3> param_names = {"d1", "d2", "noncentrality"};
	using boost_type = boost::math::non_central_f_distribution<double, StochasticPolicy>;
	using sampler_type = NonCentralFSampler;
	using sample_t = double;
	static constexpr const char *example_params = "5.0, 10.0, 2.0";
	static constexpr const char *example_x = "1.5";

	static void Validate(double d1, double d2, double noncentrality) {
		RequirePositive(name, "d1", d1);
		RequirePositive(name, "d2", d2);
		RequireNonNegative(name, "noncentrality", noncentrality);
	}
};

} // namespace

void Load_non_central_f_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<NonCentralFSpec>(loader);
}

} // namespace duckdb
