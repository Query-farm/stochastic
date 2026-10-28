#include "utils.hpp"

namespace duckdb {

namespace {

struct SkewNormalSpec : DistributionSpecBase {
	static constexpr const char *name = "skew_normal";
	static constexpr const char *display_name = "skew normal";
	using params = std::tuple<double, double, double>;
	static constexpr std::array<const char *, 3> param_names = {"location", "scale", "shape"};
	using boost_type = boost::math::skew_normal_distribution<double, StochasticPolicy>;
	using sampler_type = SkewNormalSampler;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0, 4.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double, double scale, double) {
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_skew_normal_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<SkewNormalSpec>(loader);
}

} // namespace duckdb
