#include "utils.hpp"

namespace duckdb {

namespace {

struct InverseGammaSpec : DistributionSpecBase {
	static constexpr const char *name = "inverse_gamma";
	static constexpr const char *display_name = "inverse gamma";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"shape", "scale"};
	using boost_type = boost::math::inverse_gamma_distribution<double, StochasticPolicy>;
	using sampler_type = InverseGammaSampler;
	using sample_t = double;
	static constexpr const char *example_params = "3.0, 2.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double shape, double scale) {
		RequirePositive(name, "shape", shape);
		RequirePositive(name, "scale", scale);
	}
};

} // namespace

void Load_inverse_gamma_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<InverseGammaSpec>(loader);
}

} // namespace duckdb
