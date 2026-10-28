#include "utils.hpp"

namespace duckdb {

namespace {

struct InverseGaussianSpec : DistributionSpecBase {
	static constexpr const char *name = "inverse_gaussian";
	static constexpr const char *display_name = "inverse Gaussian (Wald)";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"mean", "shape"};
	using boost_type = boost::math::inverse_gaussian_distribution<double, StochasticPolicy>;
	using sampler_type = InverseGaussianSampler;
	using sample_t = double;
	static constexpr const char *example_params = "1.0, 2.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double mean, double shape) {
		RequirePositive(name, "mean", mean);
		RequirePositive(name, "shape", shape);
	}
};

} // namespace

void Load_inverse_gaussian_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<InverseGaussianSpec>(loader);
}

} // namespace duckdb
