#include "utils.hpp"

namespace duckdb {

namespace {

struct KolmogorovSmirnovSpec : DistributionSpecBase {
	static constexpr const char *name = "kolmogorov_smirnov";
	static constexpr const char *display_name = "Kolmogorov-Smirnov";
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"sample_size"};
	using boost_type = boost::math::kolmogorov_smirnov_distribution<double, StochasticPolicy>;
	using sampler_type = InverseTransformSampler<boost_type>;
	using sample_t = double;
	static constexpr const char *example_params = "10.0";
	static constexpr const char *example_x = "0.3";

	static void Validate(double sample_size) {
		RequirePositive(name, "sample_size", sample_size);
	}
};

} // namespace

void Load_kolmogorov_smirnov_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<KolmogorovSmirnovSpec>(loader);
}

} // namespace duckdb
