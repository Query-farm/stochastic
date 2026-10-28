#include "utils.hpp"

namespace duckdb {

namespace {

struct UniformRealSpec : DistributionSpecBase {
	static constexpr const char *name = "uniform_real";
	static constexpr const char *display_name = "continuous uniform";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"min", "max"};
	using boost_type = boost::math::uniform_distribution<double, StochasticPolicy>;
	using sampler_type = std::uniform_real_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double min, double max) {
		RequireFinite(name, "min", min);
		RequireFinite(name, "max", max);
		if (!(min < max)) {
			throw InvalidInputException("%s: min must be < max, got min=%s max=%s", name, FormatParameter(min),
			                            FormatParameter(max));
		}
	}
};

} // namespace

void Load_uniform_real_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<UniformRealSpec>(loader);
}

} // namespace duckdb
