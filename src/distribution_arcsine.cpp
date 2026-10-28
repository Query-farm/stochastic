#include "utils.hpp"

namespace duckdb {

namespace {

struct ArcsineSpec : DistributionSpecBase {
	static constexpr const char *name = "arcsine";
	static constexpr const char *display_name = "arcsine";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"min", "max"};
	using boost_type = boost::math::arcsine_distribution<double, StochasticPolicy>;
	using sampler_type = InverseTransformSampler<boost_type>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 1.0";
	static constexpr const char *example_x = "0.25";

	static void Validate(double min, double max) {
		if (!(min < max)) {
			throw InvalidInputException("%s: min must be < max, got min=%s max=%s", name, FormatParameter(min),
			                            FormatParameter(max));
		}
	}
};

} // namespace

void Load_arcsine_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<ArcsineSpec>(loader);
}

} // namespace duckdb
