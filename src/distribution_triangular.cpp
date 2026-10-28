#include "utils.hpp"

namespace duckdb {

namespace {

struct TriangularSpec : DistributionSpecBase {
	static constexpr const char *name = "triangular";
	static constexpr const char *display_name = "triangular";
	using params = std::tuple<double, double, double>;
	static constexpr std::array<const char *, 3> param_names = {"lower", "mode", "upper"};
	using boost_type = boost::math::triangular_distribution<double, StochasticPolicy>;
	using sampler_type = InverseTransformSampler<boost_type>;
	using sample_t = double;
	static constexpr const char *example_params = "0.0, 0.5, 1.0";
	static constexpr const char *example_x = "0.25";

	static void Validate(double lower, double mode, double upper) {
		RequireFinite(name, "lower", lower);
		RequireFinite(name, "mode", mode);
		RequireFinite(name, "upper", upper);
		if (!(lower < upper && lower <= mode && mode <= upper)) {
			throw InvalidInputException("%s: parameters must satisfy lower <= mode <= upper and lower < upper", name);
		}
	}
};

} // namespace

void Load_triangular_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<TriangularSpec>(loader);
}

} // namespace duckdb
