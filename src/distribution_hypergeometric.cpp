#include "utils.hpp"

namespace duckdb {

namespace {

struct HypergeometricSpec : DistributionSpecBase {
	static constexpr const char *name = "hypergeometric";
	static constexpr const char *display_name = "hypergeometric";
	static constexpr bool discrete = true;
	using params = std::tuple<int64_t, int64_t, int64_t>;
	static constexpr std::array<const char *, 3> param_names = {"successes", "draws", "population"};
	using boost_type = boost::math::hypergeometric_distribution<double, StochasticDiscretePolicy>;
	using sampler_type = InverseTransformSampler<boost_type, int64_t>;
	using sample_t = int64_t;
	using quantile_t = int64_t;
	static constexpr const char *example_params = "10, 5, 50";
	static constexpr const char *example_x = "2";

	static void Validate(int64_t successes, int64_t draws, int64_t population) {
		RequireNonNegative(name, "successes", successes);
		RequireNonNegative(name, "draws", draws);
		RequirePositive(name, "population", population);
		if (population > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
			throw InvalidInputException("%s: population must be <= 4294967295, got %s", name,
			                            std::to_string(population));
		}
		if (successes > population || draws > population) {
			throw InvalidInputException("%s: successes and draws must be <= population", name);
		}
	}
};

} // namespace

void Load_hypergeometric_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<HypergeometricSpec>(loader);
}

} // namespace duckdb
