#include "utils.hpp"

namespace duckdb {

namespace {

struct GeometricSpec : DistributionSpecBase {
	static constexpr const char *name = "geometric";
	static constexpr const char *display_name = "geometric";
	static constexpr bool discrete = true;
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"prob"};
	using boost_type = boost::math::geometric_distribution<double, StochasticDiscretePolicy>;
	using sampler_type = std::geometric_distribution<int64_t>;
	using sample_t = int64_t;
	static constexpr const char *example_params = "0.5";
	static constexpr const char *example_x = "2";

	static void Validate(double prob) {
		RequireProbability(name, "prob", prob);
		if (prob == 0) {
			throw InvalidInputException("%s: prob must be > 0, got 0", name);
		}
	}
};

} // namespace

void Load_geometric_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<GeometricSpec>(loader);
}

} // namespace duckdb
