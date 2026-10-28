#include "utils.hpp"

namespace duckdb {

namespace {

struct PoissonSpec : DistributionSpecBase {
	static constexpr const char *name = "poisson";
	static constexpr const char *display_name = "Poisson";
	static constexpr bool discrete = true;
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"rate"};
	using boost_type = boost::math::poisson_distribution<double, StochasticDiscretePolicy>;
	using sampler_type = std::poisson_distribution<int64_t>;
	using sample_t = int64_t;
	static constexpr const char *example_params = "5.0";
	static constexpr const char *example_x = "3";

	static void Validate(double rate) {
		RequirePositive(name, "rate", rate);
	}
};

} // namespace

void Load_poisson_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<PoissonSpec>(loader);
}

} // namespace duckdb
