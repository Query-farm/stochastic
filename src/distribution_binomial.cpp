#include "utils.hpp"

namespace duckdb {

namespace {

struct BinomialSpec : DistributionSpecBase {
	static constexpr const char *name = "binomial";
	static constexpr const char *display_name = "binomial";
	static constexpr bool discrete = true;
	using params = std::tuple<int64_t, double>;
	static constexpr std::array<const char *, 2> param_names = {"trials", "prob"};
	using boost_type = boost::math::binomial_distribution<double, StochasticDiscretePolicy>;
	using sampler_type = std::binomial_distribution<int64_t>;
	using sample_t = int64_t;
	using quantile_t = int64_t;
	static constexpr const char *example_params = "10, 0.5";
	static constexpr const char *example_x = "5";

	static void Validate(int64_t trials, double prob) {
		RequirePositive(name, "trials", trials);
		RequireProbability(name, "prob", prob);
	}
};

} // namespace

void Load_binomial_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<BinomialSpec>(loader);
}

} // namespace duckdb
