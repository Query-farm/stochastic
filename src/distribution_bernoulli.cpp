#include "utils.hpp"

namespace duckdb {

namespace {

struct BernoulliSpec : DistributionSpecBase {
	static constexpr const char *name = "bernoulli";
	static constexpr const char *display_name = "Bernoulli";
	static constexpr bool discrete = true;
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"prob"};
	using boost_type = boost::math::bernoulli_distribution<double, StochasticDiscretePolicy>;
	using sampler_type = std::bernoulli_distribution;
	using sample_t = bool;
	static constexpr const char *example_params = "0.5";
	static constexpr const char *example_x = "1";

	static void Validate(double prob) {
		RequireProbability(name, "prob", prob);
	}
};

} // namespace

void Load_bernoulli_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<BernoulliSpec>(loader);
}

} // namespace duckdb
