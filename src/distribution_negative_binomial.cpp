#include "utils.hpp"

namespace duckdb {

namespace {

struct NegativeBinomialSpec : DistributionSpecBase {
	static constexpr const char *name = "negative_binomial";
	static constexpr const char *display_name = "negative binomial";
	static constexpr bool discrete = true;
	using params = std::tuple<int64_t, double>;
	static constexpr std::array<const char *, 2> param_names = {"successes", "prob"};
	using boost_type = boost::math::negative_binomial_distribution<double, StochasticDiscretePolicy>;
	using sampler_type = std::negative_binomial_distribution<int64_t>;
	using sample_t = int64_t;
	using quantile_t = int64_t;
	static constexpr const char *example_params = "10, 0.5";
	static constexpr const char *example_x = "5";

	static void Validate(int64_t successes, double prob) {
		RequirePositive(name, "successes", successes);
		RequireProbability(name, "prob", prob);
		if (prob == 0) {
			throw InvalidInputException("%s: prob must be > 0, got 0", name);
		}
	}
};

} // namespace

void Load_negative_binomial_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<NegativeBinomialSpec>(loader);
}

} // namespace duckdb
