#include "utils.hpp"

namespace duckdb {

namespace {

struct FisherFSpec : DistributionSpecBase {
	static constexpr const char *name = "fisher_f";
	static constexpr const char *display_name = "Fisher's F";
	using params = std::tuple<double, double>;
	static constexpr std::array<const char *, 2> param_names = {"d1", "d2"};
	using boost_type = boost::math::fisher_f_distribution<double, StochasticPolicy>;
	using sampler_type = std::fisher_f_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "5.0, 10.0";
	static constexpr const char *example_x = "0.5";

	static void Validate(double d1, double d2) {
		RequirePositive(name, "d1", d1);
		RequirePositive(name, "d2", d2);
	}
};

} // namespace

void Load_fisher_f_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<FisherFSpec>(loader);
}

} // namespace duckdb
