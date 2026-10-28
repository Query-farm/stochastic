#include "utils.hpp"

namespace duckdb {

namespace {

struct StudentsTSpec : DistributionSpecBase {
	static constexpr const char *name = "students_t";
	static constexpr const char *display_name = "Student's t";
	using params = std::tuple<double>;
	static constexpr std::array<const char *, 1> param_names = {"degrees_of_freedom"};
	using boost_type = boost::math::students_t_distribution<double, StochasticPolicy>;
	using sampler_type = std::student_t_distribution<double>;
	using sample_t = double;
	static constexpr const char *example_params = "10.0";
	static constexpr const char *example_x = "1.5";

	static void Validate(double degrees_of_freedom) {
		RequirePositive(name, "degrees_of_freedom", degrees_of_freedom);
	}
};

} // namespace

void Load_students_t_distribution(ExtensionLoader &loader) {
	RegisterStandardDistribution<StudentsTSpec>(loader);
}

} // namespace duckdb
