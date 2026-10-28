#define DUCKDB_EXTENSION_MAIN

#include "stochastic_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <random>
#include <thread>
#include "utils.hpp"
#include "query_farm_telemetry.hpp"
#include "version.hpp"

namespace duckdb {

#define STOCHASTIC_DISTRIBUTIONS(X)                                                                                    \
	X(arcsine)                                                                                                         \
	X(bernoulli)                                                                                                       \
	X(beta)                                                                                                            \
	X(binomial)                                                                                                        \
	X(cauchy)                                                                                                          \
	X(chi_squared)                                                                                                     \
	X(exponential)                                                                                                     \
	X(extreme_value)                                                                                                   \
	X(fisher_f)                                                                                                        \
	X(gamma)                                                                                                           \
	X(geometric)                                                                                                       \
	X(hypergeometric)                                                                                                  \
	X(inverse_chi_squared)                                                                                             \
	X(inverse_gamma)                                                                                                   \
	X(inverse_gaussian)                                                                                                \
	X(kolmogorov_smirnov)                                                                                              \
	X(laplace)                                                                                                         \
	X(logistic)                                                                                                        \
	X(lognormal)                                                                                                       \
	X(negative_binomial)                                                                                               \
	X(non_central_beta)                                                                                                \
	X(non_central_chi_squared)                                                                                         \
	X(non_central_f)                                                                                                   \
	X(non_central_t)                                                                                                   \
	X(normal)                                                                                                          \
	X(pareto)                                                                                                          \
	X(poisson)                                                                                                         \
	X(rayleigh)                                                                                                        \
	X(skew_normal)                                                                                                     \
	X(students_t)                                                                                                      \
	X(triangular)                                                                                                      \
	X(uniform_int)                                                                                                     \
	X(uniform_real)                                                                                                    \
	X(weibull)

#define DECLARE_LOAD(NAME) void Load_##NAME##_distribution(ExtensionLoader &loader);
STOCHASTIC_DISTRIBUTIONS(DECLARE_LOAD)
#undef DECLARE_LOAD
void Load_fit_functions(ExtensionLoader &loader);
void Load_hypothesis_tests(ExtensionLoader &loader);

static void LoadInternal(ExtensionLoader &loader) {
#define CALL_LOAD(NAME) Load_##NAME##_distribution(loader);
	STOCHASTIC_DISTRIBUTIONS(CALL_LOAD)
#undef CALL_LOAD
	Load_fit_functions(loader);
	Load_hypothesis_tests(loader);

	QueryFarmSendTelemetry(loader, "stochastic", STOCHASTIC_VERSION);
}

void StochasticExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string StochasticExtension::Name() {
	return "stochastic";
}

std::string StochasticExtension::Version() const {
	return STOCHASTIC_VERSION;
}

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(stochastic, loader) {
	duckdb::LoadInternal(loader);
}
}
