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

void Load_bernoulli_distribution(ExtensionLoader &loader);
void Load_beta_distribution(ExtensionLoader &loader);
void Load_binomial_distribution(ExtensionLoader &loader);
void Load_chi_squared_distribution(ExtensionLoader &loader);
void Load_exponential_distribution(ExtensionLoader &loader);
void Load_extreme_value_distribution(ExtensionLoader &loader);
void Load_fisher_f_distribution(ExtensionLoader &loader);
void Load_gamma_distribution(ExtensionLoader &loader);
void Load_geometric_distribution(ExtensionLoader &loader);
void Load_laplace_distribution(ExtensionLoader &loader);
void Load_logistic_distribution(ExtensionLoader &loader);
void Load_lognormal_distribution(ExtensionLoader &loader);
void Load_negative_binomial_distribution(ExtensionLoader &loader);
void Load_normal_distribution(ExtensionLoader &loader);
void Load_pareto_distribution(ExtensionLoader &loader);
void Load_poisson_distribution(ExtensionLoader &loader);
void Load_rayleigh_distribution(ExtensionLoader &loader);
void Load_students_t_distribution(ExtensionLoader &loader);
void Load_uniform_int_distribution(ExtensionLoader &loader);
void Load_uniform_real_distribution(ExtensionLoader &loader);
void Load_weibull_distribution(ExtensionLoader &loader);

static void LoadInternal(ExtensionLoader &loader) {
	Load_bernoulli_distribution(loader);
	Load_beta_distribution(loader);
	Load_binomial_distribution(loader);
	Load_chi_squared_distribution(loader);
	Load_exponential_distribution(loader);
	Load_extreme_value_distribution(loader);
	Load_fisher_f_distribution(loader);
	Load_gamma_distribution(loader);
	Load_geometric_distribution(loader);
	Load_laplace_distribution(loader);
	Load_logistic_distribution(loader);
	Load_lognormal_distribution(loader);
	Load_negative_binomial_distribution(loader);
	Load_normal_distribution(loader);
	Load_pareto_distribution(loader);
	Load_poisson_distribution(loader);
	Load_rayleigh_distribution(loader);
	Load_students_t_distribution(loader);
	Load_uniform_int_distribution(loader);
	Load_uniform_real_distribution(loader);
	Load_weibull_distribution(loader);

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
