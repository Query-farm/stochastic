#define DUCKDB_EXTENSION_MAIN

#include "quack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <random>
#include <thread>
#include "utils.hpp"

namespace duckdb {

void Load_bernoulli_distribution(DatabaseInstance &instance);
void Load_beta_distribution(DatabaseInstance &instance);
void Load_binomial_distribution(DatabaseInstance &instance);
void Load_chi_squared_distribution(DatabaseInstance &instance);
void Load_exponential_distribution(DatabaseInstance &instance);
void Load_extreme_value_distribution(DatabaseInstance &instance);
void Load_fisher_f_distribution(DatabaseInstance &instance);
void Load_gamma_distribution(DatabaseInstance &instance);
void Load_geometric_distribution(DatabaseInstance &instance);
void Load_laplace_distribution(DatabaseInstance &instance);
void Load_logistic_distribution(DatabaseInstance &instance);
void Load_lognormal_distribution(DatabaseInstance &instance);
void Load_negative_binomial_distribution(DatabaseInstance &instance);
void Load_normal_distribution(DatabaseInstance &instance);
void Load_pareto_distribution(DatabaseInstance &instance);
void Load_poisson_distribution(DatabaseInstance &instance);
void Load_rayleigh_distribution(DatabaseInstance &instance);
void Load_students_t_distribution(DatabaseInstance &instance);
void Load_uniform_int_distribution(DatabaseInstance &instance);
void Load_uniform_real_distribution(DatabaseInstance &instance);
void Load_weibull_distribution(DatabaseInstance &instance);

static void LoadInternal(DatabaseInstance &instance) {
	Load_bernoulli_distribution(instance);
	Load_beta_distribution(instance);
	Load_binomial_distribution(instance);
	Load_chi_squared_distribution(instance);
	Load_exponential_distribution(instance);
	Load_extreme_value_distribution(instance);
	Load_fisher_f_distribution(instance);
	Load_gamma_distribution(instance);
	Load_geometric_distribution(instance);
	Load_laplace_distribution(instance);
	Load_logistic_distribution(instance);
	Load_lognormal_distribution(instance);
	Load_negative_binomial_distribution(instance);
	Load_normal_distribution(instance);
	Load_pareto_distribution(instance);
	Load_poisson_distribution(instance);
	Load_rayleigh_distribution(instance);
	Load_students_t_distribution(instance);
	Load_uniform_int_distribution(instance);
	Load_uniform_real_distribution(instance);
	Load_weibull_distribution(instance);
}

void QuackExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string QuackExtension::Name() {
	return "stochastic";
}

std::string QuackExtension::Version() const {
	return "0.0.1";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void quack_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::QuackExtension>();
}

DUCKDB_EXTENSION_API const char *quack_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
