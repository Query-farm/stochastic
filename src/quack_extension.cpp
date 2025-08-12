#define DUCKDB_EXTENSION_MAIN

#include "quack_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <boost/math/distributions.hpp>

namespace duckdb {

// Template function for normal distribution operations
template <typename Operation>
inline void NormalDistributionFunc(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &mean_vector = args.data[0];
	auto &stddev_vector = args.data[1];
	auto &x_vector = args.data[2];

	TernaryExecutor::Execute<double, double, double, double>(
	    mean_vector, stddev_vector, x_vector, result, args.size(), 
	    [&](double mean, double stddev, double x) {
		    boost::math::normal_distribution<double> dist(mean, stddev);
		    return op(dist, x);
	    });
}

// Specific function implementations using the template
inline void NormalPdfFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	NormalDistributionFunc(args, state, result, 
	    [](const auto &dist, double x) { return boost::math::pdf(dist, x); });
}

inline void NormalLogPdfFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	NormalDistributionFunc(args, state, result, 
	    [](const auto &dist, double x) { return boost::math::logpdf(dist, x); });
}

inline void NormalCdfFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	NormalDistributionFunc(args, state, result, 
	    [](const auto &dist, double x) { return boost::math::cdf(dist, x); });
}

inline void NormalLogCdfFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	NormalDistributionFunc(args, state, result, 
	    [](const auto &dist, double x) { return boost::math::logcdf(dist, x); });
}

inline void NormalQuantileFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	NormalDistributionFunc(args, state, result, 
	    [](const auto &dist, double p) { return boost::math::quantile(dist, p); });
}

// Generic template for any distribution with two parameters
template <typename Distribution, typename Operation>
inline void TwoParameterDistributionFunc(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];
	auto &x_vector = args.data[2];

	TernaryExecutor::Execute<double, double, double, double>(
	    param1_vector, param2_vector, x_vector, result, args.size(), 
	    [&](double param1, double param2, double x) {
		    Distribution dist(param1, param2);
		    return op(dist, x);
	    });
}

// Helper function to create a scalar function with consistent signature
template <typename FuncType>
ScalarFunction CreateDistributionFunction(const std::string &name, FuncType func) {
	return ScalarFunction(name, 
	                     {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE},
	                     LogicalType::DOUBLE, 
	                     func);
}

static void LoadInternal(DatabaseInstance &instance) {
	// Register normal distribution functions using the template helper
	ExtensionUtil::RegisterFunction(instance, CreateDistributionFunction("normal_pdf", NormalPdfFunc));
	ExtensionUtil::RegisterFunction(instance, CreateDistributionFunction("normal_logpdf", NormalLogPdfFunc));
	ExtensionUtil::RegisterFunction(instance, CreateDistributionFunction("normal_cdf", NormalCdfFunc));
	ExtensionUtil::RegisterFunction(instance, CreateDistributionFunction("normal_logcdf", NormalLogCdfFunc));
	ExtensionUtil::RegisterFunction(instance, CreateDistributionFunction("normal_quantile", NormalQuantileFunc));
}

void QuackExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string QuackExtension::Name() {
	return "quack";
}

std::string QuackExtension::Version() const {
#ifdef EXT_VERSION_QUACK
	return EXT_VERSION_QUACK;
#else
	return "";
#endif
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
