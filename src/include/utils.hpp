#pragma once
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

#include <boost/math/distributions.hpp>
#include <boost/random.hpp>
#include "rng_utils.hpp"

namespace duckdb {

// Generic template for any distribution with two parameters
template <typename Distribution, typename Operation>
inline void TwoParameterDistributionFunc(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];
	auto &x_vector = args.data[2];

	// If these parameters are constant vectors they should be treated as such.
	if (param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {

		if (ConstantVector::IsNull(param1_vector) || ConstantVector::IsNull(param2_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		const auto param1 = ConstantVector::GetData<double>(param1_vector)[0];
		const auto param2 = ConstantVector::GetData<double>(param2_vector)[0];
		Distribution dist(param1, param2);
		UnaryExecutor::Execute<double, double>(x_vector, result, args.size(), [&](double x) { return op(dist, x); });
	}

	TernaryExecutor::Execute<double, double, double, double>(param1_vector, param2_vector, x_vector, result,
	                                                         args.size(), [&](double param1, double param2, double x) {
		                                                         Distribution dist(param1, param2);
		                                                         return op(dist, x);
	                                                         });
}

// Generic macro to define distribution functions for any distribution type
#define DEFINE_DISTRIBUTION_FUNC(func_name, distribution_type, boost_operation)                                        \
	inline void func_name(DataChunk &args, ExpressionState &state, Vector &result) {                                   \
		TwoParameterDistributionFunc<distribution_type>(args, state, result,                                           \
		                                                [](const auto &dist, auto x) { return boost_operation; });     \
	}

template <typename Distribution, typename Operation>
inline void SingleParameterDistributionFunc(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &param_vector = args.data[0];
	auto &x_vector = args.data[1];

	// If parameter is constant, optimize the computation
	if (param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		const auto param = ConstantVector::GetData<double>(param_vector)[0];
		Distribution dist(param);
		UnaryExecutor::Execute<double, double>(x_vector, result, args.size(), [&](double x) { return op(dist, x); });
	} else {
		BinaryExecutor::Execute<double, double, double>(param_vector, x_vector, result, args.size(),
		                                                [&](double param, double x) {
			                                                Distribution dist(param);
			                                                return op(dist, x);
		                                                });
	}
}

// Generic macro to define single-parameter distribution functions
#define DEFINE_SINGLE_PARAM_DISTRIBUTION_FUNC(func_name, distribution_type, boost_operation)                           \
	inline void func_name(DataChunk &args, ExpressionState &state, Vector &result) {                                   \
		SingleParameterDistributionFunc<distribution_type>(args, state, result,                                        \
		                                                   [](const auto &dist, auto x) { return boost_operation; });  \
	}

// Template for distribution statistics functions (no x parameter, just distribution parameters)
template <typename Distribution, typename Operation>
inline void TwoParameterStatisticFunc(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];

	// If these parameters are constant vectors they should be treated as such.
	if (param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {

		if (ConstantVector::IsNull(param1_vector) || ConstantVector::IsNull(param2_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}
		const auto param1 = ConstantVector::GetData<double>(param1_vector)[0];
		const auto param2 = ConstantVector::GetData<double>(param2_vector)[0];
		Distribution dist(param1, param2);
		auto stat_value = op(dist);

		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<double>(result);
		result_data[0] = stat_value;
		return;
	}

	BinaryExecutor::Execute<double, double, double>(param1_vector, param2_vector, result, args.size(),
	                                                [&](double param1, double param2) {
		                                                Distribution dist(param1, param2);
		                                                return op(dist);
	                                                });
}

// Generic macro to define distribution statistic functions
#define DEFINE_DISTRIBUTION_STATISTIC_FUNC(func_name, distribution_type, boost_operation)                              \
	inline void func_name(DataChunk &args, ExpressionState &state, Vector &result) {                                   \
		TwoParameterStatisticFunc<distribution_type>(args, state, result,                                              \
		                                             [](const auto &dist) { return boost_operation; });                \
	}

// Helper function to create a scalar function with consistent signature
template <typename FuncType>
ScalarFunction CreateThreeParameterDistributionFunction(const std::string &name, FuncType func) {
	return ScalarFunction(name, {LogicalType::DOUBLE, LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE,
	                      func);
}

// Helper function to create a binary scalar function (for sampling without seed)
template <typename FuncType>
ScalarFunction CreateBinaryFunction(const std::string &name, FuncType func) {
	return ScalarFunction(name, {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE, func, nullptr, nullptr,
	                      nullptr, nullptr, LogicalTypeId::INVALID, FunctionStability::VOLATILE,
	                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr);
}

// Helper function to create a single-parameter distribution function
template <typename FuncType>
ScalarFunction CreateTwoParameterDistributionFunction(const std::string &name, FuncType func) {
	return ScalarFunction(name, {LogicalType::DOUBLE, LogicalType::DOUBLE}, LogicalType::DOUBLE, func);
}

// Helper function to create a unary scalar function (for single-parameter sampling)
template <typename FuncType>
ScalarFunction CreateUnaryFunction(const std::string &name, const LogicalType result_type, FuncType func) {
	return ScalarFunction(name, {LogicalType::DOUBLE}, result_type, func, nullptr, nullptr, nullptr, nullptr,
	                      LogicalTypeId::INVALID, FunctionStability::VOLATILE,
	                      FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr);
}

// Helper function to register a three-parameter distribution function with description
template <typename FuncType>
void RegisterDistributionFunction(DatabaseInstance &instance, const std::string &name, FuncType func,
                                  const std::string &description, const std::string &example,
                                  const duckdb::vector<std::string> &param_names,
                                  const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateThreeParameterDistributionFunction(name, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Helper function to register a two-parameter sampling function with description
template <typename FuncType>
void RegisterSamplingFunction(DatabaseInstance &instance, const std::string &name, FuncType func,
                              const std::string &description, const std::string &example,
                              const duckdb::vector<std::string> &param_names,
                              const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateBinaryFunction(name, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Helper function to register a single-parameter distribution function with description
template <typename FuncType>
void RegisterSingleParamDistributionFunction(DatabaseInstance &instance, const std::string &name, FuncType func,
                                             const std::string &description, const std::string &example,
                                             const duckdb::vector<std::string> &param_names,
                                             const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateTwoParameterDistributionFunction(name, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Helper function to register a single-parameter sampling function with description
template <typename FuncType>
void RegisterSingleParamSamplingFunction(DatabaseInstance &instance, const std::string &name,
                                         const LogicalType &result_type, FuncType func, const std::string &description,
                                         const std::string &example, const duckdb::vector<std::string> &param_names,
                                         const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateUnaryFunction(name, result_type, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Helper function to register a two-parameter statistic function with description
template <typename FuncType>
void RegisterStatisticFunction(DatabaseInstance &instance, const std::string &name, FuncType func,
                               const std::string &description, const std::string &example,
                               const duckdb::vector<std::string> &param_names,
                               const duckdb::vector<LogicalType> &param_types) {
	CreateScalarFunctionInfo info(CreateBinaryFunction(name, func));
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Macro to register normal distribution functions with less boilerplate
#define REGISTER_NORMAL_FUNC(instance, func_name, func_ptr, description, example, third_param)                         \
	RegisterDistributionFunction(instance, func_name, func_ptr, description, example, {"mean", "stddev", third_param}, \
	                             normal_param_types)

// Macro to register normal distribution statistic functions with less boilerplate
#define REGISTER_NORMAL_STATISTIC_FUNC(instance, func_name, func_ptr, description, example)                            \
	RegisterStatisticFunction(instance, func_name, func_ptr, description, example, {"mean", "stddev"},                 \
	                          normal_statistic_param_types)

// Macro to register Bernoulli distribution functions with less boilerplate
#define REGISTER_BERNOULLI_FUNC(instance, func_name, func_ptr, description, example, second_param)                     \
	RegisterSingleParamDistributionFunction(instance, func_name, func_ptr, description, example, {"p", second_param},  \
	                                        bernoulli_param_types)

// Macro to register binomial distribution functions with less boilerplate
#define REGISTER_BINOMIAL_FUNC(instance, func_name, func_ptr, description, example, third_param)                       \
	RegisterDistributionFunction(instance, func_name, func_ptr, description, example, {"n", "p", third_param},         \
	                             binomial_param_types)

// Macro to register binomial distribution statistic functions with less boilerplate
#define REGISTER_BINOMIAL_STATISTIC_FUNC(instance, func_name, func_ptr, description, example)                          \
	RegisterStatisticFunction(instance, func_name, func_ptr, description, example, {"n", "p"},                         \
	                          binomial_statistic_param_types)

// Macro to register exponential distribution functions with less boilerplate
#define REGISTER_EXPONENTIAL_FUNC(instance, func_name, func_ptr, description, example, second_param)                   \
	RegisterSingleParamDistributionFunction(instance, func_name, func_ptr, description, example,                       \
	                                        {"lambda", second_param}, exponential_param_types)

// Macro to register exponential distribution statistic functions with less boilerplate
#define REGISTER_EXPONENTIAL_STATISTIC_FUNC(instance, func_name, func_ptr, description, example)                       \
	RegisterSingleParamDistributionFunction(instance, func_name, func_ptr, description, example, {"lambda"},           \
	                                        exponential_statistic_param_types)

// Generic sampling function template for single parameter distributions
template <typename DistributionType, typename ReturnType>
inline void GenericSingleParamSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &param_vector = args.data[0];

	if (param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto param = ConstantVector::GetData<double>(param_vector)[0];

		// Create distribution once and reuse for constant vectors
		DistributionType dist(param);
		const auto results = FlatVector::GetData<ReturnType>(result);

		for (size_t i = 0; i < args.size(); i++) {
			results[i] = static_cast<ReturnType>(dist(rng));
		}

		// Set vector type appropriately
		if (args.size() == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	// Handle non-constant vectors
	UnaryExecutor::Execute<double, ReturnType>(param_vector, result, args.size(), [&](double param) {
		DistributionType dist(param);
		return static_cast<ReturnType>(dist(rng));
	});
}

// Generic sampling function template for two parameter distributions
template <typename DistributionType, typename ReturnType>
inline void GenericTwoParamSampleFunc(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];

	BinaryExecutor::Execute<double, double, ReturnType>(param1_vector, param2_vector, result, args.size(),
	                                                    [&](double param1, double param2) {
		                                                    DistributionType dist(param1, param2);
		                                                    return static_cast<ReturnType>(dist(rng));
	                                                    });
}

void LoadDistributionNormal(DatabaseInstance &instance);
void LoadDistributionBernoulli(DatabaseInstance &instance);
void LoadDistributionBinomial(DatabaseInstance &instance);
void LoadDistributionExponential(DatabaseInstance &instance);

} // namespace duckdb