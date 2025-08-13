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

template <typename FuncType>
void RegisterFunction(DatabaseInstance &instance, const std::string &name, const FunctionStability &stability,
                      const vector<LogicalType> &param_types, const LogicalType &result_type, FuncType func,
                      const std::string &description, const std::string &example,
                      const duckdb::vector<std::string> &param_names) {
	auto function =
	    ScalarFunction(name, param_types, result_type, func, nullptr, nullptr, nullptr, nullptr, LogicalTypeId::INVALID,
	                   stability, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr);

	CreateScalarFunctionInfo info(function);
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(example);
	desc.parameter_names = param_names;
	desc.parameter_types = param_types;
	info.descriptions.push_back(desc);
	ExtensionUtil::RegisterFunction(instance, info);
}

// Generic sampling function template for single parameter distributions
template <typename DistributionType, typename Param1Type, typename ReturnType>
inline void DistributionSampleUnary(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &param_vector = args.data[0];

	if (param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto param = ConstantVector::GetData<Param1Type>(param_vector)[0];

		// Create distribution once and reuse for constant vectors
		DistributionType dist(param);
		const auto results = FlatVector::GetData<ReturnType>(result);

		for (size_t i = 0; i < args.size(); i++) {
			results[i] = dist(rng);
		}

		// Set vector type appropriately
		if (args.size() == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	// Handle non-constant vectors
	UnaryExecutor::Execute<Param1Type, ReturnType>(param_vector, result, args.size(), [&](Param1Type param) {
		DistributionType dist(param);
		return dist(rng);
	});
}

template <typename DistributionType, typename Param1Type, typename Param2Type, typename ReturnType>
inline void DistributionSampleBinary(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];

	if (param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param1_vector) || ConstantVector::IsNull(param2_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto param1 = ConstantVector::GetData<Param1Type>(param1_vector)[0];
		const auto param2 = ConstantVector::GetData<Param2Type>(param2_vector)[0];

		// Create distribution once and reuse for constant vectors
		DistributionType dist(param1, param2);
		const auto results = FlatVector::GetData<ReturnType>(result);

		for (size_t i = 0; i < args.size(); i++) {
			results[i] = dist(rng);
		}

		// Set vector type appropriately
		if (args.size() == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	// Handle non-constant vectors
	BinaryExecutor::Execute<Param1Type, Param2Type, ReturnType>(param1_vector, param2_vector, result, args.size(),
	                                                            [&](Param1Type param1, Param2Type param2) {
		                                                            DistributionType dist(param1, param2);
		                                                            return dist(rng);
	                                                            });
}

template <typename DistributionType, typename Param1Type, typename Param2Type, typename Param3Type, typename ReturnType>
inline void DistributionSampleTernary(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];
	auto &param3_vector = args.data[2];

	if (param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    param3_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param1_vector) || ConstantVector::IsNull(param2_vector) ||
		    ConstantVector::IsNull(param3_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto param1 = ConstantVector::GetData<Param1Type>(param1_vector)[0];
		const auto param2 = ConstantVector::GetData<Param2Type>(param2_vector)[0];
		const auto param3 = ConstantVector::GetData<Param3Type>(param3_vector)[0];

		// Create distribution once and reuse for constant vectors
		DistributionType dist(param1, param2, param3);
		const auto results = FlatVector::GetData<ReturnType>(result);

		for (size_t i = 0; i < args.size(); i++) {
			results[i] = dist(rng);
		}

		// Set vector type appropriately
		if (args.size() == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
		return;
	}

	TernaryExecutor::Execute<Param1Type, Param2Type, Param3Type, ReturnType>(
	    param1_vector, param2_vector, param3_vector, result, args.size(),
	    [&](Param1Type param1, Param2Type param2, Param3Type param3) {
		    DistributionType dist(param1, param2, param3);
		    return dist(rng);
	    });
}

// Generic function template for single parameter distributions with single call parameter
template <typename DistributionType, typename DistParam, typename CallParam, typename ReturnType, typename Operation>
inline void DistributionCallUnaryUnary(DataChunk &args, ExpressionState &state, Vector &result, Operation op) {
	auto &dist_param_vector = args.data[0];
	auto &call_param_vector = args.data[1];

	// Handle constant vectors optimization
	if (dist_param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    call_param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {

		if (ConstantVector::IsNull(dist_param_vector) || ConstantVector::IsNull(call_param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto dist_param = ConstantVector::GetData<DistParam>(dist_param_vector)[0];
		const auto call_param = ConstantVector::GetData<CallParam>(call_param_vector)[0];

		// Create distribution once for constant case
		DistributionType dist(dist_param);
		auto constant_result = op(dist, call_param);

		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<ReturnType>(result);
		result_data[0] = constant_result;
		return;
	}

	// Handle distribution parameter constant, call parameter varying
	if (dist_param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(dist_param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto dist_param = ConstantVector::GetData<DistParam>(dist_param_vector)[0];
		DistributionType dist(dist_param);

		UnaryExecutor::Execute<CallParam, ReturnType>(call_param_vector, result, args.size(),
		                                              [&](CallParam call_param) { return op(dist, call_param); });
		return;
	}

	// Handle general case where both parameters vary
	BinaryExecutor::Execute<DistParam, CallParam, ReturnType>(dist_param_vector, call_param_vector, result, args.size(),
	                                                          [&](DistParam dist_param, CallParam call_param) {
		                                                          DistributionType dist(dist_param);
		                                                          return op(dist, call_param);
	                                                          });
}

// Generic function template for single parameter distributions with single call parameter
template <typename DistributionType, typename DistParam1, typename DistParam2, typename CallParam, typename ReturnType,
          typename Func>
inline void DistributionCallBinaryUnary(DataChunk &args, ExpressionState &state, Vector &result, Func op) {
	auto &dist_param1_vector = args.data[0];
	auto &dist_param2_vector = args.data[1];
	auto &call_param_vector = args.data[2];

	// Handle constant vectors optimization
	if (dist_param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    dist_param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    call_param_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {

		if (ConstantVector::IsNull(dist_param1_vector) || ConstantVector::IsNull(dist_param2_vector) ||
		    ConstantVector::IsNull(call_param_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto dist_param1 = ConstantVector::GetData<DistParam1>(dist_param1_vector)[0];
		const auto dist_param2 = ConstantVector::GetData<DistParam2>(dist_param2_vector)[0];
		const auto call_param = ConstantVector::GetData<CallParam>(call_param_vector)[0];

		// Create distribution once for constant case
		DistributionType dist(dist_param1, dist_param2);

		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<ReturnType>(result);
		result_data[0] = op(dist, call_param);
		return;
	}

	// Handle distribution parameter constant, call parameter varying
	if (dist_param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    dist_param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(dist_param1_vector) || ConstantVector::IsNull(dist_param2_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto dist_param1 = ConstantVector::GetData<DistParam1>(dist_param1_vector)[0];
		const auto dist_param2 = ConstantVector::GetData<DistParam2>(dist_param2_vector)[0];
		DistributionType dist(dist_param1, dist_param2);

		UnaryExecutor::Execute<CallParam, ReturnType>(call_param_vector, result, args.size(),
		                                              [&](CallParam call_param) { return op(dist, call_param); });
		return;
	}

	// Handle general case where both parameters vary
	TernaryExecutor::Execute<DistParam1, DistParam2, CallParam, ReturnType>(
	    dist_param1_vector, dist_param2_vector, call_param_vector, result, args.size(),
	    [&](DistParam1 dist_param1, DistParam2 dist_param2, CallParam call_param) {
		    DistributionType dist(dist_param1, dist_param2);
		    return op(dist, call_param);
	    });
}

// Generic function template for single parameter distributions with single call parameter
template <typename DistributionType, typename DistParam1, typename DistParam2, typename ReturnType, typename Func>
inline void DistributionCallBinaryNone(DataChunk &args, ExpressionState &state, Vector &result, Func op) {
	auto &dist_param1_vector = args.data[0];
	auto &dist_param2_vector = args.data[1];

	// Handle constant vectors optimization
	if (dist_param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    dist_param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {

		if (ConstantVector::IsNull(dist_param1_vector) || ConstantVector::IsNull(dist_param2_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto dist_param1 = ConstantVector::GetData<DistParam1>(dist_param1_vector)[0];
		const auto dist_param2 = ConstantVector::GetData<DistParam2>(dist_param2_vector)[0];

		// Create distribution once for constant case
		DistributionType dist(dist_param1, dist_param2);

		result.SetVectorType(VectorType::CONSTANT_VECTOR);
		auto result_data = ConstantVector::GetData<ReturnType>(result);
		result_data[0] = op(dist);
		return;
	}

	// Handle general case where both parameters vary
	BinaryExecutor::Execute<DistParam1, DistParam2, ReturnType>(dist_param1_vector, dist_param2_vector, result,
	                                                            args.size(),
	                                                            [&](DistParam1 dist_param1, DistParam2 dist_param2) {
		                                                            DistributionType dist(dist_param1, dist_param2);
		                                                            return op(dist);
	                                                            });
}

void LoadDistributionNormal(DatabaseInstance &instance);

} // namespace duckdb