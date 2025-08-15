#pragma once
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include "duckdb/common/vector_operations/generic_executor.hpp"
#include <boost/math/distributions.hpp>
#include <boost/random.hpp>
#include "rng_utils.hpp"
#include "distribution_traits.hpp"
#include <type_traits>
#include <utility> // std::declval
namespace duckdb {

template <typename DistributionType, typename FuncType>
void RegisterFunction(DatabaseInstance &instance, const std::string &name, const FunctionStability &stability,
                      const LogicalType &result_type, FuncType func, const std::string &description,
                      const std::string &example, vector<std::pair<string, LogicalType>> additional_params = {}) {

	vector<LogicalType> final_types;
	for (auto item : distribution_traits<DistributionType>::LogicalParamTypes()) {
		final_types.push_back(item);
	}
	for (auto item : additional_params) {
		final_types.push_back(item.second);
	}

	const auto final_name = string(distribution_traits<DistributionType>::prefix + "_" + name);
	const auto final_example = string(distribution_traits<DistributionType>::prefix + "_" + example);

	auto function =
	    ScalarFunction(final_name, final_types, result_type, func, nullptr, nullptr, nullptr, nullptr,
	                   LogicalTypeId::INVALID, stability, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr);

	CreateScalarFunctionInfo info(function);
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(final_example);

	desc.parameter_types = final_types;

	vector<string> final_names(begin(distribution_traits<DistributionType>::param_names),
	                           end(distribution_traits<DistributionType>::param_names));
	for (auto &item : additional_params) {
		final_names.push_back(item.first);
	}
	desc.parameter_names = final_names;

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

template <typename DistributionType, typename ReturnType>
inline void DistributionSampleBinary(DataChunk &args, ExpressionState &state, Vector &result) {
	using DistParam1 = typename distribution_traits<DistributionType>::param1_t;
	using DistParam2 = typename distribution_traits<DistributionType>::param2_t;

	auto &param1_vector = args.data[0];
	auto &param2_vector = args.data[1];

	if (param1_vector.GetVectorType() == VectorType::CONSTANT_VECTOR &&
	    param2_vector.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		if (ConstantVector::IsNull(param1_vector) || ConstantVector::IsNull(param2_vector)) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
			ConstantVector::SetNull(result, true);
			return;
		}

		const auto param1 = ConstantVector::GetData<DistParam1>(param1_vector)[0];
		const auto param2 = ConstantVector::GetData<DistParam2>(param2_vector)[0];

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
	BinaryExecutor::Execute<DistParam1, DistParam2, ReturnType>(param1_vector, param2_vector, result, args.size(),
	                                                            [&](DistParam1 param1, DistParam2 param2) {
		                                                            DistributionType dist(param1, param2);
		                                                            return dist(rng);
	                                                            });
}

template <typename DistributionType, typename CallParam, typename Func>
inline void DistributionCallBinaryUnary(DataChunk &args, ExpressionState &state, Vector &result, Func op) {

	using DistParam1 = typename distribution_traits<DistributionType>::param1_t;
	using DistParam2 = typename distribution_traits<DistributionType>::param2_t;
	using ReturnType = decltype(op(std::declval<Vector &>(), std::declval<DistributionType &>()));

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

//
// Generic function template for single parameter distributions with single call parameter
template <typename DistributionType, typename Func>
inline void DistributionCallBinaryNone(DataChunk &args, ExpressionState &state, Vector &result, Func op) {
	using DistParam1 = typename distribution_traits<DistributionType>::param1_t;
	using DistParam2 = typename distribution_traits<DistributionType>::param2_t;

	auto &dist_param1_vector = args.data[0];
	auto &dist_param2_vector = args.data[1];

	using FuncReturnType = decltype(op(std::declval<Vector &>(), std::declval<DistributionType &>()));

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
		if constexpr (std::is_same_v<FuncReturnType, std::pair<double, double>>) {
			auto &result_data_children = ArrayVector::GetEntry(result);
			auto data_ptr = FlatVector::GetData<double>(result_data_children);
			auto array_result = op(result, dist);
			data_ptr[0] = array_result.first;
			data_ptr[1] = array_result.second;
		} else {
			auto result_data = ConstantVector::GetData<FuncReturnType>(result);
			result_data[0] = op(result, dist);
		}
		return;
	}

	// We need to specialize here because GenericExecutor doesn't support std::pair
	// as a return type into an array.
	if constexpr (std::is_same_v<FuncReturnType, std::pair<double, double>>) {
		UnifiedVectorFormat dist_param1_data;
		UnifiedVectorFormat dist_param2_data;

		dist_param1_vector.ToUnifiedFormat(args.size(), dist_param1_data);
		dist_param2_vector.ToUnifiedFormat(args.size(), dist_param2_data);

		result.SetVectorType(VectorType::FLAT_VECTOR);

		auto &result_data_children = ArrayVector::GetEntry(result);
		auto result_data = FlatVector::GetData<double>(result_data_children);

		if (!dist_param1_data.validity.AllValid() || !dist_param1_data.validity.AllValid()) {
			auto result_validity = FlatVector::Validity(result);
			for (idx_t i = 0; i < args.size(); i++) {
				auto dist_param1_index = dist_param1_data.sel->get_index(i);
				auto dist_param2_index = dist_param2_data.sel->get_index(i);
				if (dist_param1_data.validity.RowIsValid(dist_param1_index) &&
				    dist_param2_data.validity.RowIsValid(dist_param2_index)) {
					auto dist_param1_entry =
					    UnifiedVectorFormat::GetData<DistParam1>(dist_param1_data)[dist_param1_index];
					auto dist_param2_entry =
					    UnifiedVectorFormat::GetData<DistParam2>(dist_param2_data)[dist_param2_index];

					DistributionType dist(dist_param1_entry, dist_param2_entry);
					auto op_result = op(result, dist);
					result_data[i * 2] = op_result.first;
					result_data[i * 2 + 1] = op_result.second;
				} else {
					result_validity.SetInvalid(i);
				}
			}
		} else {
			auto dist_param1_entries = UnifiedVectorFormat::GetData<DistParam1>(dist_param1_data);
			auto dist_param2_entries = UnifiedVectorFormat::GetData<DistParam2>(dist_param2_data);
			for (idx_t i = 0; i < args.size(); i++) {
				auto dist_param1_entry = dist_param1_entries[dist_param1_data.sel->get_index(i)];
				auto dist_param2_entry = dist_param2_entries[dist_param2_data.sel->get_index(i)];
				DistributionType dist(dist_param1_entry, dist_param2_entry);
				auto op_result = op(result, dist);
				result_data[i * 2] = op_result.first;
				result_data[i * 2 + 1] = op_result.second;
			}
		}
	} else {
		BinaryExecutor::Execute<DistParam1, DistParam2, FuncReturnType>(
		    dist_param1_vector, dist_param2_vector, result, args.size(),
		    [&](DistParam1 dist_param1, DistParam2 dist_param2) {
			    DistributionType dist(dist_param1, dist_param2);
			    return op(result, dist);
		    });
	}
}

void LoadDistributionNormal(DatabaseInstance &instance);

} // namespace duckdb