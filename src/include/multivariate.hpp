#pragma once

#include "utils.hpp"
#include <boost/math/special_functions/gamma.hpp>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Helpers for multivariate distributions whose parameters and values are LISTs
//===--------------------------------------------------------------------===//

template <class T>
using ListIterator = VectorIterator<VectorListType<T>>;

//! Reads a LIST argument; the list iterator does not accept dictionary vectors.
template <class T>
ListIterator<T> ListValues(Vector &vector) {
	if (vector.GetVectorType() == VectorType::DICTIONARY_VECTOR) {
		vector.Flatten();
	}
	return ListIterator<T>(vector);
}

//! Copies a non-NULL list into `out`, rejecting NULL and non-finite elements.
template <class T>
void ReadList(const typename ListIterator<T>::ValueEntry &entry, const char *dist, const char *param,
              std::vector<T> &out) {
	out.clear();
	out.reserve(entry.GetListLength());
	for (idx_t i = 0; i < entry.GetListLength(); i++) {
		auto child = entry.GetChildValue(i);
		if (!child.IsValid()) {
			throw InvalidInputException("%s: %s must not contain NULL elements", dist, param);
		}
		RequireFinite(dist, param, child.GetValue());
		out.push_back(child.GetValue());
	}
}

inline double LogGamma(double x) {
	return boost::math::lgamma(x, StochasticPolicy());
}

inline void RegisterListFunction(ExtensionLoader &loader, const char *dist, const string &name,
                                 FunctionStability stability, const vector<std::pair<string, LogicalType>> &params,
                                 const LogicalType &result_type, scalar_function_t func, const string &description,
                                 const string &example_args, const string &category) {
	const string final_name = string("dist_") + dist + "_" + name;
	vector<LogicalType> types;
	vector<string> names;
	for (auto &param : params) {
		names.push_back(param.first);
		types.push_back(param.second);
	}
	ScalarFunction function(Identifier(final_name), types, result_type, std::move(func), nullptr, nullptr, nullptr,
	                        LogicalTypeId::INVALID, stability, FunctionNullHandling::DEFAULT_NULL_HANDLING, nullptr);
	function.SetFallible();
	for (idx_t i = 0; i < names.size(); i++) {
		function.GetSignature().GetParameter(i).SetName(Identifier(names[i]));
	}
	CreateScalarFunctionInfo info(function);
	FunctionDescription desc;
	desc.description = description;
	desc.examples.push_back(final_name + "(" + example_args + ")");
	desc.parameter_types = types;
	desc.parameter_names = names;
	desc.categories = {"statistics", category, dist};
	info.descriptions.push_back(desc);
	loader.RegisterFunction(info);
}

} // namespace duckdb
