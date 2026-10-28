#pragma once
#include "duckdb.hpp"
#include <random>
#include <type_traits>

namespace duckdb {

template <typename T>
constexpr bool AlwaysFalse_v = !std::is_same<T, T>::value;

template <typename T>
struct logical_type_map {
	static LogicalType Get() {
		using fail_type = T;
		static_assert(AlwaysFalse_v<fail_type>, "Unsupported return type");
		return LogicalType {};
	}
};

template <>
struct logical_type_map<double> {
	static LogicalType Get() {
		return LogicalType::DOUBLE;
	}
};

template <>
struct logical_type_map<int64_t> {
	static LogicalType Get() {
		return LogicalType::BIGINT;
	}
};

template <>
struct logical_type_map<bool> {
	static LogicalType Get() {
		return LogicalType::BOOLEAN;
	}
};

template <>
struct logical_type_map<std::pair<int64_t, int64_t>> {
	static LogicalType Get() {
		return LogicalType::ARRAY(LogicalType::BIGINT, 2);
	}
};

template <>
struct logical_type_map<std::pair<double, double>> {
	static LogicalType Get() {
		return LogicalType::ARRAY(LogicalType::DOUBLE, 2);
	}
};

} // namespace duckdb