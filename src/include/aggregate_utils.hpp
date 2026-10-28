#pragma once

#include "duckdb.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/common/vector/struct_vector.hpp"
#include "duckdb/common/vector/vector_iterator.hpp"
#include "duckdb/common/vector/vector_writer.hpp"
#include "stochastic_policy.hpp"
#include <array>
#include <optional>
#include <tuple>
#include <utility>

namespace duckdb {

//===--------------------------------------------------------------------===//
// Aggregates returning a STRUCT of DOUBLEs
//===--------------------------------------------------------------------===//
// An aggregate operation OP provides:
//
//   static constexpr const char *name;                       // used in error messages
//   using inputs = std::tuple<double, ...>;                  // argument types; rows with a NULL are skipped
//   static constexpr std::array<const char *, K> result_names;
//   struct State { ... };                                    // trivially movable, zero-initialized
//   static void Update(State &state, inputs...);
//   static void Combine(const State &source, State &target);
//   static std::optional<std::array<double, K>> Finalize(State &state);  // nullopt gives NULL
//   static void Destroy(State &state);                       // frees anything the state owns

template <class T, std::size_t>
using RepeatType = T;

template <class T, class Indices>
struct RepeatedStruct;

template <class T, std::size_t... Is>
struct RepeatedStruct<T, std::index_sequence<Is...>> {
	using type = VectorStructType<RepeatType<T, Is>...>;
};

template <class OP>
struct StructAggregate {
	using State = typename OP::State;
	using Inputs = typename OP::inputs;
	static constexpr std::size_t result_count = OP::result_names.size();
	using Result = typename RepeatedStruct<double, std::make_index_sequence<result_count>>::type;

	template <class... Ts, std::size_t... Is>
	static void UpdateImpl(Vector inputs[], Vector &state_vector, idx_t count, std::tuple<Ts...> *,
	                       std::index_sequence<Is...>) {
		std::tuple<VectorIterator<Ts>...> values(inputs[Is]...);
		auto states = state_vector.Values<State *>();
		for (idx_t i = 0; i < count; i++) {
			if (!(std::get<Is>(values)[i].IsValid() && ...)) {
				continue;
			}
			OP::Update(*states[i].GetValue(), std::get<Is>(values)[i].GetValue()...);
		}
	}

	static void Update(Vector inputs[], AggregateInputData &, idx_t, Vector &state_vector, idx_t count) {
		try {
			UpdateImpl(inputs, state_vector, count, static_cast<Inputs *>(nullptr),
			           std::make_index_sequence<std::tuple_size<Inputs>::value> {});
		} catch (MathError &error) {
			throw InvalidInputException("%s: %s", OP::name, error.message);
		}
	}

	static void Combine(Vector &source, Vector &target, AggregateInputData &, idx_t count) {
		auto sources = source.Values<State *>();
		auto targets = target.Values<State *>();
		for (idx_t i = 0; i < count; i++) {
			OP::Combine(*sources[i].GetValue(), *targets[i].GetValue());
		}
	}

	static void Finalize(Vector &state_vector, AggregateFinalizeInputData &, Vector &result, idx_t count,
	                     idx_t offset) {
		auto states = state_vector.Values<State *>();
		auto writer = FlatVector::Writer<Result>(result, count, offset);
		try {
			for (idx_t i = 0; i < count; i++) {
				auto values = OP::Finalize(*states[i].GetValue());
				if (!values) {
					writer.WriteNull();
					continue;
				}
				writer.WriteValue([&](auto &...children) {
					idx_t child = 0;
					(children.WriteValue((*values)[child++]), ...);
				});
			}
		} catch (MathError &error) {
			throw InvalidInputException("%s: %s", OP::name, error.message);
		}
	}

	static void Destroy(Vector &state_vector, AggregateInputData &, idx_t count) {
		auto states = state_vector.Values<State *>();
		for (idx_t i = 0; i < count; i++) {
			OP::Destroy(*states[i].GetValue());
		}
	}
};

struct AggregateDoc {
	string description;
	string example;
	string category;
	//! Distribution name for distribution-specific aggregates, empty otherwise
	string distribution;
};

template <class OP>
void RegisterStructAggregate(ExtensionLoader &loader, const string &name,
                             const vector<std::pair<string, LogicalType>> &params, const AggregateDoc &doc) {
	using Agg = StructAggregate<OP>;
	child_list_t<LogicalType> children;
	for (auto result_name : OP::result_names) {
		children.emplace_back(result_name, LogicalType::DOUBLE);
	}
	AggregateFunction function(Identifier(name), {}, LogicalType::STRUCT(std::move(children)),
	                           AggregateFunction::StateSize<typename Agg::State>,
	                           AggregateFunction::StateInitialize<typename Agg::State, OP>, Agg::Update, Agg::Combine,
	                           Agg::Finalize, nullptr, nullptr, Agg::Destroy);
	vector<LogicalType> types;
	vector<string> names;
	for (auto &param : params) {
		function.GetSignature().AddParameter(Identifier(param.first), param.second);
		types.push_back(param.second);
		names.push_back(param.first);
	}
	function.SetFallible();

	CreateAggregateFunctionInfo info(function);
	FunctionDescription desc;
	desc.description = doc.description;
	desc.examples.push_back(doc.example);
	desc.parameter_types = types;
	desc.parameter_names = names;
	desc.categories = {"statistics", doc.category};
	if (!doc.distribution.empty()) {
		desc.categories.push_back(doc.distribution);
	}
	info.descriptions.push_back(desc);
	loader.RegisterFunction(info);
}

//! Welford's running mean and sum of squared deviations, mergeable across states (Chan et al.).
struct RunningMoments {
	double count;
	double mean;
	double m2;

	void Add(double x) {
		count += 1;
		const double delta = x - mean;
		mean += delta / count;
		m2 += delta * (x - mean);
	}

	void Merge(const RunningMoments &other) {
		if (other.count == 0) {
			return;
		}
		if (count == 0) {
			*this = other;
			return;
		}
		const double total = count + other.count;
		const double delta = other.mean - mean;
		mean += delta * other.count / total;
		m2 += other.m2 + delta * delta * count * other.count / total;
		count = total;
	}

	double SampleVariance() const {
		return m2 / (count - 1);
	}
};

//! Values collected by aggregates that need the whole sample (e.g. the Kolmogorov-Smirnov test).
struct CollectedValues {
	vector<double> *values;

	void Add(double x) {
		if (!values) {
			values = new vector<double>();
		}
		values->push_back(x);
	}

	void Merge(const CollectedValues &other) {
		if (other.values) {
			for (auto x : *other.values) {
				Add(x);
			}
		}
	}

	void Destroy() {
		delete values;
		values = nullptr;
	}

	idx_t size() const {
		return values ? values->size() : 0;
	}
};

//! Parameters of a test aggregate that must be the same for every row of a group.
template <idx_t N>
struct GroupParameters {
	bool set;
	double values[N];

	template <class... Ts>
	void Check(const char *name, Ts... params) {
		const double incoming[] = {static_cast<double>(params)...};
		if (!set) {
			for (idx_t i = 0; i < N; i++) {
				values[i] = incoming[i];
			}
			set = true;
			return;
		}
		for (idx_t i = 0; i < N; i++) {
			if (values[i] != incoming[i]) {
				throw InvalidInputException("%s: distribution parameters must be the same for every row of a group",
				                            name);
			}
		}
	}

	void Merge(const char *name, const GroupParameters &other) {
		if (!other.set) {
			return;
		}
		if (!set) {
			*this = other;
			return;
		}
		for (idx_t i = 0; i < N; i++) {
			if (values[i] != other.values[i]) {
				throw InvalidInputException("%s: distribution parameters must be the same for every row of a group",
				                            name);
			}
		}
	}
};

} // namespace duckdb
