#include "multivariate.hpp"

namespace duckdb {

namespace {

constexpr const char *MULTINOMIAL = "multinomial";

void ValidateTrials(int64_t trials) {
	RequireNonNegative(MULTINOMIAL, "trials", trials);
	if (static_cast<double>(trials) >= std::ldexp(1.0, 53)) {
		throw InvalidInputException("%s: trials must be < 2^53, got %s", MULTINOMIAL, std::to_string(trials));
	}
}

//! Reads probs, which must be in [0, 1] and sum to 1 (within 1e-9); they are renormalized to sum exactly to 1.
void ReadProbs(const ListIterator<double>::ValueEntry &entry, std::vector<double> &probs) {
	ReadList<double>(entry, MULTINOMIAL, "probs", probs);
	if (probs.empty()) {
		throw InvalidInputException("%s: probs must not be empty", MULTINOMIAL);
	}
	double total = 0;
	for (auto p : probs) {
		RequireProbability(MULTINOMIAL, "probs", p);
		total += p;
	}
	if (std::abs(total - 1) > 1e-9) {
		throw InvalidInputException("%s: probs must sum to 1, got %s", MULTINOMIAL, FormatParameter(total));
	}
	for (auto &p : probs) {
		p /= total;
	}
}

double MultinomialLogPmf(int64_t trials, const std::vector<double> &probs, const std::vector<int64_t> &x) {
	if (x.size() != probs.size()) {
		throw InvalidInputException("%s: x must have the same length as probs (%s), got %s", MULTINOMIAL,
		                            std::to_string(probs.size()), std::to_string(x.size()));
	}
	int64_t total = 0;
	for (auto count : x) {
		if (count < 0) {
			return -std::numeric_limits<double>::infinity();
		}
		total += count;
	}
	if (total != trials) {
		return -std::numeric_limits<double>::infinity();
	}
	double result = LogGamma(static_cast<double>(trials) + 1);
	for (idx_t i = 0; i < x.size(); i++) {
		result -= LogGamma(static_cast<double>(x[i]) + 1);
		if (x[i] > 0) {
			if (probs[i] == 0) {
				return -std::numeric_limits<double>::infinity();
			}
			result += static_cast<double>(x[i]) * std::log(probs[i]);
		}
	}
	return result;
}

template <class Result, class Fn>
void ExecuteParams(DataChunk &args, Vector &result, Fn fn) {
	auto trials_values = args.data[0].Values<int64_t>();
	auto probs_values = ListValues<double>(args.data[1]);
	ResultOutput<Result> output(result, args.size(), false);
	std::vector<double> probs;
	for (idx_t row = 0; row < args.size(); row++) {
		auto trials = trials_values[row];
		auto probs_entry = probs_values[row];
		if (!trials.IsValid() || !probs_entry.IsValid()) {
			output.WriteNull();
			continue;
		}
		ValidateTrials(trials.GetValue());
		ReadProbs(probs_entry, probs);
		output.Write(fn(trials.GetValue(), probs));
	}
}

template <class Fn>
void ExecuteMass(DataChunk &args, Vector &result, Fn fn) {
	auto trials_values = args.data[0].Values<int64_t>();
	auto probs_values = ListValues<double>(args.data[1]);
	auto x_values = ListValues<int64_t>(args.data[2]);
	ResultOutput<double> output(result, args.size(), false);
	std::vector<double> probs;
	std::vector<int64_t> x;
	for (idx_t row = 0; row < args.size(); row++) {
		auto trials = trials_values[row];
		auto probs_entry = probs_values[row];
		auto x_entry = x_values[row];
		if (!trials.IsValid() || !probs_entry.IsValid() || !x_entry.IsValid()) {
			output.WriteNull();
			continue;
		}
		ValidateTrials(trials.GetValue());
		ReadProbs(probs_entry, probs);
		ReadList<int64_t>(x_entry, MULTINOMIAL, "x", x);
		output.Write(fn(MultinomialLogPmf(trials.GetValue(), probs, x)));
	}
}

} // namespace

void Load_multinomial_distribution(ExtensionLoader &loader) {
	const auto probs_type = LogicalType::LIST(LogicalType::DOUBLE);
	const auto counts_type = LogicalType::LIST(LogicalType::BIGINT);
	const vector<std::pair<string, LogicalType>> params = {{"trials", LogicalType::BIGINT}, {"probs", probs_type}};
	const vector<std::pair<string, LogicalType>> mass_params = {
	    {"trials", LogicalType::BIGINT}, {"probs", probs_type}, {"x", counts_type}};
	const string example = "10, [0.2, 0.3, 0.5]";
	const string example_x = example + ", [2, 3, 5]";

	RegisterListFunction(
	    loader, MULTINOMIAL, "sample", FunctionStability::VOLATILE, params, counts_type,
	    [](DataChunk &args, ExpressionState &state, Vector &result) {
		    RngAdapter rng(RandomEngine::Get(state.GetContext()));
		    ExecuteParams<std::vector<int64_t>>(args, result, [&rng](int64_t trials, const std::vector<double> &probs) {
			    // Conditional binomials: x_i ~ Binomial(remaining trials, p_i / remaining probability)
			    std::vector<int64_t> counts(probs.size(), 0);
			    int64_t remaining = trials;
			    double remaining_prob = 1;
			    for (idx_t i = 0; i + 1 < probs.size() && remaining > 0; i++) {
				    const double p = remaining_prob > 0 ? std::min(1.0, probs[i] / remaining_prob) : 0;
				    std::binomial_distribution<int64_t> binomial(remaining, p);
				    counts[i] = binomial(rng);
				    remaining -= counts[i];
				    remaining_prob -= probs[i];
			    }
			    counts.back() += remaining;
			    return counts;
		    });
	    },
	    "Generates a random sample from the multinomial distribution: the counts of each outcome in trials draws "
	    "with outcome probabilities probs. Returns a list of counts that sums to trials.",
	    example, "sampling");

	auto pmf = [](DataChunk &args, ExpressionState &, Vector &result) {
		ExecuteMass(args, result, [](double log_pmf) { return std::exp(log_pmf); });
	};
	auto log_pmf = [](DataChunk &args, ExpressionState &, Vector &result) {
		ExecuteMass(args, result, [](double log_pmf) { return log_pmf; });
	};
	for (auto name : {"pmf", "pdf"}) {
		RegisterListFunction(loader, MULTINOMIAL, name, FunctionStability::CONSISTENT, mass_params, LogicalType::DOUBLE,
		                     pmf,
		                     "Computes the probability mass function (PMF) of the multinomial distribution: the "
		                     "probability of observing the counts x.",
		                     example_x, "density");
	}
	for (auto name : {"log_pmf", "log_pdf"}) {
		RegisterListFunction(loader, MULTINOMIAL, name, FunctionStability::CONSISTENT, mass_params, LogicalType::DOUBLE,
		                     log_pmf,
		                     "Computes the natural logarithm of the probability mass function (PMF) of the "
		                     "multinomial distribution at the counts x.",
		                     example_x, "density");
	}

	RegisterListFunction(
	    loader, MULTINOMIAL, "mean", FunctionStability::CONSISTENT, params, probs_type,
	    [](DataChunk &args, ExpressionState &, Vector &result) {
		    ExecuteParams<std::vector<double>>(args, result, [](int64_t trials, const std::vector<double> &probs) {
			    std::vector<double> mean;
			    for (auto p : probs) {
				    mean.push_back(static_cast<double>(trials) * p);
			    }
			    return mean;
		    });
	    },
	    "Returns the mean count of each outcome of the multinomial distribution, trials * probs_i.", example,
	    "property");
	RegisterListFunction(
	    loader, MULTINOMIAL, "variance", FunctionStability::CONSISTENT, params, probs_type,
	    [](DataChunk &args, ExpressionState &, Vector &result) {
		    ExecuteParams<std::vector<double>>(args, result, [](int64_t trials, const std::vector<double> &probs) {
			    std::vector<double> variance;
			    for (auto p : probs) {
				    variance.push_back(static_cast<double>(trials) * p * (1 - p));
			    }
			    return variance;
		    });
	    },
	    "Returns the variance of the count of each outcome of the multinomial distribution, trials * p_i * (1 - p_i).",
	    example, "property");
}

} // namespace duckdb
