#include "multivariate.hpp"

namespace duckdb {

namespace {

constexpr const char *DIRICHLET = "dirichlet";

void ReadAlpha(const ListIterator<double>::ValueEntry &entry, std::vector<double> &alpha) {
	ReadList<double>(entry, DIRICHLET, "alpha", alpha);
	if (alpha.size() < 2) {
		throw InvalidInputException("%s: alpha must have at least 2 elements, got %s", DIRICHLET,
		                            std::to_string(alpha.size()));
	}
	for (auto a : alpha) {
		RequirePositive(DIRICHLET, "alpha", a);
	}
}

double DirichletLogPdf(const std::vector<double> &alpha, const std::vector<double> &x) {
	if (x.size() != alpha.size()) {
		throw InvalidInputException("%s: x must have the same length as alpha (%s), got %s", DIRICHLET,
		                            std::to_string(alpha.size()), std::to_string(x.size()));
	}
	double sum_x = 0;
	for (auto v : x) {
		if (v < 0 || v > 1) {
			throw InvalidInputException("%s: x elements must be between 0 and 1, got %s", DIRICHLET,
			                            FormatParameter(v));
		}
		sum_x += v;
	}
	if (std::abs(sum_x - 1) > 1e-9) {
		throw InvalidInputException("%s: x must sum to 1, got %s", DIRICHLET, FormatParameter(sum_x));
	}
	double alpha_sum = 0;
	double result = 0;
	for (idx_t i = 0; i < alpha.size(); i++) {
		alpha_sum += alpha[i];
		result -= LogGamma(alpha[i]);
		if (alpha[i] != 1) {
			result += (alpha[i] - 1) * std::log(x[i]);
		}
	}
	return result + LogGamma(alpha_sum);
}

//! Mean alpha_i / a0 and variance alpha_i (a0 - alpha_i) / (a0^2 (a0 + 1)) of each component.
std::vector<double> DirichletMoments(const std::vector<double> &alpha, bool variance) {
	double a0 = 0;
	for (auto a : alpha) {
		a0 += a;
	}
	std::vector<double> result;
	for (auto a : alpha) {
		result.push_back(variance ? a * (a0 - a) / (a0 * a0 * (a0 + 1)) : a / a0);
	}
	return result;
}

template <class Result, class Fn>
void ExecuteAlpha(DataChunk &args, Vector &result, Fn fn) {
	auto alphas = ListValues<double>(args.data[0]);
	ResultOutput<Result> output(result, args.size(), false);
	std::vector<double> alpha;
	for (idx_t row = 0; row < args.size(); row++) {
		auto entry = alphas[row];
		if (!entry.IsValid()) {
			output.WriteNull();
			continue;
		}
		ReadAlpha(entry, alpha);
		output.Write(fn(alpha, row));
	}
}

template <class Fn>
void ExecuteDensity(DataChunk &args, Vector &result, Fn fn) {
	auto alphas = ListValues<double>(args.data[0]);
	auto xs = ListValues<double>(args.data[1]);
	ResultOutput<double> output(result, args.size(), false);
	std::vector<double> alpha;
	std::vector<double> x;
	for (idx_t row = 0; row < args.size(); row++) {
		auto alpha_entry = alphas[row];
		auto x_entry = xs[row];
		if (!alpha_entry.IsValid() || !x_entry.IsValid()) {
			output.WriteNull();
			continue;
		}
		ReadAlpha(alpha_entry, alpha);
		ReadList<double>(x_entry, DIRICHLET, "x", x);
		output.Write(fn(DirichletLogPdf(alpha, x)));
	}
}

} // namespace

void Load_dirichlet_distribution(ExtensionLoader &loader) {
	const auto list = LogicalType::LIST(LogicalType::DOUBLE);
	const vector<std::pair<string, LogicalType>> alpha_param = {{"alpha", list}};
	const vector<std::pair<string, LogicalType>> density_params = {{"alpha", list}, {"x", list}};
	const string example = "[2.0, 3.0, 5.0]";
	const string example_x = example + ", [0.2, 0.3, 0.5]";

	RegisterListFunction(
	    loader, DIRICHLET, "sample", FunctionStability::VOLATILE, alpha_param, list,
	    [](DataChunk &args, ExpressionState &state, Vector &result) {
		    RngAdapter rng(RandomEngine::Get(state.GetContext()));
		    ExecuteAlpha<std::vector<double>>(args, result, [&rng](const std::vector<double> &alpha, idx_t) {
			    // Normalized independent Gamma(alpha_i, 1) variates; redraw in the (vanishingly rare) all-zero case
			    std::vector<double> sample(alpha.size());
			    double total;
			    do {
				    total = 0;
				    for (idx_t i = 0; i < alpha.size(); i++) {
					    std::gamma_distribution<double> gamma(alpha[i], 1.0);
					    sample[i] = gamma(rng);
					    total += sample[i];
				    }
			    } while (total == 0);
			    for (auto &value : sample) {
				    value /= total;
			    }
			    return sample;
		    });
	    },
	    "Generates a random sample from the Dirichlet distribution with concentration parameters alpha. Returns a "
	    "list of probabilities that sums to 1.",
	    example, "sampling");

	RegisterListFunction(
	    loader, DIRICHLET, "pdf", FunctionStability::CONSISTENT, density_params, LogicalType::DOUBLE,
	    [](DataChunk &args, ExpressionState &, Vector &result) {
		    ExecuteDensity(args, result, [](double log_pdf) { return std::exp(log_pdf); });
	    },
	    "Computes the probability density function (PDF) of the Dirichlet distribution at x, a list of values in "
	    "[0, 1] that sums to 1.",
	    example_x, "density");
	RegisterListFunction(
	    loader, DIRICHLET, "log_pdf", FunctionStability::CONSISTENT, density_params, LogicalType::DOUBLE,
	    [](DataChunk &args, ExpressionState &, Vector &result) {
		    ExecuteDensity(args, result, [](double log_pdf) { return log_pdf; });
	    },
	    "Computes the natural logarithm of the probability density function (PDF) of the Dirichlet distribution at x.",
	    example_x, "density");

	RegisterListFunction(
	    loader, DIRICHLET, "mean", FunctionStability::CONSISTENT, alpha_param, list,
	    [](DataChunk &args, ExpressionState &, Vector &result) {
		    ExecuteAlpha<std::vector<double>>(
		        args, result, [](const std::vector<double> &alpha, idx_t) { return DirichletMoments(alpha, false); });
	    },
	    "Returns the mean of each component of the Dirichlet distribution, alpha_i / sum(alpha).", example, "property");
	RegisterListFunction(
	    loader, DIRICHLET, "variance", FunctionStability::CONSISTENT, alpha_param, list,
	    [](DataChunk &args, ExpressionState &, Vector &result) {
		    ExecuteAlpha<std::vector<double>>(
		        args, result, [](const std::vector<double> &alpha, idx_t) { return DirichletMoments(alpha, true); });
	    },
	    "Returns the variance of each component of the Dirichlet distribution.", example, "property");
}

} // namespace duckdb
