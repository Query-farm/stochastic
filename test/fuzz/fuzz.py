"""Fuzz every stochastic function with edge-case arguments. See README.md; run.sh drives the whole run.

  fuzz.py gen <functions.json> <out_dir> <shards>              write shard SQL files (scalar and aggregate calls)
  fuzz.py flat <functions.json> <logs_dir> <out_dir>           re-run successful scalar calls as table rows
  fuzz.py check <functions.json> <logs_dir> [<flat_logs_dir>]  report hangs, crashes, unexpected errors, bad values

Each statement is preceded by `.print Q<id>`; logs interleave stdout and stderr line by line.
"""
import itertools
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

random.seed(20261028)

DOUBLES = ["0.0", "-0.0", "1.0", "-1.0", "0.5", "2.5", "100.0", "1e-10", "0.9999999999999999", "1e15",
           "1e-300", "5e-324", "1e300", "-1e300", "1.7976931348623157e308", "'inf'", "'-inf'", "'nan'", "NULL"]
BIGINTS = ["0", "1", "-1", "2", "3", "10", "1000", "1000000000", "4611686018427387904", "9223372036854775807",
           "-9223372036854775808", "NULL"]
DOUBLE_LISTS = ["NULL", "[]", "[1.0]", "[0.5, 0.5]", "[2.0, 3.0, 4.0]", "[0.2, 0.3, 0.5]", "[-1.0, 2.0]", "[0.0, 1.0]",
                "['nan'::DOUBLE, 1.0]", "['inf'::DOUBLE, 1.0]", "[1e308, 1e308]", "[NULL, 1.0]", "[1e-300, 1e-300]",
                "[1e300, 1e-300]", "[5e-324, 5e-324, 5e-324]"]
BIGINT_LISTS = ["NULL", "[]", "[1]", "[2, 3, 5]", "[-1, 2]", "[0, 0]", "[NULL, 1]", "[9223372036854775807, 1]",
                "[4611686018427387904, 4611686018427387904]", "[1, 0, 2]"]
POOLS = {"DOUBLE": DOUBLES, "BIGINT": BIGINTS, "DOUBLE[]": DOUBLE_LISTS, "BIGINT[]": BIGINT_LISTS}
RANDOM_PER_FUNCTION = 300


def typed(value, type_):
    return f"CAST({value} AS {type_})"


def split_args(s):
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def base_args(fn):
    for example in fn["examples"]:
        m = re.match(rf"^(?:SELECT )?{fn['function_name']}\((.*)\)$", example.strip())
        if m:
            args = split_args(m.group(1))
            if len(args) == len(fn["parameter_types"]):
                return args
    return None


def scalar_calls(fn):
    types = fn["parameter_types"]
    base = base_args(fn)
    calls = set()
    if base:
        calls.add(tuple(base))
        for i, t in enumerate(types):
            for v in POOLS[t]:
                calls.add(tuple(base[:i] + [v] + base[i + 1:]))
    space = [POOLS[t] for t in types]
    total = math.prod(len(p) for p in space)
    if total <= RANDOM_PER_FUNCTION:
        calls.update(itertools.product(*space))
    else:
        for _ in range(RANDOM_PER_FUNCTION):
            calls.add(tuple(random.choice(p) for p in space))
    return sorted(calls)


def call_sql(fn, args):
    typed_args = ", ".join(typed(a, t) for a, t in zip(args, fn["parameter_types"]))
    return f"{fn['function_name']}({typed_args})"


# Aggregate datasets: the x column, plus extra columns for the aggregates that need them
X_SETS = {
    "empty": [],
    "one": ["1.0"],
    "two_equal": ["2.0", "2.0"],
    "zero_one": ["0.0", "1.0"],
    "small_ints": ["0.0", "1.0", "1.0", "2.0", "3.0", "5.0", "8.0"],
    "mixed": ["-2.5", "0.1", "0.7", "1.5", "3.0", "10.0"],
    "unit": ["0.1", "0.25", "0.5", "0.75", "0.9"],
    "positive": ["0.5", "1.0", "2.0", "4.0", "8.0"],
    "huge": ["1e300", "1.5e300", "1.7e308"],
    "tiny": ["1e-300", "2e-300", "5e-324"],
    "wide": ["1e-300", "1.0", "1e300"],
    "with_inf": ["1.0", "'inf'"],
    "with_nan": ["1.0", "'nan'"],
    "with_null": ["NULL", "1.0", "2.0", "NULL", "4.0"],
    "all_null": ["NULL", "NULL"],
    "negative_zero": ["-0.0", "0.0", "1.0"],
}


def values_source(xs):
    rows = ", ".join("(" + typed(x, "DOUBLE") + ")" for x in xs)
    return f"(VALUES {rows}) t(x)"


def aggregate_queries(fn):
    name, types = fn["function_name"], fn["parameter_types"]
    queries = []
    for set_name, xs in X_SETS.items():
        rows = xs if xs else None
        if len(types) == 1:
            source = (values_source(xs) if rows
                      else "(SELECT 1.0::DOUBLE AS x WHERE false) t")
            queries.append(f"SELECT {name}(x)::VARCHAR FROM {source}")
            continue
        if name in ("welch_t_test", "student_t_test"):
            for split in ("alternate", "all_zero"):
                vals = ["(" + typed(x, "DOUBLE") + f", {0 if split == 'all_zero' else i % 2})" for i, x in enumerate(xs)]
                source = f"(VALUES {', '.join(vals)}) t(x, g)" if vals else "(SELECT 1.0 AS x, 0 AS g WHERE false) t"
                queries.append(f"SELECT {name}(x, g)::VARCHAR FROM {source}")
            continue
        if name == "chi_squared_test":
            for expected in ("1.0", "0.0", "-1.0", "'inf'", "1e-300", "NULL"):
                vals = ["(" + typed(x, "DOUBLE") + ", " + typed(expected, "DOUBLE") + ")" for x in xs]
                source = f"(VALUES {', '.join(vals)}) t(x, e)" if vals else "(SELECT 1.0 AS x, 1.0 AS e WHERE false) t"
                queries.append(f"SELECT {name}(x, e)::VARCHAR FROM {source}")
            continue
        # Aggregates with constant parameters after x (KS tests, binomial fit, one-sample t-test)
        extra = types[1:]
        combos = set()
        base = [{"DOUBLE": "1.0", "BIGINT": "10"}[t] for t in extra]
        combos.add(tuple(base))
        for i, t in enumerate(extra):
            for v in POOLS[t]:
                combos.add(tuple(base[:i] + [v] + base[i + 1:]))
        for combo in sorted(combos):
            params = ", ".join(typed(v, t) for v, t in zip(combo, extra))
            source = (values_source(xs) if rows
                      else "(SELECT 1.0::DOUBLE AS x WHERE false) t")
            queries.append(f"SELECT {name}(x, {params})::VARCHAR FROM {source}")
    return queries


def load(path):
    return json.load(open(path))


def write_shards(statements, out_dir, shards, prefix):
    os.makedirs(out_dir, exist_ok=True)
    random.shuffle(statements)
    files = [open(f"{out_dir}/{prefix}{i:02d}.sql", "w") for i in range(shards)]
    for f in files:
        f.write("LOAD '{extension}';\n")
    for i, (qid, sql) in enumerate(statements):
        f = files[i % shards]
        f.write(f".print Q{qid}\n{sql};\n")
    for f in files:
        f.close()


def gen(functions_path, out_dir, shards):
    functions = load(functions_path)
    statements, index = [], {}
    for fn in functions:
        if fn["function_type"] == "scalar":
            for args in scalar_calls(fn):
                qid = len(index)
                index[qid] = {"function": fn["function_name"], "args": list(args), "kind": "scalar"}
                statements.append((qid, f"SELECT {call_sql(fn, args)}::VARCHAR"))
        else:
            for sql in aggregate_queries(fn):
                qid = len(index)
                index[qid] = {"function": fn["function_name"], "sql": sql, "kind": "aggregate"}
                statements.append((qid, sql))
    write_shards(statements, out_dir, shards, "shard")
    json.dump(index, open(f"{out_dir}/index.json", "w"))
    print(f"{len(statements)} statements in {shards} shards")


ERROR_RE = re.compile(r"^(?:Error: )?([A-Za-z ]+ Error|INTERNAL Error|Invalid Input Error):? ?(.*)$")


def parse_logs(logs_dir):
    """Map qid -> {"lines": [...]}; also return the qid running when each log ended abnormally."""
    records, crashes = {}, []
    for name in sorted(os.listdir(logs_dir)):
        if not name.endswith(".log"):
            continue
        current = None
        lines = open(f"{logs_dir}/{name}", errors="replace").read().splitlines()
        status = None
        for line in lines:
            m = re.match(r"^(HANG|CRASH (\d+)) Q(\d+)$", line)
            if m:
                crashes.append((name, "hang" if m.group(1) == "HANG" else f"signal {int(m.group(2)) - 128}",
                                int(m.group(3))))
                continue
            if line.startswith("EXIT_STATUS="):
                status = int(line.split("=", 1)[1])
                continue
            m = re.match(r"^Q(\d+)$", line)
            if m:
                current = int(m.group(1))
                records[current] = []
            elif current is not None and not line.startswith("hint:"):
                records[current].append(line)
        if status not in (0, 1, None):
            crashes.append((name, status, current))
    return records, crashes


def as_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def check_value(function, text):
    """Return a problem description for an out-of-range result, or None."""
    if text in ("NULL", ""):
        return None
    if function.endswith(("_samples", "_range", "_support")) or text.startswith("["):
        if "nan" in text.lower():
            return "nan in list result"
        return None
    v = as_float(text)
    if v is None:
        return None
    if math.isnan(v):
        return "nan result"
    suffix = function.rsplit("_", 1)[-1]
    is_log = "_log_" in function
    if not is_log and function.endswith(("_cdf", "_cdf_complement", "_pmf")) and not (0 <= v <= 1):
        return f"probability outside [0, 1]: {v}"
    if not is_log and function.endswith(("_pdf", "_hazard", "_chf", "_variance", "_stddev")) and v < 0:
        return f"negative value: {v}"
    if function.endswith(("_log_cdf", "_log_cdf_complement", "_log_pmf")) and v > 1e-12:
        return f"log probability above 0: {v}"
    return None


def check(functions_path, logs_dir, flat_dir=None):
    index = json.load(open(f"{logs_dir}/index.json"))
    records, crashes = parse_logs(logs_dir)
    problems = defaultdict(list)
    error_kinds = Counter()
    for name, status, qid in crashes:
        entry = index.get(str(qid), {})
        problems["crash or hang"].append(f"{name} exit {status} at Q{qid}: {entry}")
    missing = [qid for qid in index if int(qid) not in records]
    if missing:
        problems["not run"].append(f"{len(missing)} statements, e.g. {[index[q] for q in missing[:3]]}")
    for qid, lines in records.items():
        entry = index[str(qid)]
        out = [l for l in lines if l.strip()]
        errors = [l for l in out if ERROR_RE.match(l)]
        if errors:
            kind, message = ERROR_RE.match(errors[0]).groups()
            error_kinds[kind] += 1
            if kind != "Invalid Input Error":
                problems[f"unexpected {kind}"].append(f"{entry} -> {errors[0]}")
            elif not re.match(r"^[a-z_]+: ", message) and not message.startswith("dist_"):
                problems["error without distribution prefix"].append(f"{entry} -> {message}")
            if re.search(r"Error in function|boost::|std::", message):
                problems["leaked implementation detail"].append(f"{entry} -> {message}")
            continue
        if entry["kind"] == "aggregate" and out and re.search(r"\bnan\b", out[-1], re.I):
            problems["nan in aggregate result"].append(f"{entry['sql']} = {out[-1]}")
        if entry["kind"] == "scalar" and out:
            issue = check_value(entry["function"], out[-1])
            if issue:
                problems[issue.split(":")[0]].append(f"{entry['function']}({', '.join(entry['args'])}) = {out[-1]}")
    if flat_dir:
        flat_index = json.load(open(f"{flat_dir}/index.json"))
        flat_records, flat_crashes = parse_logs(flat_dir)
        for name, status, qid in flat_crashes:
            problems["flat crash or hang"].append(f"{name} exit {status} at Q{qid}: {flat_index.get(str(qid))}")
        for qid, lines in flat_records.items():
            entry = flat_index[str(qid)]
            out = [l for l in lines if l.strip()]
            if any(ERROR_RE.match(l) for l in out):
                problems["flat pass error"].append(f"{entry['function']}: {out[:2]}")
                continue
            if entry["function"].endswith(("_sample", "_samples")):
                continue
            for line in out:
                row, _, value = line.partition("|")
                expected = entry["expected"].get(row)
                if expected is not None and expected != value:
                    problems["flat result differs from constant"].append(
                        f"{entry['function']}({', '.join(entry['args'][row])}) constant {expected} flat {value}")
    succeeded = Counter()
    for qid, lines in records.items():
        if not any(ERROR_RE.match(l) for l in lines if l.strip()):
            succeeded[index[str(qid)]["function"]] += 1
    for fn in load(functions_path):
        if succeeded[fn["function_name"]] == 0:
            problems["no successful call"].append(fn["function_name"])
    print("error kinds:", dict(error_kinds))
    for kind, items in sorted(problems.items(), key=lambda kv: -len(kv[1])):
        print(f"\n== {kind}: {len(items)}")
        for item in items[:25]:
            print("  ", item[:400])


def flat(functions_path, logs_dir, out_dir):
    """Re-run every successful scalar call as one row of a per-function table and compare with the constant result."""
    functions = {f["function_name"]: f for f in load(functions_path)}
    index = json.load(open(f"{logs_dir}/index.json"))
    records, _ = parse_logs(logs_dir)
    ok = defaultdict(list)
    for qid, lines in records.items():
        entry = index[str(qid)]
        out = [l for l in lines if l.strip()]
        if entry["kind"] != "scalar" or not out or any(ERROR_RE.match(l) for l in out):
            continue
        # Samples differ between runs, and a rare invalid sample in one row would fail the whole batch
        if entry["function"].endswith(("_sample", "_samples")):
            continue
        ok[entry["function"]].append((entry["args"], out[-1]))
    statements, flat_index = [], {}
    for name, calls in ok.items():
        fn = functions[name]
        types = fn["parameter_types"]
        cols = [f"a{i}" for i in range(len(types))]
        rows = ", ".join(f"({row}, {', '.join(typed(a, t) for a, t in zip(args, types))})"
                         for row, (args, _) in enumerate(calls))
        sql = (f"SELECT r, {name}({', '.join(cols)})::VARCHAR FROM (VALUES {rows}) t(r, {', '.join(cols)}) ORDER BY r")
        qid = len(flat_index)
        flat_index[qid] = {"function": name, "args": {str(r): a for r, (a, _) in enumerate(calls)},
                           "expected": {str(r): v for r, (_, v) in enumerate(calls)}}
        statements.append((qid, sql))
    write_shards(statements, out_dir, 8, "flat")
    json.dump(flat_index, open(f"{out_dir}/index.json", "w"))
    print(f"{len(statements)} flat statements")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == "flat":
        flat(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        check(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
