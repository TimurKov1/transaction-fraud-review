import argparse
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import DATABASE, IEEE_DATABASE, ROOT

REPORTS = ROOT / "reports"
MODEL_PATH = ROOT / "artifacts" / "catboost_history.cbm"
IEEE_MODEL_PATH = ROOT / "artifacts" / "ieee_catboost_history.cbm"
CURRENT = ["amount", "hour", "weekday"]
HISTORY = CURRENT + [
    "count_1h", "count_24h", "count_7d", "mean_amount_7d",
    "amount_to_mean_7d", "seconds_since_previous",
]
TRAIN_START = date(2018, 4, 8)
VALIDATION_START = date(2018, 7, 24)
TEST_START = date(2018, 8, 24)
TEST_END = date(2018, 10, 1)
BUDGETS = [10, 25, 50, 100, 150, 200]


def fetch_period(db, start, end, labels=True):
    columns = ["tx_id", "tx_time", *HISTORY]
    if labels:
        columns.extend(["fraud", "scenario"])
    return db.execute(
        f"""SELECT {', '.join(columns)} FROM features
        WHERE tx_time >= ? AND tx_time < ?
        ORDER BY tx_time, tx_id""",
        [start, end],
    ).df()


def fit_catboost(train, columns):
    model = CatBoostClassifier(
        iterations=240, depth=6, learning_rate=0.12, random_seed=42,
        thread_count=4, allow_writing_files=False, verbose=False,
    )
    return model.fit(train[columns], train["fraud"])


def ranked(frame, scores):
    return (
        frame.assign(score=scores, day=frame["tx_time"].dt.normalize())
        .sort_values(["day", "score", "tx_time", "tx_id"],
                     ascending=[True, False, True, True])
        .assign(rank=lambda rows: rows.groupby("day").cumcount() + 1)
    )


def threshold_at_budget(scores, capacity):
    # Include a tied score only if its entire group fits the budget.
    counts = scores.value_counts().sort_index(ascending=False)
    eligible = counts.index[counts.cumsum() <= capacity]
    return float(eligible[-1]) if len(eligible) else float("inf")


def summarize(rows, selected, days):
    alerts = rows.loc[selected]
    frauds = rows.loc[rows["fraud"].eq(1)]
    detected = alerts.loc[alerts["fraud"].eq(1)]
    daily = alerts.groupby("day").size()
    return {
        "alerts": len(alerts),
        "alerts_per_day": len(alerts) / days,
        "peak_alerts_per_day": int(daily.max()) if len(daily) else 0,
        "detected_fraud": len(detected),
        "legitimate_alerts": len(alerts) - len(detected),
        "precision": len(detected) / len(alerts) if len(alerts) else 0.0,
        "recall": len(detected) / len(frauds),
        "fraud_amount_recall": detected["amount"].sum() / frauds["amount"].sum(),
    }


def write_profiles(db, start, end, reports):
    query = (ROOT / "sql" / "profile.sql").read_text()
    db.execute(query, {"start": start, "end": end}).df().to_csv(
        reports / "train_profile.csv", index=False,
    )


def plot_budget_curve(curve, path, reference_budget=100):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {
        "Amount rule": "#64748b",
        "Logistic + history": "#b45309",
        "CatBoost current": "#0f766e",
        "CatBoost + history": "#4338ca",
    }
    figure, axes = plt.subplots(figsize=(8.6, 4.8), layout="constrained")
    for name, rows in curve.groupby("model", sort=False):
        axes.plot(rows["budget"], rows["recall"], marker="o", markersize=5,
                  color=colors[name], linewidth=3 if name == "CatBoost + history" else 1.8,
                  label=name)
    axes.axvline(reference_budget, color="#94a3b8", linestyle="--", linewidth=1, zorder=0)
    axes.set(xlabel="Transactions reviewed per day", ylabel="Share of fraud detected")
    axes.set_title("Fraud detection by review budget", loc="left", fontweight="bold", pad=16)
    axes.set_ylim(bottom=0)
    axes.set_xticks(sorted(curve["budget"].unique()))
    axes.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes.grid(axis="y", color="#e2e8f0")
    axes.set_axisbelow(True)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["bottom", "left"]].set_color("#94a3b8")
    axes.legend(frameon=False, loc="best")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def split_dates(db, dataset):
    if dataset == "simulation":
        return TRAIN_START, VALIDATION_START, TEST_START, TEST_END
    first, last = db.execute("SELECT min(tx_time)::DATE, max(tx_time)::DATE FROM features").fetchone()
    start = first + timedelta(days=7)
    end = last + timedelta(days=1)
    days = (end - start).days
    if days < 10:
        raise ValueError("IEEE-CIS needs at least 10 days of data after the 7-day warm-up.")
    return start, start + timedelta(days=days * 65 // 100), start + timedelta(days=days * 80 // 100), end


def experiment(budget, dataset="simulation"):
    database, model_path, reports = (DATABASE, MODEL_PATH, REPORTS)
    if dataset == "ieee":
        database, model_path, reports = (IEEE_DATABASE, IEEE_MODEL_PATH, REPORTS / "ieee")
    reports.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(exist_ok=True)
    with duckdb.connect(str(database), read_only=True) as db:
        start, validation_start, test_start, end = split_dates(db, dataset)
        train = fetch_period(db, start, validation_start)
        validation = fetch_period(db, validation_start, test_start)
        test = fetch_period(db, test_start, end)
        write_profiles(db, start, validation_start, reports)

    for name, period in [("train", train), ("validation", validation), ("test", test)]:
        if period["fraud"].nunique() != 2:
            raise ValueError(f"{name} must include both legitimate and fraudulent transactions.")
        print(f"{name}: {len(period):,} transactions; {period['fraud'].sum():,} fraud", flush=True)
    print(f"Period boundaries: {start}, {validation_start}, {test_start}, {end}", flush=True)

    logistic = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
    ).fit(train[HISTORY], train["fraud"])
    current = fit_catboost(train, CURRENT)
    history = fit_catboost(train, HISTORY)
    history.save_model(str(model_path))

    scores = {
        "Amount rule": (validation["amount"], test["amount"]),
        "Logistic + history": (
            logistic.predict_proba(validation[HISTORY])[:, 1],
            logistic.predict_proba(test[HISTORY])[:, 1],
        ),
        "CatBoost current": (
            current.predict_proba(validation[CURRENT])[:, 1],
            current.predict_proba(test[CURRENT])[:, 1],
        ),
        "CatBoost + history": (
            history.predict_proba(validation[HISTORY])[:, 1],
            history.predict_proba(test[HISTORY])[:, 1],
        ),
    }

    rows, curve, scenarios = [], [], []
    test_days = (end - test_start).days
    validation_capacity = budget * (test_start - validation_start).days
    for name, (validation_scores, test_scores) in scores.items():
        test_rows = ranked(test, test_scores)
        threshold = threshold_at_budget(pd.Series(validation_scores), validation_capacity)
        ap = average_precision_score(test["fraud"], test_scores)
        for policy, selected in {
            "daily_top_k": test_rows["rank"].le(budget),
            "fixed_threshold": test_rows["score"].ge(threshold),
        }.items():
            rows.append({
                "model": name, "policy": policy, "budget": budget, "average_precision": ap,
                "threshold": threshold if policy == "fixed_threshold" else None,
                **summarize(test_rows, selected, test_days),
            })
        for limit in sorted(set(BUDGETS + [budget])):
            selected = test_rows["rank"].le(limit)
            curve.append({
                "model": name, "budget": limit, **summarize(test_rows, selected, test_days),
            })
        labelled = test_rows.loc[test_rows["fraud"].eq(1)]
        for scenario, subset in labelled.groupby("scenario"):
            scenarios.append({
                "model": name, "scenario": scenario, "fraud": len(subset),
                "detected": int(subset["rank"].le(budget).sum()),
                "recall": subset["rank"].le(budget).mean(),
            })

    results = pd.DataFrame(rows)
    results.to_csv(reports / "metrics.csv", index=False)
    if scenarios:
        pd.DataFrame(scenarios).to_csv(reports / "scenarios.csv", index=False)
    curve = pd.DataFrame(curve)
    curve.to_csv(reports / "budget_curve.csv", index=False)
    plot_budget_curve(curve, reports / "budget_curve.png", budget)
    print(results.round(4).to_string(index=False))


def score_day(day, budget, output, dataset="simulation"):
    database, model_path = (IEEE_DATABASE, IEEE_MODEL_PATH) if dataset == "ieee" else (DATABASE, MODEL_PATH)
    model = CatBoostClassifier().load_model(str(model_path))
    with duckdb.connect(str(database), read_only=True) as db:
        _, _, test_start, _ = split_dates(db, dataset)
        if day < test_start:
            raise ValueError(f"Choose a date on or after {test_start}, following the development period.")
        transactions = fetch_period(db, day, day + timedelta(days=1), labels=False)
    if transactions.empty:
        raise ValueError(f"No transactions for {day}")
    results = ranked(transactions, model.predict_proba(transactions[HISTORY])[:, 1])
    results["review"] = results["rank"].le(budget)
    output.parent.mkdir(parents=True, exist_ok=True)
    results[["tx_id", "tx_time", "amount", "score", "rank", "review"]].to_csv(output, index=False)
    print(f"{len(results):,} scored; {results['review'].sum():,} queued for review. {output}")


def main():
    cli = argparse.ArgumentParser(description="Transaction fraud backtest and daily batch scoring.")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ["experiment", "score-day"]:
        command = commands.add_parser(name)
        command.add_argument("--budget", type=int, default=100)
        command.add_argument("--dataset", choices=["simulation", "ieee"], default="simulation")
    commands.choices["score-day"].add_argument("--date", type=date.fromisoformat, required=True)
    commands.choices["score-day"].add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.budget < 1:
        cli.error("--budget must be positive")
    if args.command == "experiment":
        experiment(args.budget, args.dataset)
    else:
        try:
            score_day(args.date, args.budget, args.output, args.dataset)
        except (ValueError, FileNotFoundError) as error:
            cli.error(str(error))


if __name__ == "__main__":
    main()
