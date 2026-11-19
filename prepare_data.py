import hashlib
import argparse
import io
import re
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlopen

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARCHIVE = DATA / "handbook.tar.gz"
DATABASE = DATA / "transactions.duckdb"
IEEE_DATABASE = DATA / "ieee_cis.duckdb"
REVISION = "6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a"
URL = f"https://codeload.github.com/Fraud-Detection-Handbook/simulated-data-raw/tar.gz/{REVISION}"
SHA256 = "8b3b161a9d583858db63df4a1ede415b19c28201e3dfb29fc9ac2405b99cc596"
COLUMNS = {
    "TRANSACTION_ID": "tx_id",
    "TX_DATETIME": "tx_time",
    "CUSTOMER_ID": "customer_id",
    "TERMINAL_ID": "terminal_id",
    "TX_AMOUNT": "amount",
    "TX_FRAUD": "fraud",
    "TX_FRAUD_SCENARIO": "scenario",
}


def verify_archive(path):
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != SHA256:
        raise ValueError(f"Unexpected SHA-256 for {path}: {digest}")


def download():
    DATA.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        partial = ARCHIVE.with_suffix(".part")
        with urlopen(URL, timeout=60) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        verify_archive(partial)
        partial.replace(ARCHIVE)
    verify_archive(ARCHIVE)


def build_simulation():
    download()
    partial = DATABASE.with_suffix(".part.duckdb")
    partial.unlink(missing_ok=True)
    with duckdb.connect(str(partial)) as db, tarfile.open(ARCHIVE) as archive:
        days = sorted(
            member.name for member in archive.getmembers()
            if member.isfile() and re.fullmatch(r"\d{4}-\d{2}-\d{2}\.pkl", Path(member.name).name)
        )
        if len(days) != 183:
            raise ValueError(f"Expected 183 daily files, got {len(days)}")
        db.execute("SET threads = 4")
        for index, day in enumerate(days):
            # Load only pickles from the pinned, checksum-verified archive.
            with archive.extractfile(day) as source:
                batch = pd.read_pickle(io.BytesIO(source.read()))
            batch = batch[list(COLUMNS)].rename(columns=COLUMNS)
            if index == 0:
                db.execute("CREATE TABLE transactions AS SELECT * FROM batch")
            else:
                db.execute("INSERT INTO transactions SELECT * FROM batch")
        query = (ROOT / "sql" / "features.sql").read_text()
        db.execute(f"CREATE TABLE features AS {query}")
        print(db.execute("SELECT count(*), sum(fraud), min(tx_time), max(tx_time) FROM features").fetchone())
    partial.replace(DATABASE)


def build_ieee(source):
    DATA.mkdir(exist_ok=True)
    partial = IEEE_DATABASE.with_suffix(".part.duckdb")
    partial.unlink(missing_ok=True)
    with duckdb.connect(str(partial)) as db:
        db.execute("SET threads = 4")
        db.execute("""
            CREATE TABLE transactions AS
            WITH imported AS (
                SELECT TransactionID::BIGINT AS tx_id, TransactionDT::BIGINT AS elapsed_seconds,
                       TransactionAmt::DOUBLE AS amount, isFraud::TINYINT AS fraud,
                       card1, card2, card3, card5, addr1
                FROM read_csv(?, header = true, all_varchar = true)
            )
            SELECT tx_id, TIMESTAMP '1970-01-01' + to_seconds(elapsed_seconds) AS tx_time,
                   CASE WHEN card1 IS NOT NULL AND card2 IS NOT NULL AND addr1 IS NOT NULL
                        THEN dense_rank() OVER (ORDER BY card1, card2, card3, card5, addr1)
                        ELSE -1 - tx_id END AS customer_id,
                   NULL::BIGINT AS terminal_id, amount, fraud, NULL::TINYINT AS scenario
            FROM imported
        """, [str(source)])
        rows, unique_ids, invalid, positives, without_proxy = db.execute("""
            SELECT count(*), count(DISTINCT tx_id),
                   count_if(tx_id IS NULL OR tx_id < 0 OR tx_time IS NULL OR amount IS NULL
                            OR NOT isfinite(amount) OR amount < 0
                            OR fraud IS NULL OR fraud NOT IN (0, 1)),
                   count_if(fraud = 1), count_if(customer_id < 0)
            FROM transactions
        """).fetchone()
        if rows != unique_ids or invalid or positives in (0, rows):
            raise ValueError("Expected unique transaction IDs and valid labelled transactions.")
        query = (ROOT / "sql" / "features.sql").read_text()
        db.execute(f"CREATE TABLE features AS {query}")
        print(f"{rows:,} transactions; {positives:,} fraud; {without_proxy:,} without a grouping key.")
    partial.replace(IEEE_DATABASE)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Build a transaction database and calculate SQL features.")
    cli.add_argument("--ieee-csv", type=Path, help="Local labelled train_transaction.csv from IEEE-CIS.")
    args = cli.parse_args()
    if args.ieee_csv:
        build_ieee(args.ieee_csv)
    else:
        build_simulation()
