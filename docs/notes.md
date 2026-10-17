# Notes on SQL and Pandas

## DuckDB

- con = duckdb.connect(): open a database with connection con
- con.sql("..."): send a SQL command through the connection con
- .df(): return a pandas dataframe
- .fetchone(): return the first row as a tuple

## SQL
- CREATE VIEW name AS SELECT ... — give a name to a query (no data copied)
- DESCRIBE table — one row per column: name and type
- COUNT(*) — number of total rows; COUNT(col) — non-null values in col
- GROUP BY col — group rows with the same value; aggregates are computed per group

## pandas
- df["col"].value_counts() — count occurrences of each value (sorted, descending)