-- The lower bound is inclusive. Transactions at the same timestamp share the
-- same history because their order of arrival is unavailable.
WITH history AS (
    SELECT
        tx_id,
        tx_time,
        customer_id,
        terminal_id,
        amount,
        date_part('hour', tx_time) AS hour,
        date_part('isodow', tx_time) AS weekday,
        fraud,
        scenario,
        count(*) OVER customer_hour AS count_1h,
        count(*) OVER customer_day AS count_24h,
        count(*) OVER customer_week AS count_7d,
        avg(amount) OVER customer_week AS mean_amount_7d,
        max(tx_time) OVER customer_past AS previous_time
    FROM transactions
    WINDOW
        customer_hour AS (
            PARTITION BY customer_id ORDER BY tx_time
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ),
        customer_day AS (
            PARTITION BY customer_id ORDER BY tx_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ),
        customer_week AS (
            PARTITION BY customer_id ORDER BY tx_time
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ),
        customer_past AS (
            PARTITION BY customer_id ORDER BY tx_time
            RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        )
)
SELECT
    * EXCLUDE (previous_time),
    amount / nullif(mean_amount_7d, 0) AS amount_to_mean_7d,
    date_diff('second', previous_time, tx_time) AS seconds_since_previous
FROM history
