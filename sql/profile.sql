-- Binning uses defined ranges and only the training period.
WITH train AS (
    SELECT
        fraud,
        CASE WHEN amount < 25 THEN '01. <25'
             WHEN amount < 100 THEN '02. [25, 100)'
             WHEN amount < 220 THEN '03. [100, 220)'
             ELSE '04. >=220' END AS amount_group,
        CASE WHEN hour < 6 THEN '01. 00–05'
             WHEN hour < 12 THEN '02. 06–11'
             WHEN hour < 18 THEN '03. 12–17'
             ELSE '04. 18–23' END AS hour_group,
        CASE WHEN count_24h = 0 THEN '01. 0'
             WHEN count_24h < 4 THEN '02. 1–3'
             ELSE '03. 4+' END AS frequency_group
    FROM features
    WHERE tx_time >= $start AND tx_time < $end
)
SELECT dimension, segment, count(*) AS transactions,
       sum(fraud) AS fraud, avg(fraud) AS fraud_rate
FROM train
UNPIVOT (segment FOR dimension IN (amount_group, hour_group, frequency_group))
GROUP BY dimension, segment
ORDER BY dimension, segment
