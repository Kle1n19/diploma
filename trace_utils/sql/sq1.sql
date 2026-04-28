SELECT
  ROW_NUMBER() OVER (ORDER BY ts) AS run,
  ROUND(dur / 1e6, 4) AS dur_ms
FROM slice
WHERE name = 'jit_run_simulation:XLA GPU module'
ORDER BY ts;
