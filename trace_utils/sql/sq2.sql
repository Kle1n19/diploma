SELECT
  COUNT(*) AS runs,
  ROUND(AVG(dur) / 1e6, 4) AS avg_ms,
  ROUND(MAX(dur) / 1e6, 4) AS max_ms,
  ROUND(MIN(dur) / 1e6, 4) AS min_ms,
  ROUND((AVG(dur * dur) - AVG(dur) * AVG(dur)) / 1e12, 6) AS variance_ms2
FROM slice
WHERE name = 'jit_run_simulation:XLA GPU module';
