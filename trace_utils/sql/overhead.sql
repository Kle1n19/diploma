SELECT
  ROUND(AVG(CASE WHEN name = 'PjitFunction(run_simulation)' THEN dur END) / 1e6, 3) AS pjit_avg_ms,
  ROUND(AVG(CASE WHEN name = 'jit_run_simulation:XLA GPU module' THEN dur END) / 1e6, 3) AS xla_avg_ms,
  ROUND(
    (AVG(CASE WHEN name = 'PjitFunction(run_simulation)' THEN dur END) -
     AVG(CASE WHEN name = 'jit_run_simulation:XLA GPU module' THEN dur END))
    / AVG(CASE WHEN name = 'PjitFunction(run_simulation)' THEN dur END) * 100,
  1) AS dispatch_overhead_pct
FROM slice
WHERE name IN ('PjitFunction(run_simulation)', 'jit_run_simulation:XLA GPU module');
