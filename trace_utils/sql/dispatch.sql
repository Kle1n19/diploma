SELECT
  name,
  COUNT(*) AS n,
  ROUND(AVG(dur) / 1e6, 3) AS avg_ms
FROM slice
WHERE name IN (
  'PjitFunction(run_simulation)',
  'PjRtCApiLoadedExecutable::Execute',
  'PJRT_LoadedExecutable_Execute',
  'CommonPjRtLoadedExecutable::Execute (jit_run_simulation)',
  'CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice',
  '[0] PjRtStreamExecutorRawLoadedExecutable::Execute',
  '[0] GpuExecutable::ExecuteThunks',
  'jit_run_simulation:XLA GPU module'
)
GROUP BY name
ORDER BY avg_ms DESC;