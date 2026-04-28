SELECT
  name,
  COUNT(*) AS calls,
  ROUND(AVG(dur) / 1e6, 4) AS avg_ms,
  ROUND(SUM(dur) / 1e6, 4) AS total_ms
FROM slice
WHERE name IN ('MemcpyD2D', 'memcpy32_post', 'copy.18', 'BUFFER_FLUSH')
GROUP BY name
ORDER BY total_ms DESC;
