"""XLA flag space definitions for each backend."""


def xla_flag_space(backend: str) -> list[str]:
    """
    Return a list of XLA_FLAGS strings worth benchmarking on the given backend.

    Each entry is a complete XLA_FLAGS value (space-separated flags).
    The empty string represents the baseline (no extra flags).
    Combinations are chosen to be progressive — each adds one meaningful axis
    so the agent can learn which flags actually help on this workload.
    """
    if backend == "cpu":
        return [
            "",
            "--xla_cpu_enable_fast_math",
            "--xla_cpu_prefer_vector_width=256",
            "--xla_backend_optimization_level=4",
            "--xla_cpu_enable_fast_math --xla_cpu_prefer_vector_width=256",
            "--xla_cpu_enable_fast_math --xla_backend_optimization_level=4",
            "--xla_cpu_enable_fast_math --xla_cpu_prefer_vector_width=256 --xla_backend_optimization_level=4",
        ]

    if backend == "gpu":
        return [
            "",
            "--xla_gpu_graph_level=1",
            "--xla_gpu_graph_level=2",
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_backend_optimization_level=4",
            "--xla_gpu_graph_level=2 --xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_gpu_enable_triton_gemm=true",
            "--xla_gpu_graph_level=2 --xla_gpu_enable_latency_hiding_scheduler=true --xla_backend_optimization_level=4",
            "--xla_gpu_graph_level=2 --xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_triton_gemm=true --xla_backend_optimization_level=4",
        ]

    if backend == "tpu":
        return [
            "",
            "--xla_backend_optimization_level=4",
        ]

    return [""]
