import jax
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc_tuner.tuner import Tuner
from mc_tuner.search_space import JAX_INJECT_DEFAULTS
from experiments.domains.american_option import make_fns as option_fns


def deserialize_params(p: dict) -> dict:
    policy_map = {
        "nothing_saveable": jax.checkpoint_policies.nothing_saveable,
        "everything_saveable": jax.checkpoint_policies.everything_saveable,
        "dots_with_no_batch_dims_saveable": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    }
    out = {}
    for k, v in p.items():
        if k == "checkpoint_policy":
            if v is None or v == "None":
                out[k] = None
            else:
                out[k] = next((fn for name, fn in policy_map.items() if name in str(v)), None)
        elif k == "jit_donate_argnums":
            if v in ("()", "[]", None, "None"):
                out[k] = ()
            elif isinstance(v, str):
                out[k] = tuple(int(x) for x in v.strip("()").split(",") if x.strip())
            else:
                out[k] = v
        else:
            out[k] = v
    return out


results_path = Path("experiments/convergence_speed/results/results.json")
with open(results_path) as f:
    data = json.load(f)

best_result = max(data["results"], key=lambda r: r.get("best_speedup") or 0)
best_params = deserialize_params(best_result["best_params"])
print(f"Agent: {best_result['name']}  (speedup {best_result['best_speedup']:.3f}x)")
print(f"Params: {best_params}\n")

step_fn, init_carry_fn = option_fns(n_steps=252)

tuner = Tuner(
    step_fn = step_fn,
    init_carry_fn = init_carry_fn,
    master_key = jax.random.PRNGKey(0),
    batch_size = 512,
    n_steps = 252,
)

tuner.trace(JAX_INJECT_DEFAULTS, output_dir="./traces/baseline")
tuner.trace(best_params, output_dir="./traces/tuned")
