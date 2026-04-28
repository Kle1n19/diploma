import optuna
from optuna.samplers import TPESampler

from .base import TuningAgent

optuna.logging.set_verbosity(optuna.logging.WARNING)


class AutoSamplerAgent(TuningAgent):
    def __init__(self, n_actions: int, configs: list[dict], seed: int = 0, n_startup_trials: int = 10):
        self.n_actions = n_actions
        self._configs  = configs
        self._param_keys: list[str] = list(configs[0].keys())
        self._param_values: dict[str, list] = {k: [] for k in self._param_keys}
        self._param_index: dict[str, dict] = {k: {} for k in self._param_keys}

        for cfg in configs:
            for k in self._param_keys:
                v = cfg[k]
                bucket = self._param_values[k]
                if not any(existing is v or existing == v for existing in bucket):
                    pos = len(bucket)
                    bucket.append(v)
                    self._param_index[k][pos] = v

        self._study = optuna.create_study(direction = "maximize", sampler = TPESampler(seed=seed, n_startup_trials=n_startup_trials))

        self._tuple_to_idx: dict[tuple, int] = {}
        for i, cfg in enumerate(configs):
            key = self._cfg_to_tuple(cfg)
            self._tuple_to_idx[key] = i

        self._current_trial: optuna.trial.Trial | None = None

    def _cfg_to_tuple(self, cfg: dict) -> tuple:
        parts = []
        for k in self._param_keys:
            v = cfg[k]
            bucket = self._param_values[k]
            for pos, existing in enumerate(bucket):
                if existing is v or existing == v:
                    parts.append(pos)
                    break
        return tuple(parts)

    def _trial_to_cfg_idx(self, trial: optuna.trial.Trial) -> int:
        parts = []
        for k in self._param_keys:
            n_vals = len(self._param_values[k])
            pos = trial.suggest_int(k, 0, n_vals - 1)
            parts.append(pos)
        key = tuple(parts)
        return self._tuple_to_idx.get(key, 0)

    def select_action(self) -> int:
        self._current_trial = self._study.ask()
        return self._trial_to_cfg_idx(self._current_trial)

    def update(self, action: int, reward: float) -> None:
        self._study.tell(self._current_trial, reward)

    def best_action(self) -> int:
        if not self._study.trials:
            return 0
        best_params = self._study.best_trial.params
        parts = tuple(best_params[k] for k in self._param_keys)
        return self._tuple_to_idx.get(parts, 0)

    def reset(self) -> None:
        sampler = self._study.sampler
        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._current_trial = None

    def best_reward(self) -> float:
        if not self._study.trials:
            return float("-inf")
        return self._study.best_value

    def n_trials(self) -> int:
        return len(self._study.trials)
