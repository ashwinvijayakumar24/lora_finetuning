"""LoRA hyperparameters and target-module selection.

Target selection follows PEFT's rule for a list of names: a module matches when
its dotted path equals a target or ends with `"." + target`. So `"q_proj"`
matches `model.layers.3.self_attn.q_proj`, and a full path matches only itself.
That rule is shared by injection and by loading PEFT-saved adapters, whose
`adapter_config.json` may list either short names or full paths.

`"all-linear"` means the seven projection layers of a Llama block. `lm_head` is
never included: it is tied to the input embeddings in Llama 3.2 1B, and PEFT's
own `"all-linear"` excludes the output layer too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")
ALL_LINEAR = ATTN_PROJ + MLP_PROJ
EXCLUDED = ("lm_head",)


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: list[str] | str = "all-linear"

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError(f"r must be positive, got {self.r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if isinstance(self.target_modules, str) and self.target_modules != "all-linear":
            raise ValueError(
                f"target_modules must be a list of names or 'all-linear', got {self.target_modules!r}"
            )

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    def resolved_targets(self) -> tuple[str, ...]:
        if self.target_modules == "all-linear":
            return ALL_LINEAR
        return tuple(self.target_modules)

    def matches(self, module_name: str) -> bool:
        """True if the module at dotted path `module_name` should get an adapter."""
        leaf = module_name.rsplit(".", 1)[-1]
        if self.target_modules == "all-linear" and leaf in EXCLUDED:
            return False
        return any(
            module_name == t or module_name.endswith("." + t) for t in self.resolved_targets()
        )

    def to_peft_dict(self, base_model_name_or_path: str | None = None) -> dict[str, Any]:
        """The `adapter_config.json` contents PEFT and vLLM expect."""
        return {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": base_model_name_or_path,
            "r": self.r,
            "lora_alpha": int(self.alpha) if float(self.alpha).is_integer() else self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": sorted(self.resolved_targets()),
            "bias": "none",
            "fan_in_fan_out": False,
            "init_lora_weights": True,
            "inference_mode": True,
            "use_rslora": False,
            "use_dora": False,
            "modules_to_save": None,
            "layers_to_transform": None,
            "layers_pattern": None,
            "rank_pattern": {},
            "alpha_pattern": {},
        }

    @classmethod
    def from_peft_dict(cls, d: dict[str, Any]) -> LoRAConfig:
        """Parse a PEFT `adapter_config.json`, refusing features this layer lacks.

        Silently ignoring e.g. `use_rslora` would load the right tensors with the
        wrong scale, which is the worst kind of bug: plausible outputs, wrong numbers.
        """
        if d.get("peft_type", "LORA") != "LORA":
            raise ValueError(f"not a LoRA adapter: peft_type={d.get('peft_type')!r}")
        unsupported = {
            "use_rslora": bool(d.get("use_rslora")),
            "use_dora": bool(d.get("use_dora")),
            "bias != none": d.get("bias", "none") != "none",
            "fan_in_fan_out": bool(d.get("fan_in_fan_out")),
            "rank_pattern": bool(d.get("rank_pattern")),
            "alpha_pattern": bool(d.get("alpha_pattern")),
            "modules_to_save": bool(d.get("modules_to_save")),
            "layers_to_transform": d.get("layers_to_transform") is not None,
        }
        bad = [k for k, v in unsupported.items() if v]
        if bad:
            raise NotImplementedError(f"adapter uses unsupported PEFT features: {bad}")
        targets = d["target_modules"]
        if isinstance(targets, str):
            if targets != "all-linear":
                raise NotImplementedError(f"regex target_modules not supported: {targets!r}")
        else:
            targets = list(targets)
        return cls(
            r=int(d["r"]),
            alpha=float(d["lora_alpha"]),
            dropout=float(d.get("lora_dropout", 0.0)),
            target_modules=targets,
        )
