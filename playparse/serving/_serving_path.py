"""Locate the sibling ``llm_serving_layer`` checkout and make ``import serving`` work.

P5b extends the owner's serving layer from the outside, exactly as P5a extends
the engine: nothing is pip-installed and nothing in the sibling repo is edited.
Its two importable packages, ``serving`` and ``bench``, are registered by file
location so that ``import serving.scheduler.scheduler`` resolves to the owner's
checkout (see :func:`ensure_serving_importable` for why not ``sys.path``).

Resolution order mirrors ``_engine_path``:

1. ``PLAYPARSE_SERVING_DIR`` if set (Slurm jobs, unusual layouts).
2. The first ``llm_serving_layer`` directory found next to this repository or
   next to any of its parents (git worktrees live three levels down).

ONE ENGINE, NOT TWO
-------------------
The serving layer vendors its own engine copy at ``vendor/llm_inference_engine``
and imports ``engine.attention_backend``. PlayParse's P5a code imports the
sibling ``llm_inference_engine``. If both ended up imported under the name
``engine`` the process would hold two ``components_gpu`` modules, and the
``linear()`` replacement that unmerged LoRA depends on would patch one while the
model ran the other: the adapter would silently not apply.

So the engine is resolved once, through ``ensure_engine_importable``. Whichever
copy is imported first is used by everything, and :func:`check_engine` asserts
it has the batched seam (``engine.attention_backend`` and
``LlamaModelGPU.forward_varlen``) that the serving layer needs. The vendored copy
and the sibling checkout were compared file by file when P5b was written and
are identical apart from the sibling's ``engine/bench`` directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from playparse.serving._engine_path import ensure_engine_importable, register_package

SERVING_DIR_ENV = "PLAYPARSE_SERVING_DIR"
_SERVING_REPO_NAME = "llm_serving_layer"


def serving_dir() -> Path:
    """Return the serving-layer repository root (the directory containing ``serving/``)."""
    override = os.environ.get(SERVING_DIR_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not (path / "serving" / "__init__.py").is_file():
            raise FileNotFoundError(
                f"{SERVING_DIR_ENV}={override!r} does not contain serving/__init__.py"
            )
        return path
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _SERVING_REPO_NAME
        if (candidate / "serving" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find a sibling {_SERVING_REPO_NAME}/ checkout above {here}. "
        f"Set {SERVING_DIR_ENV} to the serving layer repository root."
    )


def check_engine() -> Path:
    """Assert the imported ``engine`` has the batched serving seam. Returns its root."""
    import engine  # noqa: F401  (resolved by ensure_engine_importable)
    from engine.model_gpu import LlamaModelGPU

    try:
        import engine.attention_backend  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the checkout
        raise ImportError(
            f"the imported engine ({sys.modules['engine'].__file__}) has no "
            "engine.attention_backend; the serving layer needs the engine version it vendors"
        ) from exc
    if not hasattr(LlamaModelGPU, "forward_varlen"):  # pragma: no cover
        raise ImportError("the imported engine's LlamaModelGPU has no forward_varlen")
    return Path(sys.modules["engine"].__file__).resolve().parent.parent


def ensure_serving_importable() -> Path:
    """Make ``import serving`` and ``import bench`` resolve to the serving layer. Idempotent.

    Only those two packages are registered. Putting the serving layer's root on
    ``sys.path`` would also expose its ``tests`` and ``scripts`` packages, which
    are regular packages and would therefore shadow this repository's
    ``tests``/``scripts`` directories (namespace packages) for any later import
    (docs/issues/p5b-serving-root-shadows-tests.md).
    """
    ensure_engine_importable()
    check_engine()
    if "serving" in sys.modules:
        mod_file = getattr(sys.modules["serving"], "__file__", None)
        if mod_file:
            root = Path(mod_file).resolve().parent.parent
            register_package("bench", root)
            return root
    root = serving_dir()
    register_package("serving", root)
    register_package("bench", root)
    return root
