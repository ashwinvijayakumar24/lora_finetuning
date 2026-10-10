import numpy as np
import torch

from playparse.eval import forgetting as fg
from tests._lora_util import tiny_llama


class _Tok:
    """Minimal tokenizer: one id per character, letters A-D map to fixed ids."""
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, msgs, **kw):
        return msgs[0]["content"]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + (ord(c) % 60) for c in text]}

    def encode(self, x, add_special_tokens=False):
        return [1 + (ord(x) % 60)]


def test_format_question_lists_options_and_ends_with_answer():
    s = fg.format_question({"question": "Q?", "choices": ["a", "b", "c", "d"], "answer": 0})
    assert s.endswith("Answer:") and "C. c" in s


def test_score_returns_a_letter_per_question_and_padding_is_harmless():
    torch.manual_seed(0)
    model = tiny_llama(seed=0)
    qs = [{"question": "x" * k, "choices": ["a", "b", "c", "d"], "answer": 0} for k in (1, 5, 9)]
    batched = fg.score(model, _Tok(), qs, batch_size=3)
    single = np.concatenate([fg.score(model, _Tok(), [q], batch_size=1) for q in qs])
    assert batched.shape == (3,) and set(batched) <= {0, 1, 2, 3}
    assert (batched == single).all()


def test_bootstrap_ci_brackets_the_mean():
    c = np.array([1.0] * 60 + [0.0] * 40)
    lo, hi = fg.bootstrap_ci(c)
    assert lo < 0.6 < hi
