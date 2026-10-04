import pytest

from playparse.ffscore.schema import Credit, PlayLabel, SchemaError
from playparse.prompt import build_messages


def _label():
    return PlayLabel(
        False,
        (
            Credit("A.Brown", "rec_yds", 14),
            Credit("J.Hurts", "pass_yds", 14),
            Credit("A.Brown", "rec", 1),
        ),
    )


def test_roundtrip_is_canonical():
    text = _label().to_json()
    assert text.startswith('{"nullified":false,"credits":[{"player":"J.Hurts","stat":"pass_yds"')
    assert PlayLabel.from_json(text) == _label().canonical()


def test_matches_is_order_insensitive():
    a = _label()
    b = PlayLabel(False, tuple(reversed(a.credits)))
    assert a.matches(b)
    assert not a.matches(PlayLabel(False, a.credits[:2]))
    assert not a.matches(PlayLabel(True, ()))


def test_duplicate_credits_count():
    a = PlayLabel(False, (Credit("X", "rec", 1),))
    b = PlayLabel(False, (Credit("X", "rec", 1), Credit("X", "rec", 1)))
    assert not a.matches(b)


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        "[]",
        '{"nullified": false}',
        '{"nullified": "no", "credits": []}',
        '{"nullified": false, "credits": [{"player": "X", "stat": "sacks", "value": 1}]}',
        '{"nullified": false, "credits": [{"player": "X", "stat": "rec", "value": 1.5}]}',
        '{"nullified": false, "credits": [{"player": "X", "stat": "rec", "value": true}]}',
        '{"nullified": true, "credits": [{"player": "X", "stat": "rec", "value": 1}]}',
        '{"nullified": false, "credits": [], "extra": 1}',
    ],
)
def test_from_json_rejects(bad):
    with pytest.raises(SchemaError):
        PlayLabel.from_json(bad)


def test_prompt_contains_play():
    msgs = build_messages("PHI", "(3:12) 1-J.Hurts pass short right to 11-A.Brown for 14 yards")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "posteam: PHI" in msgs[1]["content"] and "A.Brown" in msgs[1]["content"]


def test_prompt_matches_labeling_rules():
    # Labels follow the final ruling on reversals (official stats), so the prompt
    # must not tell the model that reversed plays are nullified.
    from playparse.prompt import SYSTEM_PROMPT

    assert "REVERSED" in SYSTEM_PROMPT and "keep nullified false" in SYSTEM_PROMPT
    assert "or reversed" not in SYSTEM_PROMPT
    assert "value would be 0" in SYSTEM_PROMPT
