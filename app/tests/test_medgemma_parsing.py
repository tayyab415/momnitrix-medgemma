from momnitrix.model_runtime import (
    _coerce_medgemma_json,
    _coerce_medgemma_text,
    _extract_json_candidate,
    _sanitize_medgemma_output,
)


def test_coerce_medgemma_json_supports_low_mid_high():
    payload = {
        "risk_level": "LOW",
        "clinical_reasoning": "Readings are currently stable.",
        "recommended_actions": ["Continue routine prenatal care."],
    }
    result = _coerce_medgemma_json(payload)
    assert result is not None
    assert result.risk_level == "green"
    assert result.action_items


def test_coerce_medgemma_text_supports_sectioned_output():
    text = """
RISK LEVEL: MID

CLINICAL REASONING:
Fasting glucose is above pregnancy target and needs follow-up.

RECOMMENDED ACTIONS:
- Contact your OB team within 24 hours.
- Repeat fasting glucose as directed.
"""
    result = _coerce_medgemma_text(text)
    assert result is not None
    assert result.risk_level == "yellow"
    assert any("glucose" in x.lower() for x in result.reasons)
    assert len(result.action_items) >= 2


def test_sanitize_and_coerce_drop_duplicate_model_turn():
    raw = """
RISK LEVEL: HIGH

CLINICAL REASONING:
Severe fasting glucose elevation needs urgent review.

POTENTIAL COMPLICATIONS:
Risk of maternal-fetal compromise if persistent.

RECOMMENDED ACTIONS:
- Arrange urgent obstetric review today.
- Repeat fasting glucose now.

WARNING SIGNS:
- Reduced fetal movement.

model
RISK LEVEL: LOW
CLINICAL REASONING:
This duplicate tail should be ignored.
"""
    cleaned = _sanitize_medgemma_output(raw)
    assert "model\nRISK LEVEL" not in cleaned

    result = _coerce_medgemma_text(cleaned)
    assert result is not None
    assert result.risk_level == "red"
    assert not any("model\nRISK LEVEL" in x for x in result.reasons)
    assert not any(x.strip().lower() == "warning signs:" for x in result.action_items)


def test_sanitize_and_coerce_drop_duplicate_inline_model_turn():
    raw = """
RISK LEVEL: MID

CLINICAL REASONING:
Fasting glucose is above target and needs follow-up.

RECOMMENDED ACTIONS:
- Contact OB team within 24 hours.
- Repeat fasting glucose.

WARNING SIGNS:
- Severe headache with visual changes.

This line ends with leaked token.model
RISK LEVEL: LOW
CLINICAL REASONING:
This duplicate tail should be ignored.
"""
    cleaned = _sanitize_medgemma_output(raw)
    assert "RISK LEVEL: LOW" not in cleaned
    assert not cleaned.lower().endswith("model")

    result = _coerce_medgemma_text(cleaned)
    assert result is not None
    assert result.risk_level == "yellow"


def test_coerce_medgemma_text_strips_unused_tokens_and_malformed_lines():
    text = """
RISK LEVEL: MID

CLINICAL REASONING:
Fasting glucose is above pregnancy target and requires follow-up.
The patient's blood pressure of 12.0/8.0 mmHg is stable.

RECOMMENDED ACTIONS:
- Contact OB team within 24 hours.
- Urgent warning sign: reduced fetal movement.<unused95>
"""
    result = _coerce_medgemma_text(_sanitize_medgemma_output(text))
    assert result is not None
    assert result.risk_level == "yellow"
    assert any("glucose" in x.lower() for x in result.reasons)
    assert not any("12.0/8.0 mmhg" in x.lower() for x in result.reasons)
    assert not any("<unused" in x.lower() for x in result.action_items)


def test_extract_json_candidate_returns_none_for_truncated_json():
    truncated = '{"risk_level":"HIGH","clinical_reasoning":"severe","action_items":["urgent"]'
    assert _extract_json_candidate(truncated) is None
