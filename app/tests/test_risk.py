from momnitrix.risk import compute_policy_floor
from momnitrix.schemas import Inputs, PatientContext, TriageStreamRequest, Vitals


def test_policy_floor_red_for_severe_bp():
    request = TriageStreamRequest(
        vitals=Vitals(systolic_bp=165, diastolic_bp=112),
        inputs=Inputs(),
    )

    floor, reasons = compute_policy_floor(request, specialist_outputs={})

    assert floor == "red"
    assert any("blood pressure" in reason.lower() for reason in reasons)


def test_policy_floor_yellow_for_wound_infection_signal():
    request = TriageStreamRequest(vitals=Vitals(), inputs=Inputs())
    floor, reasons = compute_policy_floor(
        request,
        specialist_outputs={"wound_scores": {"infection_risk": 0.81, "urgency": 0.2}},
    )

    assert floor == "yellow"
    assert reasons


def test_policy_floor_red_for_severe_fasting_glucose():
    request = TriageStreamRequest(
        patient_context=PatientContext(gestational_weeks=31),
        vitals=Vitals(fasting_glucose=15.0),
        inputs=Inputs(),
    )
    floor, reasons = compute_policy_floor(request, specialist_outputs={})

    assert floor == "red"
    assert any("hyperglycemia" in reason.lower() or "glucose" in reason.lower() for reason in reasons)


def test_policy_floor_parses_glucose_from_free_text():
    request = TriageStreamRequest(
        patient_context=PatientContext(gestational_weeks=31),
        vitals=Vitals(),
        inputs=Inputs(free_text="Fasting plasma glucose: 13.0 mmol/L"),
    )
    floor, reasons = compute_policy_floor(request, specialist_outputs={})

    assert floor == "red"
    assert any("glucose" in reason.lower() for reason in reasons)
