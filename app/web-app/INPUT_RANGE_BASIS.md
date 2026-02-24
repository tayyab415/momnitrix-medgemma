# Input Range Basis (Dataset + Web Sources)

Date: 2026-02-22

This document explains how QA console input bounds were chosen.

## 1) Your provided dataset

Source file:
- `/Users/tayyabkhan/Downloads/medgemma/Maternal Health Risk Data Set.csv`

Detected columns:
- `Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`, `RiskLevel`

Observed stats (after mapping `BodyTemp` from degF to degC):
- Age: min 10, p25 19, p50 26, p75 39, max 70
- SystolicBP: min 70, p25 100, p50 120, p75 120, max 160
- DiastolicBP: min 49, p25 65, p50 80, p75 90, max 100
- BS: min 6.0, p25 6.9, p50 7.5, p75 8.0, max 19.0
- BodyTemp (degF): min 98.0, max 103.0
- HeartRate: min 7, p25 70, p50 76, p75 80, max 90

Notes:
- The dataset includes clear outliers/noise (for example `Age=10`, `Age=70`, `HeartRate=7`), so QA bounds intentionally avoid extreme tails.
- The dataset does not include `gestational_weeks`, `bmi_group`, `spo2`, or `hrv`.

## 2) Web-backed references for missing fields and clinical context

1. NIH NICHD pregnancy timing terms (`full term`, `late term`, `postterm`)  
   https://www.nichd.nih.gov/ncmhep/initiatives/know-your-terms/moms

2. CDC adult BMI category cutoffs  
   https://www.cdc.gov/bmi/adult-calculator/bmi-categories.html

3. Pregnancy vital sign reference ranges (systematic review/meta-analysis)  
   https://pubmed.ncbi.nlm.nih.gov/32028507/

4. Diabetes.org pregnancy glucose targets (ADA-linked patient guidance)  
   https://diabetes.org/living-with-diabetes/pregnancy/diabetes-during-pregnancy

5. MedlinePlus fever threshold (100.4 degF / 38 degC)  
   https://medlineplus.gov/ency/article/003090.htm

6. HRV in pregnancy review noting no universally established clinical cutoff standards  
   https://pubmed.ncbi.nlm.nih.gov/32202437/

## 3) Final QA bounds used in UI + audit contract

These are for **manual testing realism**, not diagnostic decision thresholds.

| Field | QA Bound | Basis |
|---|---|---|
| `age` | 13 to 50 years | Dataset spread + avoid implausible tails for pregnancy demos |
| `gestational_weeks` | 4 to 42 | NICHD pregnancy term definitions and clinical plausibility |
| `bmi_group` | underweight/normal/overweight/obese | CDC BMI categories |
| `systolic_bp` | 80 to 180 mmHg | Dataset core range + obstetric high-risk simulation headroom |
| `diastolic_bp` | 45 to 120 mmHg | Dataset core range + obstetric high-risk simulation headroom |
| `fasting_glucose_mmol_l` | 3.0 to 20.0 | ADA pregnancy target context + dataset high values |
| `temp_f` (input) | 95.0 to 104.0 degF | Medline fever threshold context + dataset values |
| `temp_c` (payload/audit) | 35.0 to 40.0 degC | Converted from degF and constrained for realistic clinical testing |
| `hr` | 45 to 140 bpm | Dataset central tendency + pregnancy vital-sign references |
| `spo2` | 88 to 100% | Pregnancy vital-sign references + warning-level simulation space |
| `hrv` | 10 to 140 ms | No universal pregnancy thresholds; practical wearable QA envelope |

## 4) Important implementation notes

- In the QA frontend, users enter temperature in degF; payload sends `temp_c`.
- Audit tooling maps dataset aliases automatically:
  - `SystolicBP -> systolic_bp`
  - `DiastolicBP -> diastolic_bp`
  - `BS -> fasting_glucose_mmol_l`
  - `BodyTemp -> temp_f -> temp_c`
  - `HeartRate -> hr`
  - BOM-safe handling for the `Age` header
