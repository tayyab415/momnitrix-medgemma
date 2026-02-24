import { VitalsPayload } from '../frontend/utils/randomizer';

export enum RiskLevel {
  GREEN = 'LOW',
  YELLOW = 'MID',
  RED = 'HIGH',
  CRITICAL = 'CRITICAL_URGENT'
}

export class LocalTriageLogic {
  static evaluateImmediateRisk(vitals: VitalsPayload): { level: RiskLevel, flags: string[] } {
    const flags: string[] = [];
    let maxRisk: RiskLevel = RiskLevel.GREEN;

    const updateRisk = (newRisk: RiskLevel) => {
      const riskRank = {
        [RiskLevel.GREEN]: 0,
        [RiskLevel.YELLOW]: 1,
        [RiskLevel.RED]: 2,
        [RiskLevel.CRITICAL]: 3,
      };
      if (riskRank[newRisk] > riskRank[maxRisk]) {
        maxRisk = newRisk;
      }
    };

    // Systolic BP Checks
    if (vitals.systolic_bp >= 160) {
      flags.push(`Severe Hypertension: Systolic BP is ${vitals.systolic_bp} (>=160)`);
      updateRisk(RiskLevel.CRITICAL);
    } else if (vitals.systolic_bp >= 140) {
      flags.push(`Hypertension: Systolic BP is ${vitals.systolic_bp} (>=140)`);
      updateRisk(RiskLevel.RED);
    }

    // Diastolic BP Checks
    if (vitals.diastolic_bp >= 110) {
      flags.push(`Severe Hypertension: Diastolic BP is ${vitals.diastolic_bp} (>=110)`);
      updateRisk(RiskLevel.CRITICAL);
    } else if (vitals.diastolic_bp >= 90) {
      flags.push(`Hypertension: Diastolic BP is ${vitals.diastolic_bp} (>=90)`);
      updateRisk(RiskLevel.YELLOW);
    }

    // SpO2 Checks
    if (vitals.spo2 < 90) {
      flags.push(`Hypoxemia: SpO2 is ${vitals.spo2}% (<90%)`);
      updateRisk(RiskLevel.CRITICAL);
    } else if (vitals.spo2 < 94) {
      flags.push(`Mild Hypoxemia: SpO2 is ${vitals.spo2}% (<94%)`);
      updateRisk(RiskLevel.YELLOW);
    }

    // Temperature Checks
    if (vitals.temperature >= 39.0) {
      flags.push(`High Fever: Temp is ${vitals.temperature}°C (>=39.0)`);
      updateRisk(RiskLevel.RED);
    } else if (vitals.temperature >= 38.0) {
      flags.push(`Fever: Temp is ${vitals.temperature}°C (>=38.0)`);
      updateRisk(RiskLevel.YELLOW);
    }

    return {
      level: maxRisk,
      flags: flags
    };
  }
}
