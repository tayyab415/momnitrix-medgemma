import ranges from '../../configs/physiological_ranges.json';

export interface VitalsPayload {
  age: number;
  gestational_weeks: number | null; // e.g. 4-42
  systolic_bp: number;
  diastolic_bp: number;
  heart_rate: number;
  temperature: number;
  spo2: number;
  fasting_glucose: number;
}

const getRandomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

const getRandomFloat = (min: number, max: number, decimals: number = 1) => {
  const num = Math.random() * (max - min) + min;
  return parseFloat(num.toFixed(decimals));
};

export const randomizeVitals = (isPregnant: boolean = true): VitalsPayload => {
  return {
    age: getRandomInt(ranges.age.min, ranges.age.max),
    gestational_weeks: isPregnant ? getRandomInt(4, 42) : null,
    systolic_bp: getRandomInt(ranges.systolic_bp.min, ranges.systolic_bp.max),
    diastolic_bp: getRandomInt(ranges.diastolic_bp.min, ranges.diastolic_bp.max),
    heart_rate: getRandomInt(ranges.heart_rate_bpm.min, ranges.heart_rate_bpm.max),
    temperature: getRandomFloat(ranges.temperature_c.min, ranges.temperature_c.max, 1),
    spo2: getRandomInt(ranges.spo2_percent.min, ranges.spo2_percent.max),
    fasting_glucose: getRandomFloat(ranges.fasting_glucose_mmol_L.min, ranges.fasting_glucose_mmol_L.max, 1),
  };
};
