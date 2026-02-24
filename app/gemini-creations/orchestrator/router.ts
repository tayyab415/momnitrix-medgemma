export enum Modality {
  TEXT = 'TEXT',
  SPEECH = 'SPEECH',
  IMAGE_WOUND = 'IMAGE_WOUND',
  IMAGE_DERM = 'IMAGE_DERM',
  STRUCTURED_VITALS = 'STRUCTURED_VITALS'
}

export enum TargetModel {
  MEDGEMMA = 'MEDGEMMA',
  MEDASR = 'MEDASR',
  MEDSIGLIP = 'MEDSIGLIP',
  DERM_FOUNDATION = 'DERM_FOUNDATION'
}

export const RouteMap: Record<Modality, TargetModel> = {
  [Modality.TEXT]: TargetModel.MEDGEMMA,
  [Modality.SPEECH]: TargetModel.MEDASR,
  [Modality.IMAGE_WOUND]: TargetModel.MEDSIGLIP,
  [Modality.IMAGE_DERM]: TargetModel.DERM_FOUNDATION,
  [Modality.STRUCTURED_VITALS]: TargetModel.MEDGEMMA
};

export class OrchestratorRouter {
  
  /**
   * Determine the target model based on the provided input modality
   */
  static getRoute(modality: Modality): TargetModel {
    const route = RouteMap[modality];
    if (!route) {
      throw new Error(`Unsupported modality: ${modality}`);
    }
    return route;
  }

  /**
   * Given a complex payload containing multiple modalities (e.g., from the frontend),
   * returns the list of models that need to be invoked.
   */
  static determineRequiredModels(payload: {
    text?: string;
    audioBase64?: string;
    woundImageBase64?: string;
    skinImageBase64?: string;
    vitals?: any;
  }): TargetModel[] {
    const targets = new Set<TargetModel>();

    if (payload.text) targets.add(this.getRoute(Modality.TEXT));
    if (payload.audioBase64) targets.add(this.getRoute(Modality.SPEECH));
    if (payload.woundImageBase64) targets.add(this.getRoute(Modality.IMAGE_WOUND));
    if (payload.skinImageBase64) targets.add(this.getRoute(Modality.IMAGE_DERM));
    if (payload.vitals) targets.add(this.getRoute(Modality.STRUCTURED_VITALS));

    return Array.from(targets);
  }
}
