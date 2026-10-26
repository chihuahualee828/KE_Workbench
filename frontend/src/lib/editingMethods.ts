export const batchWeightMethods = {
  memit: "MEMIT", emmet: "EMMET", memit_fe: "MEMIT-FE", pmet: "PMET", eamet: "EAMET", alphaedit: "AlphaEdit", sphere: "SPHERE",
} as const;
export const trainingMethods = { ft: "FT", lora: "LoRA", dpo: "DPO" } as const;
export const editMethods = {
  rome: "ROME", r_rome: "r-ROME", kn: "KN", memit: "MEMIT", ft: "FT", lora: "LoRA",
  emmet: "EMMET", memit_fe: "MEMIT-FE", pmet: "PMET", dpo: "DPO",
  eamet: "EAMET", alphaedit: "AlphaEdit", sphere: "SPHERE",
} as const;
export type BatchWeightMethod = keyof typeof batchWeightMethods;
export type TrainingMethod = keyof typeof trainingMethods;
export type EditMethod = keyof typeof editMethods;
export type EditJobOptions = { seed?: number };
export const DEFAULT_EDIT_SEED = 42;
export const MAX_EDIT_SEED = 2_147_483_647;

export function supportsEditSeed(method: EditMethod): boolean {
  // Current FT/KN paths reuse the eval-mode model without sampling or random initialization.
  return method !== "ft" && method !== "kn";
}

export function editSeedOptions(method: EditMethod, value: string): EditJobOptions {
  return supportsEditSeed(method) ? { seed: parseEditSeed(value) } : {};
}

export function parseEditSeed(value: string): number {
  const seed = Number(value);
  if (!value.trim() || !Number.isInteger(seed) || seed < 0 || seed > MAX_EDIT_SEED) {
    throw new Error(`Random seed must be a whole number from 0 to ${MAX_EDIT_SEED}.`);
  }
  return seed;
}

export type Mom2Method = "rome" | "r_rome" | BatchWeightMethod;
export const mom2Methods: Mom2Method[] = ["rome", "r_rome", ...Object.keys(batchWeightMethods) as BatchWeightMethod[]];
export function isBatchEditMethod(method: unknown): boolean {
  return typeof method === "string" && (Object.hasOwn(batchWeightMethods, method) || Object.hasOwn(trainingMethods, method));
}
