export const batchWeightMethods = { memit: "MEMIT" } as const;
export const trainingMethods = { ft: "FT", lora: "LoRA" } as const;
export const editMethods = {
  rome: "ROME", r_rome: "r-ROME", kn: "KN", memit: "MEMIT", ft: "FT", lora: "LoRA",
} as const;
export type BatchWeightMethod = keyof typeof batchWeightMethods;
export type TrainingMethod = keyof typeof trainingMethods;
export type EditMethod = keyof typeof editMethods;
export type Mom2Method = "rome" | "r_rome" | BatchWeightMethod;
export const mom2Methods: Mom2Method[] = ["rome", "r_rome", ...Object.keys(batchWeightMethods) as BatchWeightMethod[]];
export function isBatchEditMethod(method: unknown): boolean {
  return typeof method === "string" && (Object.hasOwn(batchWeightMethods, method) || Object.hasOwn(trainingMethods, method));
}
