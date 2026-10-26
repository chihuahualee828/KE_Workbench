import type { TrainingMethod } from "./editingMethods";
export type TrainingHyperparameters = {
  lr?: number;
  num_steps?: number;
  weight_decay?: number;
  objective_optimization?: "prompt_last" | "target_new";
  norm_constraint?: number | false;
  target_modules?: string[];
  rank?: number;
  lora_alpha?: number;
  lora_dropout?: number;
  alpha?: number;
  beta?: number;
};

export const trainingFields = (method: TrainingMethod) => [
  { key: "lr", label: "Learning rate", hint: "Optimizer step size, e.g. 0.0005 or 5e-4." },
  { key: "num_steps", label: "Training steps", hint: "EasyEdit training iterations per edit batch." },
  { key: "weight_decay", label: "Weight decay", hint: "Adam weight decay; 0 disables it." },
  ...(method === "ft" ? [
    { key: "objective_optimization", label: "FT objective", hint: "prompt_last uses the final prompt position; target_new trains the target continuation." },
    { key: "norm_constraint", label: "Norm constraint", hint: "Maximum parameter change from the batch baseline. Enter false to disable." },
  ] : [
    { key: "target_modules", label: "Target modules", hint: "Comma-separated projection names, e.g. q_proj, v_proj. Names depend on the model architecture." },
    { key: "rank", label: "LoRA rank", hint: "Positive integer dimension of each low-rank adapter." },
    { key: "lora_alpha", label: "LoRA alpha", hint: "Adapter scaling factor (alpha / rank)." },
    { key: "lora_dropout", label: "LoRA dropout", hint: "Dropout probability from 0 up to, but excluding, 1." },
  ]),
  ...(method === "dpo" ? [
    { key: "alpha", label: "DPO loss mix", hint: "Weight of desired-answer training (0–1); the remainder is preference loss." },
    { key: "beta", label: "DPO beta", hint: "Positive scaling factor for the preference objective." },
  ] : []),
] as Array<{ key: keyof TrainingHyperparameters; label: string; hint: string }>;

export function trainingValueText(value: TrainingHyperparameters[keyof TrainingHyperparameters]): string {
  return Array.isArray(value) ? value.join(", ") : value === undefined ? "" : String(value);
}

export function parseTrainingHyperparameters(method: TrainingMethod, values: Record<string, string>): TrainingHyperparameters {
  const result: Record<string, unknown> = {};
  for (const { key, label } of trainingFields(method)) {
    const value = values[key]?.trim();
    if (!value) throw new Error(`${label} is required.`);
    if (key === "target_modules") {
      const modules = value.split(",").map((item) => item.trim());
      if (modules.some((item) => !item || item === "all-linear")) throw new Error("Enter individual target module names separated by commas.");
      result[key] = [...new Set(modules)];
    } else if (key === "objective_optimization") {
      if (value !== "prompt_last" && value !== "target_new") throw new Error("Select an FT objective.");
      result[key] = value;
    } else if (key === "norm_constraint" && value === "false") {
      result[key] = false;
    } else {
      const number = Number(value);
      const positive = ["lr", "num_steps", "rank", "lora_alpha", "beta"].includes(key);
      const integer = ["num_steps", "rank"].includes(key);
      if (!Number.isFinite(number) || number < 0 || (positive && number === 0) ||
          (integer && !Number.isSafeInteger(number)) || (key === "lora_dropout" && number >= 1) || (key === "alpha" && number > 1)) {
        throw new Error(`Invalid ${label.toLowerCase()}. ${integer ? "Use a positive whole number." : key === "alpha" ? "Use a value from 0 to 1." : key === "lora_dropout" ? "Use a value from 0 up to, but excluding, 1." : positive ? "Use a positive number." : "Use a nonnegative number."}`);
      }
      result[key] = number;
    }
  }
  return result as TrainingHyperparameters;
}
