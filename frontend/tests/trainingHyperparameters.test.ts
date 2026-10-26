import assert from "node:assert/strict";
import test from "node:test";
import { parseTrainingHyperparameters, trainingValueText } from "../src/lib/trainingHyperparameters";

const lora = { lr: "5e-4", num_steps: "2", weight_decay: "0", rank: "8", lora_alpha: "32", lora_dropout: "0.1", target_modules: " q_proj, v_proj " };

test("DPO validates preference loss settings alongside adapter settings", () => {
  const values = { ...lora, alpha: "0.5", beta: "0.2" };
  assert.deepEqual(parseTrainingHyperparameters("dpo", values), {
    ...parseTrainingHyperparameters("lora", lora), alpha: 0.5, beta: 0.2,
  });
  for (const [key, value] of [["alpha", "-0.1"], ["alpha", "1.1"], ["beta", "0"], ["beta", "Infinity"]]) {
    assert.throws(() => parseTrainingHyperparameters("dpo", { ...values, [key]: value }));
  }
});

test("LoRA fields serialize numeric values and target module names", () => {
  assert.deepEqual(parseTrainingHyperparameters("lora", lora), {
    lr: 0.0005, num_steps: 2, weight_decay: 0, rank: 8, lora_alpha: 32, lora_dropout: 0.1, target_modules: ["q_proj", "v_proj"],
  });
});

test("invalid edited values cannot silently become zero or infinity", () => {
  for (const [key, value] of [["lr", ""], ["lr", "0"], ["lr", "Infinity"], ["rank", "1.5"], ["target_modules", "q_proj,"], ["lora_dropout", "1"]]) {
    assert.throws(() => parseTrainingHyperparameters("lora", { ...lora, [key]: value }));
  }
});

test("FT supports disabling the norm constraint and selecting the continuation objective", () => {
  assert.deepEqual(parseTrainingHyperparameters("ft", { lr: "0.01", num_steps: "3", weight_decay: "0", norm_constraint: "false", objective_optimization: "target_new" }), {
    lr: 0.01, num_steps: 3, weight_decay: 0, norm_constraint: false, objective_optimization: "target_new",
  });
  assert.equal(trainingValueText(false), "false");
  assert.equal(trainingValueText(["q_proj", "v_proj"]), "q_proj, v_proj");
});
