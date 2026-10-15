import assert from "node:assert/strict";
import test from "node:test";

import { api, invalidateApiReads } from "../src/lib/api";
import { allowedEvaluationProtocols, enabledEvaluationProtocols, EVALUATION_METRICS, evaluationMetricEntries, evaluationProtocol, evaluationScoreDisplay } from "../src/lib/evaluationProtocols";
import type { EvaluationResultOverview } from "../src/lib/types";

test("enabled metrics default to traditional and locality cannot select LLM-judge", () => {
  assert.deepEqual(enabledEvaluationProtocols(EVALUATION_METRICS), {
    reliability: "traditional", generalization: "traditional", locality: "traditional", portability: "traditional",
  });
  assert.deepEqual(allowedEvaluationProtocols("locality"), ["traditional", "generate-text"]);
  for (const metric of ["reliability", "generalization", "portability"] as const) {
    assert.deepEqual(allowedEvaluationProtocols(metric), ["traditional", "LLM-judge"]);
  }
  assert.equal(evaluationProtocol("locality", { locality: "LLM-judge" }), "traditional");
  assert.deepEqual(enabledEvaluationProtocols(["portability"], { reliability: "LLM-judge", portability: "generate-text" }), { portability: "traditional" });
});

test("locality retains generated-text preservation while stale non-locality setup choices become traditional", () => {
  const saved = { reliability: "generate-text", generalization: "generate-text", locality: "generate-text", portability: "generate-text" } as const;
  const selected = enabledEvaluationProtocols(EVALUATION_METRICS, saved);
  for (const metric of EVALUATION_METRICS) {
    assert.equal(selected[metric], metric === "locality" ? "generate-text" : "traditional");
    assert.equal(allowedEvaluationProtocols(metric).includes("generate-text"), metric === "locality");
    assert.equal(evaluationProtocol(metric, saved), "generate-text");
    assert.equal(saved[metric], "generate-text");
  }
  assert.deepEqual(enabledEvaluationProtocols(["reliability", "locality"], { reliability: "LLM-judge", locality: "LLM-judge" }), {
    reliability: "LLM-judge", locality: "traditional",
  });
});

test("saved protocols label judge accuracy and generation-only results cannot show a numeric score", () => {
  const saved = { reliability: "LLM-judge", portability: "generate-text" } as const;
  assert.equal(evaluationScoreDisplay("reliability", 0.75, saved).label, "Reliability judge accuracy");
  assert.equal(evaluationScoreDisplay("reliability", 0.75, saved).value, "0.750");
  const generationOnly = evaluationScoreDisplay("portability", 0, saved);
  assert.equal(generationOnly.value, "N/A");
  assert.match(generationOnly.caption, /Generation only; no accuracy score/);
  assert.equal(evaluationScoreDisplay("reliability", 0.5).label, "Reliability rewrite token accuracy");
  assert.equal(evaluationScoreDisplay("locality", null).value, "N/A");
});

test("historical generated-text locality retains its numeric preservation score while other generated-text metrics have no accuracy", () => {
  const protocols = {
    reliability: "generate-text", generalization: "generate-text", locality: "generate-text", portability: "generate-text",
  } as const;
  const locality = evaluationScoreDisplay("locality", 0.5, protocols);
  assert.equal(locality.label, "Locality generated-text preservation");
  assert.equal(locality.value, "0.500");
  assert.equal(locality.caption, "Original/edited generated-text exact match.");
  assert.doesNotMatch(locality.caption, /Generation only|no accuracy/);
  assert.equal(evaluationScoreDisplay("locality", 0, protocols).value, "0.000");
  for (const metric of ["reliability", "generalization", "portability"] as const) {
    const display = evaluationScoreDisplay(metric, 0.5, protocols);
    assert.equal(display.value, "N/A");
    assert.equal(display.caption, "Generation only; no accuracy score.");
  }
});

test("record protocols override saved overview protocols with traditional compatibility defaults", () => {
  const overviewProtocols = { reliability: "LLM-judge", portability: "generate-text" } as const;
  const recordProtocols = { portability: "traditional" } as const;
  const protocols = { ...overviewProtocols, ...recordProtocols };
  assert.equal(evaluationScoreDisplay("portability", 0.8, protocols).value, "0.800");
  assert.equal(evaluationScoreDisplay("reliability", 1, protocols).label, "Reliability judge accuracy");
  assert.equal(evaluationScoreDisplay("generalization", 1, protocols).protocol, "traditional");
});

test("generation-only metrics retain a result-selection tile when only fluency is stored", () => {
  const overview: EvaluationResultOverview = {
    model_id: "demo", method: "rome", active_patch_ids: ["patch"], total: 1,
    evaluation_modes: ["reliability", "portability"],
    evaluation_protocols: { reliability: "LLM-judge", portability: "generate-text" },
    available_generation_metrics: ["reliability", "portability"],
    metrics: { reliability_rewrite_acc: 1, portability_fluency_ngram_entropy: 2.5 },
  };
  assert.deepEqual(evaluationMetricEntries(overview), [
    ["reliability_rewrite_acc", 1], ["portability_score", null], ["portability_fluency_ngram_entropy", 2.5],
  ]);
  const portabilityAccuracy = evaluationMetricEntries(overview).find(([key]) => key === "portability_score");
  assert.equal(evaluationScoreDisplay("portability", portabilityAccuracy?.[1], overview.evaluation_protocols).value, "N/A");
});

test("active-model evaluation preserves locality text mode, normalizes stale setup modes, and sends no browser credentials", async () => {
  const original = globalThis.fetch;
  let requestBody: Record<string, unknown> | null = null;
  globalThis.fetch = async (_url, init) => {
    requestBody = JSON.parse(String(init?.body));
    return new Response(JSON.stringify({ job_id: "job", experiment_id: "experiment", status: "running" }));
  };
  try {
    await api.evaluateActiveModel({
      model_id: "demo", evaluation_modes: ["reliability", "locality", "portability"],
      evaluation_protocols: enabledEvaluationProtocols(["reliability", "locality", "portability"], { reliability: "LLM-judge", locality: "generate-text", portability: "generate-text" }),
      fluency_modes: ["portability"], rephrase_overrides: {}, include_generations: true,
    });
    assert.deepEqual(requestBody, {
      model_id: "demo", evaluation_modes: ["reliability", "locality", "portability"],
      evaluation_protocols: { reliability: "LLM-judge", locality: "generate-text", portability: "traditional" },
      fluency_modes: ["portability"], rephrase_overrides: {}, include_generations: true,
    });
  } finally { globalThis.fetch = original; invalidateApiReads(); }
});
