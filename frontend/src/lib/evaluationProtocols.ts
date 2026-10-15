import type { EvaluationMetric, EvaluationProtocol, EvaluationProtocols, EvaluationResultOverview } from "./types";

export const EVALUATION_METRICS: EvaluationMetric[] = ["reliability", "generalization", "locality", "portability"];

const TRADITIONAL_SCORE_LABELS: Record<EvaluationMetric, string> = {
  reliability: "Reliability rewrite token accuracy",
  generalization: "Generalization rephrase token accuracy",
  locality: "Locality token preservation",
  portability: "Portability token accuracy",
};

export function allowedEvaluationProtocols(metric: EvaluationMetric): EvaluationProtocol[] {
  return metric === "locality" ? ["traditional", "generate-text"] : ["traditional", "LLM-judge"];
}

export function evaluationProtocol(metric: EvaluationMetric, protocols?: EvaluationProtocols): EvaluationProtocol {
  const protocol = protocols?.[metric];
  // Saved generation-only results retain their original labels and scoring.
  if (protocol === "generate-text") return protocol;
  return protocol && allowedEvaluationProtocols(metric).includes(protocol) ? protocol : "traditional";
}

export function enabledEvaluationProtocols(metrics: EvaluationMetric[], protocols?: EvaluationProtocols): EvaluationProtocols {
  return Object.fromEntries(metrics.map((metric) => {
    const protocol = protocols?.[metric];
    return [metric, protocol && allowedEvaluationProtocols(metric).includes(protocol) ? protocol : "traditional"];
  }));
}

export function evaluationMetricFromKey(key: string): EvaluationMetric | undefined {
  return EVALUATION_METRICS.find((metric) => key.startsWith(`${metric}_`));
}

export function evaluationScoreDisplay(metric: EvaluationMetric, score: unknown, protocols?: EvaluationProtocols) {
  const protocol = evaluationProtocol(metric, protocols);
  const name = metric.charAt(0).toUpperCase() + metric.slice(1);
  const generatedLocality = metric === "locality" && protocol === "generate-text";
  const generationOnly = protocol === "generate-text" && !generatedLocality;
  return {
    protocol,
    label: generatedLocality ? "Locality generated-text preservation" : protocol === "LLM-judge" ? `${name} judge accuracy` : generationOnly ? `${name} accuracy` : TRADITIONAL_SCORE_LABELS[metric],
    value: !generationOnly && typeof score === "number" && Number.isFinite(score) ? score.toFixed(3) : "N/A",
    caption: generatedLocality ? "Original/edited generated-text exact match." : generationOnly ? "Generation only; no accuracy score." : protocol === "LLM-judge" ? "LLM-judge" : "EasyEdit",
  };
}

export function evaluationMetricEntries(overview: EvaluationResultOverview): Array<[string, unknown]> {
  const entries = Object.entries(overview.metrics);
  // Generation-only runs still need a selectable tile when no score is stored.
  return overview.evaluation_modes.flatMap((metric) => {
    const metricEntries = entries.filter(([key]) => evaluationMetricFromKey(key) === metric);
    return metricEntries.some(([key]) => !key.includes("_fluency_"))
      ? metricEntries
      : [[`${metric}_score`, null] as [string, unknown], ...metricEntries];
  }).concat(entries.filter(([key]) => !evaluationMetricFromKey(key)));
}
