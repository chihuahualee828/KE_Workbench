import type { EvaluationFact } from "./types";

export const REQUIRED_EVALUATION_FACT_IMPORT_FIELDS = ["subject", "prompt", "new_object"] as const;
export const OPTIONAL_EVALUATION_FACT_IMPORT_FIELDS = [
  "ground_truth", "rephrase_prompt", "locality_prompt", "locality_ground_truth", "portability_prompt", "portability_ground_truth"
] as const;
export const EVALUATION_FACT_IMPORT_FIELDS = [...REQUIRED_EVALUATION_FACT_IMPORT_FIELDS, ...OPTIONAL_EVALUATION_FACT_IMPORT_FIELDS] as const;
export type EvaluationFactImportField = typeof EVALUATION_FACT_IMPORT_FIELDS[number];
export type EvaluationFactFieldMapping = Record<EvaluationFactImportField, string>;

const FIELD_ALIASES: Record<EvaluationFactImportField, readonly string[]> = {
  subject: ["subject", "entity", "entityname", "name"],
  prompt: ["prompt", "relationprompt", "relation", "template"],
  new_object: ["newobject", "targetnew", "newtarget", "target"],
  ground_truth: ["groundtruth", "targettrue", "targetold", "originalobject"],
  rephrase_prompt: ["rephraseprompt", "rephraseprompts"],
  locality_prompt: ["localityprompt", "localityprompts", "locality"],
  locality_ground_truth: ["localitygroundtruth", "locality"],
  portability_prompt: ["portabilityprompt", "portabilityprompts", "portability"],
  portability_ground_truth: ["portabilitygroundtruth", "portability"]
};

export function parseEvaluationFactImport(value: string): Record<string, unknown>[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error("Evaluation facts JSON is malformed.");
  }
  if (!Array.isArray(parsed) && !isRecord(parsed)) {
    throw new Error("Evaluation facts JSON must be a fact object or an array of fact objects.");
  }
  const records: unknown[] = Array.isArray(parsed) ? parsed : [parsed];
  if (!records.length) throw new Error("Evaluation facts JSON must include at least one fact.");
  return records.map((record, index) => {
    if (!isRecord(record)) throw new Error(`Fact ${index + 1} must be a JSON object.`);
    return record;
  });
}

export function detectEvaluationFactMapping(records: Record<string, unknown>[]): EvaluationFactFieldMapping {
  const keys = new Map<string, string>();
  for (const record of records) {
    for (const key of Object.keys(record)) {
      const normalized = normalizeImportKey(key);
      if (!keys.has(normalized)) keys.set(normalized, key);
    }
  }
  return Object.fromEntries(EVALUATION_FACT_IMPORT_FIELDS.map((field) => [
    field, FIELD_ALIASES[field].map((alias) => keys.get(alias)).find((key) => key !== undefined) ?? ""
  ])) as EvaluationFactFieldMapping;
}

export function normalizeEvaluationFactImport(records: Record<string, unknown>[], mapping: EvaluationFactFieldMapping): EvaluationFact[] {
  if (!records.length) throw new Error("Evaluation facts JSON must include at least one fact.");
  const keys = new Set(records.flatMap((record) => Object.keys(record)));
  for (const field of REQUIRED_EVALUATION_FACT_IMPORT_FIELDS) {
    if (!mapping[field]?.trim()) throw new Error(`Choose a JSON attribute for ${evaluationFactImportFieldLabel(field)}.`);
  }
  for (const field of EVALUATION_FACT_IMPORT_FIELDS) {
    const key = mapping[field];
    if (key?.trim() && !keys.has(key)) throw new Error(`JSON attribute "${key}" was not found in this file.`);
  }

  const facts = new Map<string, EvaluationFact>();
  records.forEach((record, index) => {
    const subject = readField(record, mapping, "subject");
    const sourcePrompt = readField(record, mapping, "prompt");
    const newObject = readField(record, mapping, "new_object");
    const missing = [!subject && "subject", !sourcePrompt && "prompt", !newObject && "new object"].filter(Boolean);
    if (missing.length) throw new Error(`Fact ${index + 1} is missing ${missing.join(", ")} for the selected mapping.`);

    const prompt = sourcePrompt.replaceAll("{}", () => subject);
    const fact: EvaluationFact = {
      key: JSON.stringify([prompt, newObject]),
      subject,
      relation_prompt: sourcePrompt.includes("{}") ? sourcePrompt : sourcePrompt.replaceAll(subject, "{}"),
      prompt,
      new_object: newObject,
      ground_truth: readField(record, mapping, "ground_truth"),
      rephrase_prompt: readField(record, mapping, "rephrase_prompt") || prompt,
      locality_prompt: readField(record, mapping, "locality_prompt"),
      locality_ground_truth: readField(record, mapping, "locality_ground_truth"),
      portability_prompt: readField(record, mapping, "portability_prompt"),
      portability_ground_truth: readField(record, mapping, "portability_ground_truth")
    };
    const existing = facts.get(fact.key);
    if (!existing) {
      facts.set(fact.key, fact);
      return;
    }
    for (const field of OPTIONAL_EVALUATION_FACT_IMPORT_FIELDS) {
      if (!existing[field] || (field === "rephrase_prompt" && existing[field] === existing.prompt)) {
        existing[field] = fact[field];
      }
    }
  });
  return Array.from(facts.values());
}

export function evaluationFactImportFieldLabel(field: EvaluationFactImportField): string {
  return ({
    subject: "Subject", prompt: "Prompt / Relation Prompt", new_object: "New Object", ground_truth: "Ground Truth",
    rephrase_prompt: "Rephrase Prompt", locality_prompt: "Locality Prompt", locality_ground_truth: "Locality Ground Truth",
    portability_prompt: "Portability Prompt", portability_ground_truth: "Portability Ground Truth"
  } as const)[field];
}

function readField(record: Record<string, unknown>, mapping: EvaluationFactFieldMapping, field: EvaluationFactImportField): string {
  const key = mapping[field];
  if (!key?.trim()) return "";
  const value = record[key];
  const group = field.startsWith("locality_") ? "locality" : field.startsWith("portability_") ? "portability" : "";
  if (group && normalizeImportKey(key) === group && isRecord(value)) {
    return readMetricField(value, field.endsWith("_ground_truth") ? "ground_truth" : "prompt");
  }
  return importText(value, field === "rephrase_prompt" || Boolean(group));
}

function readMetricField(value: Record<string, unknown>, field: "prompt" | "ground_truth"): string {
  const direct = importText(value[field], true);
  if (direct) return direct;
  // EasyEdit datasets may group auxiliary questions under names such as "neighborhood".
  for (const nested of Object.values(value)) {
    if (isRecord(nested)) {
      const text = importText(nested[field], true);
      if (text) return text;
    }
  }
  return "";
}

function importText(value: unknown, allowArray = false): string {
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (isRecord(value) && typeof value.str === "string") return value.str.trim();
  if (allowArray && Array.isArray(value)) {
    return value.map((item) => importText(item)).find(Boolean) ?? "";
  }
  return "";
}

function normalizeImportKey(value: string): string {
  return value.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
