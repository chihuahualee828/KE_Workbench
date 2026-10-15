import assert from "node:assert/strict";
import test from "node:test";

import {
  detectEvaluationFactMapping, normalizeEvaluationFactImport, parseEvaluationFactImport,
  type EvaluationFactFieldMapping
} from "../src/lib/evaluationFactImport";

const originalPrompt = "Danielle Darrieux's mother tongue is";
const base = { subject: "Danielle Darrieux", prompt: originalPrompt, target_new: "English" };
const importFacts = (records: Record<string, unknown>[]) => normalizeEvaluationFactImport(records, detectEvaluationFactMapping(records));

test("the uploaded sample becomes a canonical evaluation fact with all metric prompts", () => {
  const records = parseEvaluationFactImport(JSON.stringify({
    ...base, prompt: ` ${originalPrompt} `, target_new: " English ", ground_truth: "French",
    rephrase_prompt: "What language did Danielle Darrieux first speak?",
    locality_prompt: "The capital of France is", locality_ground_truth: "Paris",
    portability_prompt: "Danielle Darrieux spoke to her parents in", portability_ground_truth: "English"
  }));
  assert.deepEqual(importFacts(records), [{
    key: JSON.stringify([originalPrompt, "English"]), subject: "Danielle Darrieux",
    relation_prompt: "{}'s mother tongue is", prompt: originalPrompt, new_object: "English", ground_truth: "French",
    rephrase_prompt: "What language did Danielle Darrieux first speak?",
    locality_prompt: "The capital of France is", locality_ground_truth: "Paris",
    portability_prompt: "Danielle Darrieux spoke to her parents in", portability_ground_truth: "English"
  }]);
});

test("JSON accepts a single object or an array and clearly rejects malformed, empty, and invalid rows", () => {
  assert.deepEqual(parseEvaluationFactImport(JSON.stringify(base)), [base]);
  assert.deepEqual(parseEvaluationFactImport(JSON.stringify([base, base])), [base, base]);
  for (const value of ["{", "", "   ", "[1,"]) {
    assert.throws(() => parseEvaluationFactImport(value), /malformed/i);
  }
  assert.throws(() => parseEvaluationFactImport("[]"), /at least one fact/i);
  for (const value of ["null", "1", '"fact"', "true"]) {
    assert.throws(() => parseEvaluationFactImport(value), /fact object or an array/i);
  }
  for (const row of [null, false, 1, "fact", []]) {
    assert.throws(() => parseEvaluationFactImport(JSON.stringify([base, row])), /Fact 2 must be a JSON object/);
  }
});

test("all edit-page aliases and case or punctuation variants are detected", () => {
  const aliases = {
    subject: ["subject", "entity", "entity_name", "name"],
    prompt: ["prompt", "relation_prompt", "relation", "template"],
    new_object: ["new_object", "target_new", "new_target", "target"],
    ground_truth: ["ground_truth", "target_true", "target_old", "original_object"]
  } as const;
  const canonical = { subject: base.subject, prompt: originalPrompt, new_object: "English", ground_truth: "French" };
  for (const field of Object.keys(aliases) as Array<keyof typeof aliases>) {
    for (const alias of aliases[field]) {
      const record: Record<string, unknown> = { ...canonical };
      delete record[field];
      const key = ` ${alias.toUpperCase().replaceAll("_", " .-")} `;
      record[key] = canonical[field];
      const mapping = detectEvaluationFactMapping([record]);
      assert.equal(mapping[field], key);
      assert.equal(normalizeEvaluationFactImport([record], mapping)[0][field], canonical[field]);
    }
  }
});

test("optional aliases are detected across the union of keys in the file", () => {
  const records = [
    { ...base, "Rephrase.Prompts": [" ", "Which language did she first speak?"] },
    { ...base, target_new: "Spanish", "Locality-Prompt": "Paris is in", "Locality.Ground Truth": "France" },
    { ...base, target_new: "German", "Portability Prompt": "She spoke", "Portability-Ground-Truth": "German" }
  ];
  const mapping = detectEvaluationFactMapping(records);
  assert.equal(mapping.rephrase_prompt, "Rephrase.Prompts");
  assert.equal(mapping.locality_prompt, "Locality-Prompt");
  assert.equal(mapping.locality_ground_truth, "Locality.Ground Truth");
  assert.equal(mapping.portability_prompt, "Portability Prompt");
  assert.equal(mapping.portability_ground_truth, "Portability-Ground-Truth");
  const facts = normalizeEvaluationFactImport(records, mapping);
  assert.equal(facts[0].rephrase_prompt, "Which language did she first speak?");
  assert.equal(facts[1].rephrase_prompt, originalPrompt);
  assert.equal(facts[2].portability_ground_truth, "German");
});

test("missing ground truth and auxiliary prompts stay optional with a per-fact rephrase fallback", () => {
  const records = [
    { ...base, rephrase_prompt: "Which language did she first speak?" },
    { subject: "Paris", prompt: "{} is the capital of", target_new: "Germany", rephrase_prompt: " " }
  ];
  const facts = importFacts(records);
  assert.equal(facts[0].rephrase_prompt, "Which language did she first speak?");
  assert.equal(facts[1].rephrase_prompt, "Paris is the capital of");
  for (const fact of facts) {
    assert.equal(fact.ground_truth, "");
    assert.equal(fact.locality_prompt, "");
    assert.equal(fact.locality_ground_truth, "");
    assert.equal(fact.portability_prompt, "");
    assert.equal(fact.portability_ground_truth, "");
  }
});

test("exact prompt and target duplicates merge missing metadata while conflicting targets remain separate", () => {
  const records = [
    { ...base, ground_truth: "French" },
    { ...base, prompt: "{}'s mother tongue is", rephrase_prompt: "Which language did she first speak?", locality_prompt: "Paris is in" },
    { ...base, target_new: "Spanish", ground_truth: "French" },
    { ...base, ground_truth: "Conflicting old target", locality_ground_truth: "France", portability_prompt: "She spoke", portability_ground_truth: "English" }
  ];
  const facts = importFacts(records);
  assert.equal(facts.length, 2);
  assert.equal(facts[0].key, JSON.stringify([originalPrompt, "English"]));
  assert.equal(facts[1].key, JSON.stringify([originalPrompt, "Spanish"]));
  assert.equal(facts[0].ground_truth, "French");
  assert.equal(facts[0].rephrase_prompt, "Which language did she first speak?");
  assert.equal(facts[0].locality_prompt, "Paris is in");
  assert.equal(facts[0].locality_ground_truth, "France");
  assert.equal(facts[0].portability_prompt, "She spoke");
  assert.equal(facts[0].portability_ground_truth, "English");
});

test("an invalid mapped row reports its index and required fields", () => {
  const records = [base, { subject: " ", prompt: "The language is", target_new: { unexpected: "English" } }];
  assert.throws(() => importFacts(records), /Fact 2 is missing subject, new object for the selected mapping/);
  assert.throws(() => normalizeEvaluationFactImport([], detectEvaluationFactMapping([])), /at least one fact/);
});

test("manual mappings import unknown names and validate both required and optional selected attributes", () => {
  const records = [{ person: base.subject, question: originalPrompt, answer: "English", old_answer: "French", paraphrase: "Her first language is" }];
  const mapping: EvaluationFactFieldMapping = {
    ...detectEvaluationFactMapping(records), subject: "person", prompt: "question", new_object: "answer",
    ground_truth: "old_answer", rephrase_prompt: "paraphrase"
  };
  const [fact] = normalizeEvaluationFactImport(records, mapping);
  assert.equal(fact.subject, base.subject);
  assert.equal(fact.ground_truth, "French");
  assert.equal(fact.rephrase_prompt, "Her first language is");
  assert.throws(() => normalizeEvaluationFactImport(records, { ...mapping, subject: "" }), /Choose a JSON attribute for Subject/);
  assert.throws(() => normalizeEvaluationFactImport(records, { ...mapping, prompt: "missing" }), /JSON attribute "missing" was not found/);
  assert.throws(() => normalizeEvaluationFactImport(records, { ...mapping, locality_prompt: "missing" }), /JSON attribute "missing" was not found/);
});

test("full prompts are preserved and relation templates render their subject without adding it twice", () => {
  const [full] = importFacts([base]);
  const [template] = importFacts([{ subject: base.subject, relation_prompt: "{}'s mother tongue is", new_object: "English" }]);
  assert.equal(template.prompt, full.prompt);
  assert.equal(template.relation_prompt, full.relation_prompt);
  assert.equal(template.key, full.key);
  const [unrelated] = importFacts([{ ...base, prompt: " State the language {code} ", relation_prompt: "{} is irrelevant" }]);
  assert.equal(unrelated.prompt, "State the language {code}");
  assert.equal(unrelated.relation_prompt, "State the language {code}");
  const [literalSubject] = importFacts([{ subject: "C++ [city]", prompt: "C++ [city] and C++ [city] are", new_object: "same" }]);
  assert.equal(literalSubject.relation_prompt, "{} and {} are");
  const [dollarSubject] = importFacts([{ subject: "$&", relation_prompt: "{} is a literal subject", target_new: "symbol" }]);
  assert.equal(dollarSubject.prompt, "$& is a literal subject");
});

test("nested EasyEdit targets and grouped auxiliary questions are supported", () => {
  const [fact] = importFacts([{
    ...base, target_new: { str: " English ", id: "Q1860" }, target_true: { str: "French" },
    rephrase_prompts: [null, "", "Her mother tongue is"],
    locality: { neighborhood: { prompt: ["Paris is in"], ground_truth: ["France"] } },
    portability: { prompt: "She spoke", ground_truth: { str: "English" } }
  }]);
  assert.equal(fact.new_object, "English");
  assert.equal(fact.ground_truth, "French");
  assert.equal(fact.rephrase_prompt, "Her mother tongue is");
  assert.equal(fact.locality_prompt, "Paris is in");
  assert.equal(fact.locality_ground_truth, "France");
  assert.equal(fact.portability_prompt, "She spoke");
  assert.equal(fact.portability_ground_truth, "English");
});
