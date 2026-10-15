import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { EditEvaluationCard, HighlightedEvaluationText } from "../src/components/EditEvaluationCard";

const prompt = "Michael Jordan plays the sport of";
const defaults = { prompt, target: "golf", groundTruth: "basketball" };
const renderText = (props: Parameters<typeof HighlightedEvaluationText>[0]) => renderToStaticMarkup(createElement(HighlightedEvaluationText, props));
const plainText = (html: string) => html.replace(/<[^>]*>/g, "");

test("EasyEdit continuations and full workbench generations display the same prompt and text", () => {
  const full = `${prompt} golf at the Golf Links.`;
  const workbench = renderText({ ...defaults, text: full });
  const easyedit = renderText({ ...defaults, text: "golf at the Golf Links.", completionOnly: true });
  assert.equal(easyedit, workbench);
  assert.equal(plainText(easyedit), full);
  assert.match(easyedit, /title="Prompt"/);
  assert.equal((easyedit.match(/title="Edited target"/g) ?? []).length, 2);
});

test("already-complete generations do not receive a duplicate prompt", () => {
  const full = `${prompt} golf`;
  assert.equal(plainText(renderText({ ...defaults, text: full, completionOnly: true })), full);
  const lowerCase = full.toLowerCase();
  assert.equal(plainText(renderText({ ...defaults, text: lowerCase, completionOnly: true })), lowerCase);
});

test("display normalization preserves whitespace and the returned completion", () => {
  const continuation = "\n  golf\nnext line";
  assert.equal(plainText(renderText({ ...defaults, text: continuation, completionOnly: true })), prompt + continuation);
  assert.equal(plainText(renderText({ ...defaults, prompt: prompt + " ", text: "golf", completionOnly: true })), prompt + " golf");
  assert.equal(plainText(renderText({ ...defaults, text: "", completionOnly: true })), prompt);
});

test("only the input prefix is highlighted as the prompt and answer highlights remain", () => {
  const result = renderText({ ...defaults, text: `${prompt} basketball, not golf. ${prompt}` });
  assert.equal((result.match(/title="Prompt"/g) ?? []).length, 1);
  assert.match(result, /title="Ground truth" class="[^"]*text-rust/);
  assert.match(result, /title="Edited target" class="[^"]*text-cyan/);
});

test("special characters are literal and answer highlights never overlap the prompt", () => {
  const result = renderText({ prompt: "C++ [city]?", target: "C++", groundTruth: "[city]", text: "C++ [city]? C++" });
  assert.equal(plainText(result), "C++ [city]? C++");
  assert.equal((result.match(/title="Prompt"/g) ?? []).length, 1);
  assert.equal((result.match(/title="Edited target"/g) ?? []).length, 1);
  assert.doesNotMatch(result, /title="Ground truth"/);
  assert.match(result, /title="Prompt"[^>]*>C\+\+ \[city\]\?<\/mark>/);
});

test("BBC is highlighted only outside the BBC One prompt for both generation sources", () => {
  const fact = { prompt: "BBC One, by ", target: "ITV", groundTruth: "BBC" };
  for (const props of [
    { ...fact, text: "BBC One, by BBC, not ITV" },
    { ...fact, text: "BBC, not ITV", completionOnly: true },
  ]) {
    const result = renderText(props);
    assert.equal(plainText(result), "BBC One, by BBC, not ITV");
    assert.match(result, /title="Prompt"[^>]*>BBC One, by <\/mark>/);
    assert.equal((result.match(/title="Ground truth"/g) ?? []).length, 1);
    assert.equal((result.match(/title="Edited target"/g) ?? []).length, 1);
  }
});

test("missing generation placeholders are not prefixed and locality has no edited-target highlights", () => {
  assert.equal(renderText({ ...defaults, text: "—" }), "—");
  const result = renderText({ prompt: "Lidingö, which is located in", target: "", groundTruth: "Sweden", text: "Sweden", completionOnly: true });
  assert.match(result, /title="Prompt"/);
  assert.match(result, /title="Ground truth"/);
  assert.doesNotMatch(result, /title="Edited target"/);
});

test("edit-page result cards highlight the prompt in both original and edited outputs", () => {
  const result = renderToStaticMarkup(createElement(EditEvaluationCard, {
    ...defaults, original: `${prompt} basketball`, edited: `${prompt} golf`,
  }));
  assert.equal((result.match(/title="Prompt"/g) ?? []).length, 2);
  assert.match(result, /title="Ground truth"/);
  assert.match(result, /title="Edited target"/);
});
