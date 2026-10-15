import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { EvaluationProgressIndicator } from "../src/components/EvaluationProgressIndicator";

test("baseline progress is visible before the first edited fact completes", () => {
  for (const current of [0, 19]) {
    const html = renderToStaticMarkup(createElement(EvaluationProgressIndicator, { current, total: 1919, phase: "baseline" }));
    assert.match(html, /Evaluating original model/);
    assert.ok(html.includes(`${current} / 1919 facts`));
    assert.doesNotMatch(html, /Loading evaluator|Preparing EasyEdit/);
  }
});

test("edited evaluation at zero does not regress to evaluator loading", () => {
  const html = renderToStaticMarkup(createElement(EvaluationProgressIndicator, { current: 0, total: 1919, phase: "evaluation" }));
  assert.match(html, /Running evaluation/);
  assert.match(html, /0 \/ 1919 facts/);
  assert.doesNotMatch(html, /Loading evaluator/);
  const loading = renderToStaticMarkup(createElement(EvaluationProgressIndicator, {}));
  assert.match(loading, /Loading evaluator/);
});

test("single-fact baseline evaluation retains an animated progress bar", () => {
  const html = renderToStaticMarkup(createElement(EvaluationProgressIndicator, { current: 0, total: 1, phase: "baseline" }));
  assert.match(html, /Evaluating original model/);
  assert.match(html, /edit-progress-indeterminate/);
});
