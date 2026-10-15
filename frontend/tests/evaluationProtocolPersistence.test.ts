import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";

test("evaluation protocol persistence reuses a module default to avoid storage hydration render loops", () => {
  const sourcePath = [
    resolve(process.cwd(), "src/components/EvaluationDashboard.tsx"),
    resolve(process.cwd(), "frontend/src/components/EvaluationDashboard.tsx"),
    resolve(__dirname, "../src/components/EvaluationDashboard.tsx"),
  ].find(existsSync);
  assert.ok(sourcePath, "Locate the actual EvaluationDashboard source for this regression guard.");
  const source = readFileSync(sourcePath, "utf8");
  const componentStart = source.indexOf("export function EvaluationDashboard");
  assert.ok(componentStart >= 0);
  assert.match(source.slice(0, componentStart), /\bconst\s+DEFAULT_EVALUATION_PROTOCOLS\s*:\s*EvaluationProtocols\s*=\s*\{\s*\}\s*;/);

  const componentSource = source.slice(componentStart);
  const calls = [...componentSource.matchAll(/\busePersistentState(?:<[^>]*>)?\(\s*["']kew:evaluate:evaluationProtocols["']\s*,\s*([^)]*)\)/g)];
  assert.equal(calls.length, 1);
  assert.equal(calls[0][1].trim(), "DEFAULT_EVALUATION_PROTOCOLS",
    "A fresh {} default changes the hydration effect dependency every render and repeatedly reparses persisted storage.");
  assert.doesNotMatch(componentSource, /\b(?:const|let|var)\s+DEFAULT_EVALUATION_PROTOCOLS\b/,
    "The persistent default must keep its module identity rather than be recreated inside the component.");
});
