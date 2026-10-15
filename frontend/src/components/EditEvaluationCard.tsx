"use client";

import type { ReactNode } from "react";

export function HighlightedEvaluationText({ text, prompt, target, groundTruth, completionOnly = false }: {
  text: string; prompt: string; target: string; groundTruth: string; completionOnly?: boolean;
}) {
  const startsWithPrompt = (value: string) => Boolean(prompt.trim()) && value.slice(0, prompt.length).toLowerCase() === prompt.toLowerCase();
  // EasyEdit returns a continuation; the workbench generator includes the prompt.
  // Normalize only the display, leaving the evaluated text and scores unchanged.
  const displayText = completionOnly && prompt.trim() && !startsWithPrompt(text)
    ? `${prompt}${text && !/\s$/.test(prompt) && !/^\s/.test(text) ? " " : ""}${text}`
    : text;
  const values = [target, groundTruth].map((value) => value.trim()).filter(Boolean);
  const escape = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = values.length ? new RegExp(`(${[...new Set(values)].sort((a, b) => b.length - a.length).map(escape).join("|")})`, "gi") : null;
  const highlightAnswers = (value: string) => !pattern ? value : value.split(pattern).map((part, index) => {
    const edited = Boolean(target.trim()) && part.toLowerCase() === target.trim().toLowerCase();
    const original = Boolean(groundTruth.trim()) && part.toLowerCase() === groundTruth.trim().toLowerCase();
    return edited || original ? <mark key={index} title={edited ? "Edited target" : "Ground truth"} className={`rounded px-0.5 font-medium ${edited ? "bg-cyan/15 text-cyan" : "bg-rust/15 text-rust"}`}>{part}</mark> : part;
  });
  return startsWithPrompt(displayText)
    ? <><mark title="Prompt" className="rounded bg-ink/10 px-0.5 font-medium text-ink">{displayText.slice(0, prompt.length)}</mark>{highlightAnswers(displayText.slice(prompt.length))}</>
    : <>{highlightAnswers(displayText)}</>;
}

export function EditEvaluationCard({ prompt, target, groundTruth = "", original, edited, children }: {
  prompt: string; target: string; groundTruth?: string; original?: string | null; edited: string; children?: ReactNode;
}) {
  return <div className="h-max min-w-0 overflow-hidden rounded-lg border border-line">
    <p className="break-words border-b border-line bg-panel px-4 py-3 text-xs font-medium leading-5">{prompt}</p>
    <div className="flex flex-wrap gap-x-4 gap-y-1 border-b border-line bg-panel/60 px-4 py-2 text-xs leading-5 text-ink/60">
      {groundTruth ? <span>Ground truth: <span className="font-medium text-rust">{groundTruth}</span></span> : null}
      {target ? <span>Edited target: <span className="font-medium text-cyan">{target}</span></span> : null}{children}
    </div>
    <div className={`grid divide-y divide-line ${original != null ? "sm:grid-cols-2 sm:divide-x sm:divide-y-0" : ""}`}>
      {original != null ? <div className="min-w-0 p-4"><div className="label">Original model</div><p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6"><HighlightedEvaluationText text={original} prompt={prompt} target={target} groundTruth={groundTruth} /></p></div> : null}
      <div className="min-w-0 p-4"><div className="label">Edited model</div><p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6"><HighlightedEvaluationText text={edited || "Waiting for generation…"} prompt={prompt} target={target} groundTruth={groundTruth} /></p></div>
    </div>
  </div>;
}
