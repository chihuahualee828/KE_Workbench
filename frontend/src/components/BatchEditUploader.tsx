"use client";

export function parseBatchEdits(text: string) {
  if (text.trim().startsWith("[")) {
    return JSON.parse(text);
  }
  return text
    .split("\n")
    .slice(1)
    .filter(Boolean)
    .map((line) => {
      const [subject, relation_prompt, new_object] = line.split(",");
      return { subject, relation_prompt, new_object };
    });
}

