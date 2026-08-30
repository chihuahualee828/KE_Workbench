"use client";

export function PromptEditor({
  value,
  onChange,
  rows = 5
}: {
  value: string;
  onChange: (value: string) => void;
  rows?: number;
}) {
  return (
    <label className="grid gap-2">
      <span className="label">Prompt</span>
      <textarea className="control min-h-32 resize-y" rows={rows} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

