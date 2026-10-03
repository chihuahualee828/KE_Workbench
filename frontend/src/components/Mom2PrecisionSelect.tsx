"use client";

import type { Mom2Dtype, Mom2Profile } from "@/lib/types";

export function formatMom2Precision(dtype: Mom2Dtype) {
  return `FP${dtype.slice(5)}`;
}

export function Mom2PrecisionSelect({ profile, disabled, onChange }: {
  profile: Mom2Profile | null;
  disabled?: boolean;
  onChange: (dtype: Mom2Dtype) => void;
}) {
  return (
    <label className="grid min-w-0 content-start gap-1">
      <span className="label">Precision</span>
      <select className="control" disabled={disabled || !profile} value={profile?.mom2_dtype ?? ""} onChange={(event) => onChange(event.target.value as Mom2Dtype)}>
        {!profile ? <option value="">Loading...</option> : null}
        {(["float16", "float32", "float64"] as const).map((dtype) => (
          <option key={dtype} value={dtype} disabled={!profile?.available_mom2_dtypes.includes(dtype)}>
            {formatMom2Precision(dtype)} ({dtype}){profile && !profile.available_mom2_dtypes.includes(dtype) ? " - unavailable on MPS" : ""}
          </option>
        ))}
      </select>
      <span className="text-xs text-ink/60">EasyEdit default: {profile ? `${formatMom2Precision(profile.default_mom2_dtype)} (${profile.default_mom2_dtype})` : "..."}</span>
    </label>
  );
}
