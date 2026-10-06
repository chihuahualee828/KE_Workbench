export function formatLayerRanges(layers: readonly number[]): string {
  const segments: string[] = [];
  for (let start = 0; start < layers.length;) {
    let end = start;
    while (end + 1 < layers.length && layers[end + 1] === layers[end] + 1) end += 1;
    if (end - start >= 2) {
      segments.push(`${layers[start]}~${layers[end]}`);
    } else {
      segments.push(...layers.slice(start, end + 1).map(String));
    }
    start = end + 1;
  }
  return segments.join(", ");
}
