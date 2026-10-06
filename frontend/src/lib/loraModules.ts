export type LoraModule = {
  path: string;
  type: string;
  weight_shape: number[];
  available: boolean;
  reason: string | null;
};

export type LoraModuleGroup = { name: string; targets: string[]; modules: LoraModule[] };
export type LoraModuleCatalog = { model_id: string; module_count: number; groups: LoraModuleGroup[] };

function matches(path: string, target: string) {
  return path === target || path.endsWith(`.${target}`);
}

function selectedTargets(value: string) {
  return value.split(",").map((target) => target.trim()).filter(Boolean);
}

export function isLoraGroupSelected(group: LoraModuleGroup, value: string): boolean {
  const available = group.modules.filter((module) => module.available);
  const targets = selectedTargets(value);
  return available.length > 0 && available.every((module) => targets.some((target) => matches(module.path, target)));
}

export function toggleLoraGroup(group: LoraModuleGroup, value: string, checked: boolean): string {
  const paths = group.modules.filter((module) => module.available).map((module) => module.path);
  const remaining = selectedTargets(value).filter((target) => !paths.some((path) => matches(path, target)));
  return [...new Set([...remaining, ...(checked ? group.targets : [])])].join(", ");
}
