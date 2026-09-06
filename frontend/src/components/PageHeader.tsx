import { ApiStatus } from "./DataState";

export function PageHeader({ eyebrow, title, description, status }: { eyebrow: string; title: string; description: string; status: "loading" | "ready" | "offline" }) {
  return (
    <section className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-ink sm:text-[34px]">{title}</h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-ink/60">{description}</p>
      </div>
      <ApiStatus status={status} />
    </section>
  );
}
