"use client";

import { Activity, ArrowUpRight, Blocks, ChevronRight, FlaskConical, Layers3, LayoutDashboard, MessageSquare, Network, Settings2, SlidersHorizontal } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { useWorkbenchData } from "./DataState";
import { GpuVramMonitor } from "./GpuVramMonitor";
import { TopbarJobsMenu } from "./TopbarJobsMenu";

const navigation = [
  { href: "/", label: "Overview", icon: LayoutDashboard, group: "Workspace" },
  { href: "/playground", label: "Generate", icon: MessageSquare, group: "Explore" },
  { href: "/logit-lens", label: "Logit lens", icon: Layers3, group: "Explore" },
  { href: "/trace", label: "Causal tracing", icon: Network, group: "Explore" },
  { href: "/edit", label: "Knowledge edits", icon: SlidersHorizontal, group: "Experiment" },
  { href: "/evaluate", label: "Evaluation", icon: FlaskConical, group: "Experiment" },
  { href: "/jobs", label: "Activity", icon: Activity, group: "Manage" },
  { href: "/settings", label: "Settings", icon: Settings2, group: "Manage" }
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const current = navigation.find((item) => item.href === pathname);
  const { runtime } = useWorkbenchData();

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <aside className="app-sidebar">
        <Link href="/" className="brand-link" aria-label="Knowledge Editing Workbench home">
          <span className="brand-mark"><Blocks size={22} strokeWidth={1.7} /></span>
          <span><span className="block text-[15px] font-semibold tracking-tight">Workbench</span><span className="mt-0.5 block text-[11px] text-slate-400">Knowledge editing studio</span></span>
        </Link>
        <nav aria-label="Main navigation" className="sidebar-navigation">
          {navigation.map((item, index) => (
            <div key={item.href}>
              {navigation[index - 1]?.group !== item.group ? <div className="nav-group-label">{item.group}</div> : null}
              <Link href={item.href} className={`nav-link ${pathname === item.href ? "nav-link-active" : ""}`} aria-current={pathname === item.href ? "page" : undefined}>
                <item.icon size={18} strokeWidth={1.7} /><span>{item.label}</span>
              </Link>
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <GpuVramMonitor runtime={runtime} variant="sidebar" />
          <div className="mb-2 mt-4 flex items-center gap-2 text-xs font-medium text-slate-200"><span className="h-1.5 w-1.5 rounded-full bg-[rgb(var(--accent-light))]" />Your local workspace</div>
          <p className="text-xs leading-5 text-slate-400">Explore how models know.<br />Shape what they remember.</p>
          <Link href="/settings" className="mt-4 inline-flex items-center gap-2 text-xs text-slate-300 hover:text-white">Manage workspace<ArrowUpRight size={13} /></Link>
        </div>
      </aside>
      <div className="app-body">
        <header className="app-topbar">
          <div className="flex min-w-0 items-center gap-2 text-xs sm:gap-3 sm:text-sm">
            <Blocks size={18} className="text-cyan lg:hidden" />
            <span className="hidden text-ink/45 sm:inline">Workspace</span><ChevronRight size={14} className="hidden text-ink/30 sm:block" />
            <span className="truncate font-medium">{current?.label ?? "Workbench"}</span>
          </div>
          <div className="flex shrink-0 items-center gap-3"><span className="hidden rounded-md border border-line bg-panel px-2 py-1 text-[10px] font-semibold tracking-widest text-ink/50 sm:inline">LOCAL</span><TopbarJobsMenu /></div>
        </header>
        <nav aria-label="Mobile navigation" className="mobile-navigation">
          {navigation.map((item) => <Link key={item.href} href={item.href} aria-current={pathname === item.href ? "page" : undefined} className={`mobile-nav-link ${pathname === item.href ? "mobile-nav-link-active" : ""}`}><item.icon size={15} />{item.label}</Link>)}
        </nav>
        <main id="main-content" tabIndex={-1} className="app-main">{children}</main>
        <footer className="app-footer"><span>Knowledge Editing Workbench</span><span>Explore · Understand · Edit</span></footer>
      </div>
    </div>
  );
}
