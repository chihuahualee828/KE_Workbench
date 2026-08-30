import type { Metadata } from "next";
import Link from "next/link";
import { TopbarJobsMenu } from "@/components/TopbarJobsMenu";
import "./globals.css";

export const metadata: Metadata = {
  title: "Knowledge Editing Workbench",
  description: "Interactive LLM knowledge editing and interpretability workbench"
};

const nav = [
  { href: "/", label: "Workbench" },
  { href: "/playground", label: "Generate" },
  { href: "/logit-lens", label: "Logit Lens" },
  { href: "/trace", label: "Trace" },
  { href: "/edit", label: "Edit" },
  { href: "/evaluate", label: "Evaluate" },
  { href: "/settings", label: "Settings" }
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="sticky top-0 z-20 border-b border-line bg-panel/95 backdrop-blur">
          <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 md:flex-row md:items-center md:justify-between md:px-8">
            <Link href="/" className="text-base font-semibold text-ink">
              Knowledge Editing Workbench
            </Link>
            <nav className="flex flex-wrap gap-1">
              {nav.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md px-3 py-2 text-sm text-ink/70 transition hover:bg-white hover:text-ink"
                >
                  {item.label}
                </Link>
              ))}
              <TopbarJobsMenu />
            </nav>
          </div>
        </header>
        <main className="mx-auto min-h-screen max-w-7xl">{children}</main>
      </body>
    </html>
  );
}
