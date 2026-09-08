import type { Metadata } from "next";
import { AccentThemeProvider } from "@/components/AccentTheme";
import { AppShell } from "@/components/AppShell";
import { WorkbenchDataProvider } from "@/components/DataState";
import { WorkbenchSessionProvider } from "@/components/WorkbenchSession";
import "./globals.css";

export const metadata: Metadata = {
  title: "Knowledge Editing Workbench",
  description: "Interactive LLM knowledge editing and interpretability workbench"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AccentThemeProvider>
          <WorkbenchDataProvider>
            <WorkbenchSessionProvider>
              <AppShell>{children}</AppShell>
            </WorkbenchSessionProvider>
          </WorkbenchDataProvider>
        </AccentThemeProvider>
      </body>
    </html>
  );
}
