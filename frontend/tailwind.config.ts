import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#202939",
        panel: "#f6f7fb",
        line: "#e3e7ef",
        moss: "rgb(var(--accent-hover) / <alpha-value>)",
        rust: "#b45332",
        cyan: "rgb(var(--accent) / <alpha-value>)"
      }
    }
  },
  plugins: []
};

export default config;
