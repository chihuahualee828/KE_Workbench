import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#202939",
        panel: "#f6f7fb",
        line: "#e3e7ef",
        moss: "#4338ca",
        rust: "#b45332",
        cyan: "#5b57d6"
      }
    }
  },
  plugins: []
};

export default config;
