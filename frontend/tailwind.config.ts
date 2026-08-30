import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#172026",
        panel: "#f7f8f5",
        line: "#d9ded5",
        moss: "#50694f",
        rust: "#a85632",
        cyan: "#1c7784"
      }
    }
  },
  plugins: []
};

export default config;

