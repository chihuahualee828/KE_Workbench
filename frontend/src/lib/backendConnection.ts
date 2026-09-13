export type BackendConnection = {
  baseUrl: string;
  token: string;
};

const storageKey = "kew:backend-connection";
const defaultBaseUrl = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000").replace(/\/+$/, "");

type StoredBackendConnection = {
  baseUrl?: string;
  token?: string;
};

export function getDefaultBackendUrl() {
  return defaultBaseUrl;
}

export function normalizeBackendUrl(value: string) {
  const candidate = value.trim();
  if (!candidate) return defaultBaseUrl;

  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error("Backend URL must be a complete http:// or https:// URL.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("Backend URL must use http:// or https://.");
  }
  if (parsed.search || parsed.hash) {
    throw new Error("Backend URL cannot include a query string or hash.");
  }
  return `${parsed.origin}${parsed.pathname}`.replace(/\/+$/, "");
}

export function getBackendConnection(): BackendConnection {
  if (typeof window === "undefined") return { baseUrl: defaultBaseUrl, token: "" };

  try {
    const raw = window.localStorage.getItem(storageKey);
    if (!raw) return { baseUrl: defaultBaseUrl, token: "" };
    const stored = JSON.parse(raw) as StoredBackendConnection;
    return {
      baseUrl: normalizeBackendUrl(stored.baseUrl ?? ""),
      token: typeof stored.token === "string" ? stored.token : ""
    };
  } catch {
    return { baseUrl: defaultBaseUrl, token: "" };
  }
}

export function saveBackendConnection(connection: BackendConnection): BackendConnection {
  const normalized = {
    baseUrl: normalizeBackendUrl(connection.baseUrl),
    token: connection.token.trim()
  };
  if (typeof window === "undefined") return normalized;

  const usesDefaultWithoutToken = normalized.baseUrl === defaultBaseUrl && !normalized.token;
  if (usesDefaultWithoutToken) {
    window.localStorage.removeItem(storageKey);
  } else {
    window.localStorage.setItem(storageKey, JSON.stringify(normalized));
  }
  return normalized;
}

export function backendUrl(path: string) {
  const { baseUrl } = getBackendConnection();
  return `${baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
}

export function backendAuthorizationHeaders() {
  const { token } = getBackendConnection();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
