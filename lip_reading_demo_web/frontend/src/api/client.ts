export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

/**
 * Sends a GET request to the FastAPI backend and returns the typed JSON response.
 */
export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`);

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`GET ${path} failed: ${response.status} ${text}`);
  }

  return response.json() as Promise<T>;
}

/**
 * Sends a JSON POST request to the FastAPI backend and returns the typed JSON response.
 */
export async function apiPost<TResponse, TBody = unknown>(
  path: string,
  body?: TBody,
): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`POST ${path} failed: ${response.status} ${text}`);
  }

  return response.json() as Promise<TResponse>;
}

/**
 * Sends a multipart/form-data POST request.
 * The browser must set the Content-Type header because it also adds the boundary.
 */
export async function apiPostFormData<TResponse>(
  path: string,
  formData: FormData,
): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`POST ${path} failed: ${response.status} ${text}`);
  }

  return response.json() as Promise<TResponse>;
}
