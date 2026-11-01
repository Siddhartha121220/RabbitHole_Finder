// Anonymous id for feedback: random, per browser, never tied to a person.
const KEY = "rhf-session";

export function sessionId(): string {
  try {
    let id = window.localStorage.getItem(KEY);
    if (!id) {
      id = crypto.randomUUID();
      window.localStorage.setItem(KEY, id);
    }
    return id;
  } catch {
    return "anonymous";
  }
}
