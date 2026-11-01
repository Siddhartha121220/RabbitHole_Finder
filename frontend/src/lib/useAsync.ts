"use client";

import { useEffect, useRef, useState } from "react";

export type AsyncState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: Error }
  | { status: "done"; data: T };

type Settled<T> = { key: string; result: { status: "error"; error: Error } | { status: "done"; data: T } };

/** Run `fn` whenever `key` changes (null = don't run); the previous call is aborted.
 *  "loading" is derived: a key whose result has not arrived yet is loading. */
export function useAsync<T>(key: string | null, fn: (signal: AbortSignal) => Promise<T>): AsyncState<T> {
  const [settled, setSettled] = useState<Settled<T> | null>(null);
  const fnRef = useRef(fn);
  useEffect(() => {
    fnRef.current = fn;
  });
  useEffect(() => {
    if (key === null) return;
    const ctrl = new AbortController();
    fnRef.current(ctrl.signal).then(
      (data) => setSettled({ key, result: { status: "done", data } }),
      (error: unknown) => {
        if (ctrl.signal.aborted) return;
        const err = error instanceof Error ? error : new Error(String(error));
        setSettled({ key, result: { status: "error", error: err } });
      },
    );
    return () => ctrl.abort();
  }, [key]);
  if (key === null) return { status: "idle" };
  if (settled?.key !== key) return { status: "loading" };
  return settled.result;
}
