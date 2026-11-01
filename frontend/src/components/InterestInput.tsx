"use client";

import { useEffect, useId, useState } from "react";
import { api, type ArticleHit } from "@/lib/api";
import { cleanTitle } from "@/lib/format";

/** Up to 5 interests as chips; typing suggests matching articles from /seeds/search. */
export function InterestInput({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const [draft, setDraft] = useState("");
  const [hits, setHits] = useState<ArticleHit[]>([]);
  const [active, setActive] = useState(-1);
  const listId = useId();
  const full = value.length >= 5;

  useEffect(() => {
    const q = draft.trim();
    if (q.length < 2) return;
    const ctrl = new AbortController();
    const t = setTimeout(() => {
      api.seeds(q, ctrl.signal).then(setHits, () => setHits([]));
    }, 200);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
  }, [draft]);

  const suggestions = draft.trim().length >= 2 ? hits : [];

  function add(text: string) {
    const t = text.trim();
    if (!t || full || value.some((v) => v.toLowerCase() === t.toLowerCase())) return;
    onChange([...value, t]);
    setDraft("");
    setHits([]);
    setActive(-1);
  }

  return (
    <div>
      <label htmlFor={`${listId}-input`} className="font-mono text-xs uppercase tracking-widest text-ink-soft">
        Things you already love <span className="normal-case tracking-normal">(1–5)</span>
      </label>
      <div className="mt-2 flex flex-wrap items-center gap-2 rounded-md border border-line bg-card p-2 focus-within:border-rust">
        {value.map((v) => (
          <span key={v} className="flex items-center gap-1 rounded-full bg-moss/15 py-1 pl-3 pr-1 text-sm text-moss">
            {v}
            <button
              type="button"
              onClick={() => onChange(value.filter((x) => x !== v))}
              className="rounded-full px-1.5 hover:bg-moss/20"
              aria-label={`Remove ${v}`}
            >
              ×
            </button>
          </span>
        ))}
        <input
          id={`${listId}-input`}
          role="combobox"
          aria-expanded={suggestions.length > 0}
          aria-controls={listId}
          aria-activedescendant={active >= 0 ? `${listId}-${active}` : undefined}
          value={draft}
          disabled={full}
          onChange={(e) => {
            setDraft(e.target.value);
            setActive(-1);
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setActive((a) => Math.min(a + 1, suggestions.length - 1));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((a) => Math.max(a - 1, -1));
            } else if (e.key === "Enter" || e.key === ",") {
              if (draft.trim()) {
                e.preventDefault();
                const pick = suggestions[active];
                add(pick ? cleanTitle(pick.title) : draft);
              }
            } else if (e.key === "Backspace" && !draft && value.length) {
              onChange(value.slice(0, -1));
            }
          }}
          placeholder={full ? "that's five" : value.length ? "add another…" : "mechanical keyboards, Byzantine Empire, chess…"}
          className="min-w-[12rem] flex-1 bg-transparent px-1 py-1 outline-none placeholder:text-ink-soft/70"
        />
      </div>
      {suggestions.length > 0 && (
        <ul id={listId} role="listbox" className="mt-1 overflow-hidden rounded-md border border-line bg-card text-sm shadow-lg">
          {suggestions.map((h, i) => (
            <li
              key={h.id}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => {
                e.preventDefault();
                add(cleanTitle(h.title));
              }}
              className={`flex cursor-pointer justify-between px-3 py-2 ${i === active ? "bg-line/50" : "hover:bg-line/30"}`}
            >
              <span>{cleanTitle(h.title)}</span>
              <span className="font-mono text-xs text-ink-soft">{h.similarity.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
