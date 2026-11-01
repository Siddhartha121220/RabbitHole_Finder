export function Loading({ label }: { label: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-3 py-8 text-ink-soft">
      <span className="relative inline-flex h-3 w-3">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-lantern opacity-60" />
        <span className="relative inline-flex h-3 w-3 rounded-full bg-lantern" />
      </span>
      <span className="font-mono text-sm">{label}</span>
    </div>
  );
}

export function ErrorBox({ error, onRetry }: { error: Error; onRetry?: () => void }) {
  return (
    <div role="alert" className="my-6 rounded-md border border-danger/40 bg-danger/5 p-4 text-sm">
      <p className="font-medium text-danger">Something caved in.</p>
      <p className="mt-1 text-ink-soft">{error.message}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-3 font-mono text-xs underline underline-offset-4">
          try again
        </button>
      )}
    </div>
  );
}
