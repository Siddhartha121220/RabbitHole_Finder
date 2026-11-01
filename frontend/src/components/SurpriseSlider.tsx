export function SurpriseSlider({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <span className="flex items-baseline justify-between">
        <span className="font-mono text-xs uppercase tracking-widest text-ink-soft">Surprise</span>
        <span className="font-mono text-xs text-ink-soft">{value.toFixed(2)}</span>
      </span>
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full accent-rust"
        aria-valuetext={value < 0.34 ? "safe, next door" : value < 0.67 ? "a little further" : "weird and wonderful"}
      />
      <span className="mt-1 flex justify-between text-xs text-ink-soft">
        <span>next door</span>
        <span>weird &amp; wonderful</span>
      </span>
    </label>
  );
}
