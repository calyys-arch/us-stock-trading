interface Props {
  running: boolean
  mode: 'observe' | 'auto'
  onStart: () => void
  onStop: () => void
}

export default function Header({ running, mode, onStart, onStop }: Props) {
  const pillClass = !running ? 'stopped' : mode === 'auto' ? 'auto' : 'observe'
  const pillLabel = !running ? 'STOPPED' : mode === 'auto' ? 'AUTO-EXECUTE' : 'OBSERVE ONLY'

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '12px 20px', borderBottom: '1px solid var(--border)',
      background: 'var(--panel)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <strong style={{ fontSize: 16 }}>US Equity Quant Trading</strong>
        <span className={`pill ${pillClass}`}>{pillLabel}</span>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        {!running ? (
          <button onClick={onStart} style={btnStyle('var(--accent)')}>Start (paper)</button>
        ) : (
          <button onClick={onStop} style={btnStyle('var(--red)')}>Stop</button>
        )}
      </div>
    </div>
  )
}

function btnStyle(color: string): React.CSSProperties {
  return {
    background: 'transparent', border: `1px solid ${color}`, color,
    borderRadius: 6, padding: '6px 14px', fontSize: 13, fontWeight: 600,
  }
}
