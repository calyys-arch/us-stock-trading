import { useEffect, useRef, useState } from 'react'

/**
 * Polls `fetcher` every `intervalMs` and returns the latest result.
 * Simple polling (not a WebSocket) is deliberate for the MVP dashboard —
 * this system's two strategies operate on daily bars / periodic pair scans,
 * not sub-second tick streams, so a 2s poll is more than fast enough and
 * avoids the added complexity of a WS reconnect/backoff state machine for
 * a display-only surface.
 */
export function usePolling<T>(fetcher: () => Promise<T>, intervalMs = 2000): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  useEffect(() => {
    let cancelled = false

    async function tick() {
      try {
        const result = await fetcherRef.current()
        if (!cancelled) {
          setData(result)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err))
        }
      }
    }

    tick()
    const id = setInterval(tick, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [intervalMs])

  return { data, error }
}
