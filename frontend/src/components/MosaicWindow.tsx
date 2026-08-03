import type { ReactNode } from 'react'

interface Props {
  title: string
  children: ReactNode
  className?: string
  actions?: ReactNode
}

/**
 * Generic IBKR TWS "Mosaic"-style tiled window: dark title bar with an
 * uppercase label + decorative window-control dots, and a scrollable
 * content area below. Purely a visual/layout wrapper — no drag/resize
 * behavior — every panel keeps its existing data and functionality.
 */
export default function MosaicWindow({ title, children, className, actions }: Props) {
  return (
    <div className={`mosaic-window ${className ?? ''}`}>
      <div className="mosaic-titlebar">
        <span className="mosaic-title">{title}</span>
        <div className="mosaic-titlebar-right">
          {actions}
          <div className="mosaic-dots">
            <span className="mosaic-dot" />
            <span className="mosaic-dot" />
            <span className="mosaic-dot" />
          </div>
        </div>
      </div>
      <div className="mosaic-content">{children}</div>
    </div>
  )
}
