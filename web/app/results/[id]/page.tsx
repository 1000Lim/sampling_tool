'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'

const API_BASE = '/api'

function join(...parts: string[]) {
  return parts.join('/').replace(/\/+/g, '/').replace(':/', '://')
}

export default function TaskDetail({ params }: { params: { id: string } }) {
  const sp = useSearchParams()
  const exportDir = sp.get('export') || ''
  const [samples, setSamples] = useState<{ left: string; right?: string; title: string }[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [page, setPage] = useState<number>(1)
  const [pageSize, setPageSize] = useState<number>(8)

  useEffect(() => {
    async function load() {
      if (!exportDir) { setSamples([]); setLoading(false); return }
      setLoading(true)
      // List exportDir, detect images and overlay dirs/files dynamically
      async function list(path: string, onlyDirs: boolean) {
        const r = await fetch(`${API_BASE}/fs/entries?path=${encodeURIComponent(path)}&only_dirs=${onlyDirs}`)
        return r.ok ? (await r.json()).entries || [] : []
      }
      const dirs = (await list(exportDir, true)).filter((d: any) => !d.name.startsWith('.'))
      const prefImgs = ['images', 'image', 'img', 'camera', 'cams', 'cam']
      const prefOvls = ['overlays', 'overlay', 'webp', 'overlay_webp']
      let imagesDir = dirs.find((d: any) => prefImgs.includes(d.name))?.path
      let overlaysDir = dirs.find((d: any) => prefOvls.includes(d.name))?.path
      // fallback: choose dir that has many jpg/png
      if (!imagesDir) {
        let best: { path: string; count: number } | null = null
        for (const d of dirs) {
          const files = await list(d.path, false)
          const count = files.filter((f: any) => /\.(jpe?g|png)$/i.test(f.name)).length
          if (count > 3 && (!best || count > best.count)) best = { path: d.path, count }
        }
        imagesDir = best?.path
      }
      if (!overlaysDir) {
        for (const d of dirs) {
          const files = await list(d.path, false)
          if (files.some((f: any) => /\.webp$/i.test(f.name))) { overlaysDir = d.path; break }
        }
      }
      const built: { left: string; right?: string; title: string }[] = []
      if (imagesDir) {
        const files = await list(imagesDir, false)
        // Build set of overlay basenames to ensure existence
        let overlaySet: Set<string> = new Set()
        if (overlaysDir) {
          const ofiles = await list(overlaysDir, false)
          overlaySet = new Set(
            ofiles
              .filter((f: any) => /\.webp$/i.test(f.name))
              .map((f: any) => f.name.replace(/\.[^.]+$/, ''))
          )
        }
        for (const f of files) {
          if (!/\.(jpe?g|png)$/i.test(f.name)) continue
          const base = f.name.replace(/\.[^.]+$/, '')
          // Only include if overlay exists
          if (!overlaysDir || !overlaySet.has(base)) continue
          const left = `${API_BASE}/files?path=${encodeURIComponent(join(imagesDir, f.name))}`
          const right = `${API_BASE}/files?path=${encodeURIComponent(join(overlaysDir, base + '.webp'))}`
          built.push({ left, right, title: base })
        }
      }
      setSamples(built)
      setLoading(false)
    }
    load()
  }, [exportDir])

  useEffect(() => {
    // Reset to first page when dataset changes
    setPage(1)
  }, [exportDir])

  const total = samples.length
  const totalPages = Math.max(1, Math.ceil(total / pageSize))
  const start = (page - 1) * pageSize
  const end = start + pageSize
  const pageItems = samples.slice(start, end)

  return (
    <main className="mx-auto max-w-6xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Task {params.id}</h1>
        <Link className="btn btn-secondary" href="/results">← Back to Tasks</Link>
      </div>

      {!exportDir && (
        <div className="card">Export directory missing.</div>
      )}

      {loading && <div className="text-neutral-400">Scanning export directory...</div>}

      <div className="mb-4 flex items-center justify-between text-sm">
        <div className="text-neutral-400">{total.toLocaleString()} pairs</div>
        <div className="flex items-center gap-2">
          <button className="btn btn-secondary" disabled={page <= 1} onClick={() => setPage(p => Math.max(1, p - 1))}>Prev</button>
          <span className="text-neutral-400">Page {page} / {totalPages}</span>
          <button className="btn btn-secondary" disabled={page >= totalPages} onClick={() => setPage(p => Math.min(totalPages, p + 1))}>Next</button>
          <select className="input w-auto" value={pageSize} onChange={e => { setPageSize(Number(e.target.value)); setPage(1) }}>
            {[4,8,12,16,24,32].map(n => <option key={n} value={n}>{n}/page</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {pageItems.map((p) => (
          <div key={p.title} className="card">
            <div className="mb-3 text-sm text-neutral-400">{p.title}</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-start">
              <div>
                <div className="mb-2 text-xs text-neutral-400">Sampled</div>
                <img className="w-full rounded-lg border border-neutral-800" src={p.left} alt={p.title + ' left'} />
              </div>
              <div>
                <div className="mb-2 text-xs text-neutral-400">Overlay</div>
                {p.right ? (
                  <img className="w-full rounded-lg border border-neutral-800 opacity-100" src={p.right} onError={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = '0.3' }} alt={p.title + ' right'} />
                ) : (
                  <div className="text-xs text-neutral-500">No overlay file</div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {!loading && totalPages > 1 && (
        <div className="mt-6 flex items-center justify-center gap-2">
          <button className="btn btn-secondary" disabled={page <= 1} onClick={() => setPage(p => Math.max(1, p - 1))}>Prev</button>
          <span className="text-neutral-400 text-sm">Page {page} / {totalPages}</span>
          <button className="btn btn-secondary" disabled={page >= totalPages} onClick={() => setPage(p => Math.min(totalPages, p + 1))}>Next</button>
        </div>
      )}
    </main>
  )
}


