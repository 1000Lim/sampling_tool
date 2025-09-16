'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

const API_BASE = '/api'

type RunRequest = {
  path: string
  data_type: 'surf' | 'valeo'
  stride: number
  cam_num?: number
  export?: string
  compress?: boolean
  delete?: boolean
  cam_info?: string
  overlay_every?: number
  overlay_intensity?: boolean
  overlay_point_radius?: number
  overlay_alpha?: number
}

export default function CreateTask() {
  const router = useRouter()
  const [form, setForm] = useState<RunRequest>({
    path: '', data_type: 'valeo', stride: 1, overlay_every: 0, overlay_point_radius: 2, overlay_alpha: 1
  })
  const [status, setStatus] = useState<string>('')
  const [jobId, setJobId] = useState<string>('')
  const [pickerOpen, setPickerOpen] = useState<null | 'path' | 'export' | 'cam_info'>(null)
  const [browserCwd, setBrowserCwd] = useState<string>('')
  const [browserEntries, setBrowserEntries] = useState<{ name: string; path: string; is_dir: boolean }[]>([])
  const [browserParent, setBrowserParent] = useState<string | null>(null)
  const [dirsOnly, setDirsOnly] = useState<boolean>(true)
  const [roots, setRoots] = useState<{ name: string; path: string; is_dir: boolean }[]>([])
  const [selectedRoot, setSelectedRoot] = useState<string>('')

  const canCreate = useMemo(() => {
    if (!form.path || !form.stride || form.stride < 1) return false
    if (form.data_type === 'surf' && (form.cam_num === undefined || form.cam_num === null || Number.isNaN(form.cam_num))) return false
    if (form.data_type === 'surf' && (!form.cam_info || form.cam_info.trim() === '')) return false
    return true
  }, [form])

  async function submit() {
    setStatus('')
    if (!canCreate) {
      setStatus('Please fill required fields.')
      return
    }
    const res = await fetch(API_BASE + '/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form)
    })
    if (!res.ok) { setStatus('Failed: ' + (await res.text())); return }
    const data = await res.json(); setJobId(data.id); setStatus('running')
    router.push('/tasks')
  }

  async function loadRoots() {
    const res = await fetch(API_BASE + '/fs/roots')
    if (res.ok) {
      const data = await res.json()
      setRoots(data)
      const preferred = data.find((r: any) => r.path === '/mnt') || data[0]
      setSelectedRoot(preferred?.path || '')
      if (preferred) await navTo(preferred.path)
    }
  }

  async function openPicker(which: 'path' | 'export' | 'cam_info') {
    setPickerOpen(which)
    if (which === 'cam_info') setDirsOnly(false); else setDirsOnly(true)
    if (!roots.length) await loadRoots()
    else await navTo(selectedRoot || roots[0]?.path)
  }

  async function navTo(path: string) {
    const res = await fetch(API_BASE + '/fs/entries?path=' + encodeURIComponent(path) + '&only_dirs=' + String(dirsOnly))
    if (res.ok) {
      const data = await res.json()
      setBrowserCwd(data.cwd)
      setBrowserParent(data.parent)
      setBrowserEntries((data.entries || []).filter((e: any) => !(e?.name || '').startsWith('.')))
    }
  }

  function chooseCurrent() {
    if (!pickerOpen) return
    if (pickerOpen === 'path') setForm({ ...form, path: browserCwd })
    if (pickerOpen === 'export') setForm({ ...form, export: browserCwd })
    if (pickerOpen === 'cam_info') setForm({ ...form, cam_info: browserCwd })
    setPickerOpen(null)
  }

  return (
    <main className="mx-auto max-w-5xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Create Task</h1>
        <Link href="/tasks" className="btn btn-secondary">← Back to Tasks</Link>
      </div>

      <section className="card">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label>Path</label>
            <div className="flex gap-2">
              <input className="input flex-1" value={form.path} onChange={e => setForm({ ...form, path: e.target.value })} />
              <button type="button" className="btn btn-secondary" onClick={() => openPicker('path')}>Browse</button>
            </div>
          </div>
          <div>
            <label>Data Type</label>
            <select className="input" value={form.data_type} onChange={e => setForm({ ...form, data_type: e.target.value as any })}>
              <option value="surf">surf</option>
              <option value="valeo">valeo</option>
            </select>
          </div>
          <div>
            <label>Stride</label>
            <input className="input" type="number" min={1} value={form.stride} onChange={e => setForm({ ...form, stride: Number(e.target.value) })} />
          </div>
          {form.data_type === 'surf' && (
            <div>
              <label>Camera number (SURF)</label>
              <input
                className="input"
                type="number"
                value={form.cam_num ?? ''}
                onChange={e => {
                  const v = e.target.value
                  setForm({ ...form, cam_num: v === '' ? undefined : Number(v) })
                }}
              />
            </div>
          )}
          <div className="md:col-span-2">
            <label>Export</label>
            <div className="flex gap-2">
              <input className="input flex-1" value={form.export ?? ''} onChange={e => setForm({ ...form, export: e.target.value })} />
              <button type="button" className="btn btn-secondary" onClick={() => openPicker('export')}>Browse</button>
            </div>
          </div>
          {form.data_type === 'surf' && (
            <div className="md:col-span-2">
              <label>Camera info file (SURF)</label>
              <div className="flex gap-2">
                <input className="input flex-1" value={form.cam_info ?? ''} onChange={e => setForm({ ...form, cam_info: e.target.value })} />
                <button type="button" className="btn btn-secondary" onClick={() => openPicker('cam_info')}>Browse</button>
              </div>
            </div>
          )}
          <div className="flex items-center gap-6">
            <label className="inline-flex items-center gap-2 text-sm"><input type="checkbox" checked={!!form.compress} onChange={e => setForm({ ...form, compress: e.target.checked })} /> Compress</label>
            <label className="inline-flex items-center gap-2 text-sm"><input type="checkbox" checked={!!form.delete} onChange={e => setForm({ ...form, delete: e.target.checked })} /> Delete after compress</label>
          </div>
          <div className="flex items-center gap-6">
            <label className="inline-flex items-center gap-2 text-sm"><input type="checkbox" checked={!!form.overlay_intensity} onChange={e => setForm({ ...form, overlay_intensity: e.target.checked })} /> Overlay Intensity</label>
          </div>
          <div>
            <label>Overlay Every</label>
            <input className="input" type="number" min={0} value={form.overlay_every ?? 0} onChange={e => setForm({ ...form, overlay_every: Number(e.target.value) })} />
          </div>
          <div>
            <label>Overlay Point Radius</label>
            <input className="input" type="number" min={1} value={form.overlay_point_radius ?? 2} onChange={e => setForm({ ...form, overlay_point_radius: Number(e.target.value) })} />
          </div>
          <div>
            <label>Overlay Alpha</label>
            <input className="input" type="number" min={0} max={1} step={0.1} value={form.overlay_alpha ?? 1} onChange={e => setForm({ ...form, overlay_alpha: Number(e.target.value) })} />
          </div>
        </div>
        <div className="mt-6 flex gap-3">
          <button className="btn btn-primary disabled:opacity-60" disabled={!canCreate} onClick={submit}>Create</button>
        </div>
      </section>

      <section className="mt-6 card text-sm text-neutral-400">
        Status: {status || '-'} {jobId && <span>(job: {jobId})</span>}
      </section>

      {pickerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-[min(92vw,1000px)] h-[min(80vh,700px)] card flex flex-col">
            <div className="flex items-center justify-between pb-4 border-b border-neutral-800">
              <div className="flex items-center gap-3">
                <div className="text-lg font-medium">File Browser</div>
                <div className="text-sm text-neutral-500">{pickerOpen === 'path' ? 'Select dataset path' : (pickerOpen === 'export' ? 'Select export directory' : 'Select camera info file')}</div>
              </div>
              <div className="flex items-center gap-3">
                <label className="inline-flex items-center gap-2 text-xs text-neutral-400">
                  <input type="checkbox" className="rounded" checked={dirsOnly} onChange={(e) => { setDirsOnly(e.target.checked); navTo(browserCwd) }} />
                  Directories only
                </label>
                <button className="btn btn-secondary text-xs" onClick={() => setPickerOpen(null)}>Close</button>
              </div>
            </div>

            <div className="py-3 border-b border-neutral-800 flex items-center gap-3">
              <select className="input w-auto" value={selectedRoot} onChange={e => { setSelectedRoot(e.target.value); navTo(e.target.value) }}>
                {roots.map(r => (
                  <option key={r.path} value={r.path}>{r.path}</option>
                ))}
              </select>
              <div className="text-sm text-neutral-400 truncate">{browserCwd}</div>
            </div>

            <div className="py-3 flex items-center justify-between border-b border-neutral-800">
              <div className="flex gap-2">
                <button className="btn btn-secondary" disabled={!browserParent} onClick={() => browserParent && navTo(browserParent!)}>Up</button>
              </div>
              <div className="flex gap-2">
                <button className="btn btn-primary" onClick={chooseCurrent}>Select Current</button>
              </div>
            </div>

            <div className="flex-1 overflow-auto p-2">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                {browserEntries.map((e) => (
                  <button
                    key={e.path}
                    className="flex items-center gap-3 p-3 rounded-lg border border-neutral-800 hover:border-neutral-700 hover:bg-neutral-800/50 transition-all text-left group"
                    onClick={() => {
                      if (e.is_dir) return navTo(e.path)
                      if (!pickerOpen) return
                      if (pickerOpen === 'path') setForm({ ...form, path: e.path })
                      if (pickerOpen === 'export') setForm({ ...form, export: e.path })
                      if (pickerOpen === 'cam_info') setForm({ ...form, cam_info: e.path })
                      setPickerOpen(null)
                    }}
                    title={e.path}
                  >
                    <div className={`h-5 w-5 rounded-sm ${e.is_dir ? 'bg-blue-500/30 border border-blue-400/50' : 'bg-neutral-600/40 border border-neutral-500/50'}`}></div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-sm truncate group-hover:text-white transition-colors">{e.name}</div>
                      <div className="text-xs text-neutral-500">{e.is_dir ? 'Directory' : 'File'}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}

// File picker modal
// Minimalist, no emojis
// Uses roots dropdown to switch between allowed roots (e.g., /mnt)

/* JSX modal appended at bottom of component */