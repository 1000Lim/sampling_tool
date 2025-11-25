'use client'

import { useEffect, useMemo, useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'

const API_BASE = '/api'

type RunRequest = {
  path: string
  data_type: 'surf' | 'valeo' | 'aclm'
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
  convert_raw_to_jpg?: boolean
  raw_output_format?: 'gray' | 'rgb'
  raw_dgain?: number
}

export default function CreateTask() {
  const router = useRouter()
  const [form, setForm] = useState<RunRequest>({
    path: '', data_type: 'valeo', stride: 10, overlay_every: 0, overlay_intensity: true, overlay_point_radius: 2, overlay_alpha: 1,
    convert_raw_to_jpg: false, raw_output_format: 'gray', raw_dgain: 1.5
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
  const [newDirName, setNewDirName] = useState<string>('')
  const [showMkdir, setShowMkdir] = useState<boolean>(false)

  const canCreate = useMemo(() => {
    if (!form.path || !form.stride || form.stride < 1) return false
    if (form.data_type === 'surf' && (form.cam_num === undefined || form.cam_num === null || Number.isNaN(form.cam_num))) return false
    // cam_info is optional for SURF; pipeline will auto-detect if empty
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

      {/* Dataset & Output */}
      <section className="card">
        <div className="mb-3 text-sm text-neutral-400">Dataset & Output</div>
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
              <option value="surf">SURF</option>
              <option value="valeo">Valeo</option>
              <option value="aclm">ACLM</option>
            </select>
          </div>
          <div className="md:col-span-2">
            <label>Export</label>
            <div className="flex gap-2">
              <input className="input flex-1" value={form.export ?? ''} onChange={e => setForm({ ...form, export: e.target.value })} />
              <button type="button" className="btn btn-secondary" onClick={() => openPicker('export')}>Browse</button>
            </div>
          </div>
        </div>
      </section>

      {/* Sampling options */}
      <section className="mt-4 card">
        <div className="mb-3 text-sm text-neutral-400">Sampling</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label>Stride</label>
            <input className="input" type="number" min={1} value={form.stride} onChange={e => setForm({ ...form, stride: Number(e.target.value) })} />
          </div>
          <div>
            <label>Skip first N LiDAR frames</label>
            <input className="input" type="number" min={0} value={(form as any).skip_head ?? 0} onChange={e => setForm({ ...form, ...(form as any), skip_head: Number(e.target.value) })} />
          </div>
          <div>
            <label>Skip last N LiDAR frames</label>
            <input className="input" type="number" min={0} value={(form as any).skip_tail ?? 0} onChange={e => setForm({ ...form, ...(form as any), skip_tail: Number(e.target.value) })} />
          </div>
        </div>
      </section>

      {/* SURF options */}
      {form.data_type === 'surf' && (
        <section className="mt-4 card">
          <div className="mb-3 text-sm text-neutral-400">SURF Options</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label>Camera number</label>
              <input className="input" type="number" value={form.cam_num ?? ''} onChange={e => {
                const v = e.target.value; setForm({ ...form, cam_num: v === '' ? undefined : Number(v) })
              }} />
            </div>
            <div className="md:col-span-2">
              <label>Camera info file (optional, auto-detect if empty)</label>
              <div className="flex gap-2">
                <input className="input flex-1" placeholder="Leave empty to auto-detect from raw" value={form.cam_info ?? ''} onChange={e => setForm({ ...form, cam_info: e.target.value })} />
                <button type="button" className="btn btn-secondary" onClick={() => openPicker('cam_info')}>Browse</button>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ACLM Raw Conversion Options */}
      {form.data_type === 'aclm' && (
        <section className="mt-4 card">
          <div className="mb-3 text-sm text-neutral-400">ACLM Raw Conversion</div>
          <div className="flex items-center gap-3 mb-4">
            <label className="inline-flex items-center gap-2 text-sm">
              <input type="checkbox" checked={!!form.convert_raw_to_jpg} onChange={e => setForm({ ...form, convert_raw_to_jpg: e.target.checked })} />
              Convert .raw to .jpg
            </label>
            <span className="text-xs text-neutral-500">(Raw files need ISP processing for visualization)</span>
          </div>

          {form.convert_raw_to_jpg && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label>Output Format</label>
                <select className="input" value={form.raw_output_format ?? 'gray'} onChange={e => setForm({ ...form, raw_output_format: e.target.value as 'gray' | 'rgb' })}>
                  <option value="gray">Grayscale</option>
                  <option value="rgb">RGB</option>
                </select>
              </div>
              <div>
                <label>Digital Gain (RGB only)</label>
                <input className="input" type="number" min={0.1} max={10} step={0.1} value={form.raw_dgain ?? 1.5} onChange={e => setForm({ ...form, raw_dgain: Number(e.target.value) })} disabled={form.raw_output_format === 'gray'} />
              </div>
            </div>
          )}
        </section>
      )}

      {/* Overlay options - Show for SURF/Valeo OR ACLM with conversion enabled */}
      {(form.data_type !== 'aclm' || (form.data_type === 'aclm' && form.convert_raw_to_jpg)) && (
        <section className="mt-4 card">
          <div className="mb-3 text-sm text-neutral-400">Overlay</div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="md:col-span-1 flex items-center">
              <label className="inline-flex items-center gap-2 text-sm"><input type="checkbox" checked={!!form.overlay_intensity} onChange={e => setForm({ ...form, overlay_intensity: e.target.checked })} /> Intensity coloring</label>
            </div>
            <div>
              <label>Every Nth pair (0=off)</label>
              <input className="input" type="number" min={0} value={form.overlay_every ?? 0} onChange={e => setForm({ ...form, overlay_every: Number(e.target.value) })} />
            </div>
            <div>
              <label>Point radius</label>
              <input className="input" type="number" min={1} value={form.overlay_point_radius ?? 2} onChange={e => setForm({ ...form, overlay_point_radius: Number(e.target.value) })} />
            </div>
            <div>
              <label>Alpha</label>
              <input className="input" type="number" min={0} max={1} step={0.1} value={form.overlay_alpha ?? 1} onChange={e => setForm({ ...form, overlay_alpha: Number(e.target.value) })} />
            </div>
          </div>
        </section>
      )}

      {/* Packaging */}
      <section className="mt-4 card">
        <div className="mb-3 text-sm text-neutral-400">Packaging</div>
        <div className="flex items-center gap-6">
          <label className="inline-flex items-center gap-2 text-sm"><input type="checkbox" checked={!!form.compress} onChange={e => setForm({ ...form, compress: e.target.checked })} /> Compress</label>
          <label className="inline-flex items-center gap-2 text-sm"><input type="checkbox" checked={!!form.delete} onChange={e => setForm({ ...form, delete: e.target.checked })} /> Delete after compress</label>
        </div>
      </section>

      <div className="mt-6 flex gap-3">
        <button className="btn btn-primary disabled:opacity-60" disabled={!canCreate} onClick={submit}>Create</button>
      </div>

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

            <div className="py-3 border-b border-neutral-800 flex flex-col gap-2">
              <div className="flex items-center gap-3">
                <label className="text-xs text-neutral-500 shrink-0">Root:</label>
                <select className="input w-auto flex-1" value={selectedRoot} onChange={e => { setSelectedRoot(e.target.value); navTo(e.target.value) }}>
                  {roots.map(r => (
                    <option key={r.path} value={r.path}>{r.path}</option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-3">
                <label className="text-xs text-neutral-500 shrink-0">Current:</label>
                <div className="text-sm text-neutral-300 font-mono break-all">{browserCwd}</div>
              </div>
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