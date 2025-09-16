'use client'

import useSWR from 'swr'
import Link from 'next/link'

const API_BASE = '/api'
const fetcher = (url: string) => fetch(url).then(r => r.json())

export default function ResultsPage() {
  const { data, isLoading, error, mutate } = useSWR(API_BASE + '/tasks', fetcher, { refreshInterval: 5000 })

  async function remove(id: string) {
    await fetch(API_BASE + '/tasks/' + id, { method: 'DELETE' })
    mutate()
  }

  return (
    <main className="mx-auto max-w-6xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Tasks</h1>
        <Link className="btn btn-secondary" href="/">← Back</Link>
      </div>

      {isLoading && <div className="text-neutral-400">Loading...</div>}
      {error && <div className="text-red-400">Failed to load</div>}

      <div className="grid grid-cols-1 gap-3">
        {Array.isArray(data) && data.map((t: any) => (
          <div key={t.id} className="card">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`px-2 py-1 rounded text-xs ${t.status === 'completed' ? 'bg-green-600/20 text-green-400' : t.status === 'failed' ? 'bg-red-600/20 text-red-400' : 'bg-yellow-600/20 text-yellow-400'}`}>{t.status}</div>
                <div className="text-sm text-neutral-400">{t.id}</div>
              </div>
              <div className="flex gap-2">
                <button className="btn btn-secondary" onClick={() => remove(t.id)}>Delete</button>
                {t.status === 'completed' && t.export_dir ? (
                  <Link className="btn btn-primary" href={`/results/${encodeURIComponent(t.id)}?export=${encodeURIComponent(t.export_dir)}`}>View</Link>
                ) : (
                  <button className="btn btn-secondary" disabled title="Not ready">View</button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </main>
  )
}


