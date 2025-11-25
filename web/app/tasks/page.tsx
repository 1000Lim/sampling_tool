'use client'

import useSWR from 'swr'
import Link from 'next/link'

const API_BASE = '/api'
const fetcher = (url: string) => fetch(url).then(r => r.json())

export default function Tasks() {
  const { data, isLoading, error, mutate } = useSWR(API_BASE + '/tasks', fetcher, { refreshInterval: 3000 })

  async function cancel(id: string) {
    const confirmed = confirm('Cancel this running task?')
    if (!confirmed) return
    await fetch(API_BASE + '/tasks/' + id + '/cancel', { method: 'POST' })
    mutate()
  }

  async function remove(id: string) {
    const confirmed = confirm('Delete this task?')
    if (!confirmed) return
    await fetch(API_BASE + '/tasks/' + id, { method: 'DELETE' })
    mutate()
  }

  return (
    <main className="mx-auto max-w-6xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Tasks</h1>
        <div className="flex items-center gap-2">
          <Link href="/tasks/create" className="btn btn-primary">Create Task</Link>
        </div>
      </div>

      {isLoading && <div className="text-neutral-400">Loading...</div>}
      {error && <div className="text-red-400">Failed to load</div>}

      <div className="overflow-auto rounded-xl border border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-900/50">
            <tr>
              <th className="px-4 py-3 text-left font-medium text-neutral-400">ID</th>
              <th className="px-4 py-3 text-left font-medium text-neutral-400">Status</th>
              <th className="px-4 py-3 text-left font-medium text-neutral-400">Progress</th>
              <th className="px-4 py-3 text-left font-medium text-neutral-400">Export</th>
              <th className="px-4 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {Array.isArray(data) && data.map((t: any) => {
              let progress = null
              try {
                progress = t.progress ? JSON.parse(t.progress) : null
              } catch (e) {
                // ignore parse errors
              }
              return (
                <tr key={t.id} className="border-t border-neutral-800">
                  <td className="px-4 py-3 text-neutral-200 truncate max-w-[320px]">{t.id}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs ${t.status === 'completed' ? 'bg-green-600/20 text-green-400' : t.status === 'failed' ? 'bg-red-600/20 text-red-400' : t.status === 'cancelled' ? 'bg-neutral-600/20 text-neutral-400' : 'bg-yellow-600/20 text-yellow-400'}`}>{t.status}</span>
                  </td>
                  <td className="px-4 py-3 text-neutral-400">
                    {progress ? (
                      <div className="space-y-1">
                        <div className="text-xs">{progress.stage}: {progress.current}/{progress.total}</div>
                        {progress.message && <div className="text-xs text-neutral-500">{progress.message}</div>}
                        <div className="w-full bg-neutral-800 rounded-full h-1.5">
                          <div className="bg-blue-500 h-1.5 rounded-full transition-all" style={{width: `${Math.min(100, (progress.current / progress.total) * 100)}%`}}></div>
                        </div>
                      </div>
                    ) : t.status === 'running' ? (
                      <span className="text-xs text-neutral-500">Processing...</span>
                    ) : '-'}
                  </td>
                  <td className="px-4 py-3 text-neutral-400 truncate max-w-[360px]">{t.export_dir || '-'}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2 justify-end">
                      {t.status === 'running' && (
                        <button className="btn btn-secondary" onClick={() => cancel(t.id)}>Cancel</button>
                      )}
                      <Link href={`/tasks/${t.id}`} className="btn btn-secondary">View</Link>
                      <button className="btn btn-secondary" onClick={() => remove(t.id)}>Delete</button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </main>
  )
}


