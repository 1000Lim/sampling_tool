'use client'

import useSWR from 'swr'
import Link from 'next/link'

const API_BASE = '/api'
const fetcher = (url: string) => fetch(url).then(r => r.json())

export default function TaskView({ params }: { params: { id: string } }) {
  const { data, isLoading, error } = useSWR(API_BASE + '/jobs/' + params.id, fetcher, { refreshInterval: 2000 })

  return (
    <main className="mx-auto max-w-5xl p-6">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Task {params.id}</h1>
        <div className="flex items-center gap-2">
          <Link href="/tasks" className="btn btn-secondary">← Back</Link>
        </div>
      </div>

      {isLoading && <div className="text-neutral-400">Loading...</div>}
      {error && <div className="text-red-400">Failed to load</div>}

      <div className="card">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div><span className="text-neutral-400">Status:</span> {data?.status || '-'}</div>
          <div><span className="text-neutral-400">Export:</span> {data?.export || '-'}</div>
          <div><span className="text-neutral-400">Started:</span> {data?.started_at ? new Date(data.started_at * 1000).toLocaleString() : '-'}</div>
          <div><span className="text-neutral-400">Finished:</span> {data?.finished_at ? new Date(data.finished_at * 1000).toLocaleString() : '-'}</div>
          {data?.error && <div className="col-span-2 text-red-400">{data.error}</div>}
        </div>
      </div>

      {data?.status === 'completed' && data?.export && (
        <section className="mt-6">
          <h2 className="mb-3 text-lg font-medium">Results</h2>
          <iframe
            src={`/results/${params.id}?export=${encodeURIComponent(data.export)}`}
            className="w-full h-[70vh] rounded-xl border border-neutral-800 bg-neutral-950"
          />
        </section>
      )}
    </main>
  )
}


