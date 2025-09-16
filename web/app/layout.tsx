import './globals.css'
export const metadata = {
  title: 'Sampling Web',
  description: 'Frontend for Sampling Tool API'
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body suppressHydrationWarning className="min-h-screen bg-neutral-950 text-neutral-100">
        {children}
      </body>
    </html>
  )
}


