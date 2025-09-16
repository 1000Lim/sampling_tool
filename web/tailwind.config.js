/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef6ff',
          100: '#dbebff',
          200: '#b7d7ff',
          300: '#8fbfff',
          400: '#62a2ff',
          500: '#3b86ff',
          600: '#2468e6',
          700: '#1a50b3',
          800: '#163f8a',
          900: '#142f66'
        }
      }
    }
  },
  plugins: []
}


