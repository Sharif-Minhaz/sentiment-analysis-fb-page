/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js}'],
  theme: {
    extend: {
      // Preflight's fallback border color is gray-200 (#e5e7eb), which glares on
      // the dark surface. Any border or divide-* that doesn't set its own color
      // gets the theme-aware hairline instead.
      borderColor: { DEFAULT: 'var(--hairline)' },
    },
  },
  plugins: [],
}
