/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        // Inter for UI, JetBrains Mono for ids/code — loaded in index.html
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        // Single accent used consistently — azure, deliberately not stock blue-600
        brand: {
          50: '#eef4ff', 100: '#d9e6ff', 200: '#bcd3ff', 300: '#8eb5ff',
          400: '#598dff', 500: '#3366f6', 600: '#1f49eb', 700: '#1a39d4',
          800: '#1c31ab', 900: '#1d2f87', 950: '#161e50',
        },
        // Ink = header/shell; cool slate neutrals for everything else
        ink: '#0d1326',
      },
      borderRadius: { xl: '0.75rem' },
      boxShadow: {
        card: '0 1px 2px 0 rgb(13 19 38 / 0.04), 0 1px 3px 0 rgb(13 19 38 / 0.06)',
      },
    },
  },
  plugins: [],
};
