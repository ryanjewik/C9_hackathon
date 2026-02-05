/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        valorant: {
          red: '#ff4655',
          dark: '#0f1923',
          gray: '#1a242d',
          teal: '#00c8c8',
          orange: '#ff9b00',
        },
      },
    },
  },
  plugins: [],
}
