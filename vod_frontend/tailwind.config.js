/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        c9: {
          cyan:   '#4dd9e8',   // Primary teal/cyan accent
          blue:   '#5bbfe8',   // Secondary blue
          bg:     '#f0f8ff',   // Page background
          card:   '#ffffff',   // Card/panel background
          blob1:  '#a8e8f2',   // Aqua blob
          blob2:  '#ddbece',   // Dusty rose blob
          text:   '#1a2a3a',   // Primary dark text
          muted:  '#6b8ca8',   // Secondary muted text
          border: '#4dd9e8',   // Teal border
        },
      },
    },
  },
  plugins: [],
}
