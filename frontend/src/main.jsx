import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Create container for the application
const rootElement = document.getElementById('root');
const root = createRoot(rootElement);

// Render the application
root.render(
  <StrictMode>
    <App />
  </StrictMode>,
);