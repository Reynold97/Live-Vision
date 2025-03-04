// frontend/src/config.js

// Configuration variables
const config = {
    // API base URL - change this for different environments
    apiBaseUrl: 'http://localhost:8000',
    
    // WebSocket base URL - defaults to same host as API
    get wsBaseUrl() {
      // Use secure WebSocket (wss://) if using HTTPS, otherwise use ws://
      const protocol = this.apiBaseUrl.startsWith('https') ? 'wss' : 'ws';
      
      // Extract domain and port from API URL
      let domain = this.apiBaseUrl.replace(/^https?:\/\//, '');
      
      // Return full WebSocket URL
      return `${protocol}://${domain}`;
    }
  };
  
  export default config;