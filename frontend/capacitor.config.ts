import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.bridgemind.app',
  appName: 'BridgeMind',
  webDir: 'dist',
  backgroundColor: '#f5f5f0',
  // Bundled assets only. Never point an installed build at a Vite dev server.
  // Retain the LAN model gateway in both variants. Native release updates require HTTPS.
  server: { androidScheme: 'http' },
  android: { backgroundColor: '#f5f5f0' },
  plugins: { SystemBars: { insetsHandling: 'css' } },
};

export default config;
