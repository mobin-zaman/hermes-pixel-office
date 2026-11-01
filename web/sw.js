/* Hermes Pixel Office PWA service worker.
 * Live monitor: network-first for the shell, never cache /state.
 * Only caches static assets so the app opens offline as a shell. */
const RUNTIME = 'pixel-office-v1';
const SHELL = ['/', '/manifest.json', '/icon-512.png', '/icon-maskable-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(RUNTIME).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== RUNTIME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  // Never cache the live state API — always hit network.
  if (url.pathname === '/state' || e.request.method !== 'GET') return;
  // Network-first for the shell; fall back to cache when offline.
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(RUNTIME).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(() => caches.match(e.request))
  );
});
