const CACHE_VERSION = "vs-comics-v1";
const APP_SHELL = [
  "/",               // hilft je nach Server
  "/index.html",
  "/library.html",
  "/viewer.html",
  "/comics/index.json",
  "/manifest.webmanifest"
  // optional: "/assets/intro.mp4" (würde ich erstmal NICHT cachen; kann groß sein)
];

// Install: App-Shell vorcachen
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

// Activate: alte Caches löschen
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => (k === CACHE_VERSION ? null : caches.delete(k))))
    )
  );
  self.clients.claim();
});

// Fetch-Strategie:
// - App-Shell: cache-first
// - JSON/WebP/MP4 etc: stale-while-revalidate (schnell + aktualisiert)
self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // nur gleiche Origin cachen
  if (url.origin !== location.origin) return;

  const isStaticAsset =
    url.pathname.endsWith(".html") ||
    url.pathname.endsWith(".json") ||
    url.pathname.endsWith(".webp") ||
    url.pathname.endsWith(".png") ||
    url.pathname.endsWith(".jpg") ||
    url.pathname.endsWith(".jpeg") ||
    url.pathname.endsWith(".svg") ||
    url.pathname.endsWith(".mp4") ||
    url.pathname.endsWith(".webmanifest");

  if (!isStaticAsset) return;

  event.respondWith(staleWhileRevalidate(req));
});

async function staleWhileRevalidate(request) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(request);

  const networkPromise = fetch(request)
    .then((response) => {
      // nur erfolgreiche GETs cachen
      if (response && response.status === 200) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => null);

  // wenn was im Cache ist: sofort liefern, sonst Netzwerk versuchen
  return cached || (await networkPromise) || cachedFallback();
}

function cachedFallback() {
  // Minimaler Offline-Fallback
  return new Response("Offline – Datei nicht im Cache.", {
    headers: { "Content-Type": "text/plain; charset=utf-8" }
  });
}
