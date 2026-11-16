/* MiniVLM offline cache: model, scripts, and UI are local static resources. */
const CACHE='minivlm-browser-v1';
const FILES=['./','./index.html','./styles.css','./app.js','./model.js','./assets/model.json','./assets/model.bin'];
self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(FILES)).then(()=>self.skipWaiting()));
});
self.addEventListener('activate', event => {
  event.waitUntil(Promise.all([caches.keys().then(names => Promise.all(names.filter(n=>n!==CACHE).map(n=>caches.delete(n)))),self.clients.claim()]));
});
self.addEventListener('fetch', event => {
  if(event.request.method!=='GET' || new URL(event.request.url).origin!==self.location.origin)return;
  event.respondWith(caches.match(event.request).then(cached => cached || fetch(event.request).then(response => {
    if(response.ok){const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(event.request,copy));}
    return response;
  })));
});
