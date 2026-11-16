"""Start a local static file server. All neural network inference still runs in browser."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser

PORT=8000
ROOT=Path(__file__).resolve().parent/'web'

def main():
    handler=partial(SimpleHTTPRequestHandler,directory=str(ROOT))
    with ThreadingHTTPServer(('127.0.0.1',PORT),handler) as server:
        url=f'http://127.0.0.1:{PORT}'
        print('MiniVLM available at',url)
        print('Serving static model assets only. Inference runs locally in Chrome/Edge.')
        print('Press Ctrl+C to stop the file server.')
        webbrowser.open(url)
        try:server.serve_forever()
        except KeyboardInterrupt:pass

if __name__=='__main__':main()
