"""
Interfaz web local del inventario con drone.

    python interfaz.py            (o doble clic en iniciar_interfaz.bat)

Abre el navegador en http://127.0.0.1:8765 con tres pasos:
  1. Configuración: carpeta de los videos, archivo de lotes y parámetros de procesamiento.
  2. Videos: lista de los MP4 de la carpeta (con o sin .SRT, ya procesados o no) y botón para procesarlos.
  3. Resultados: cada vuelo procesado, con su mapa.
Solo escucha en esta computadora (127.0.0.1).
"""
import collections
import contextlib
import csv
import datetime as dt
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import procesar_vuelo as pv

PUERTO = 8765
BASE = pv.BASE
SALIDA = BASE / "salida"
CONFIG = BASE / "config.json"
EXT_VIDEO = {".mp4", ".mov"}

DEFAULTS = {
    "carpeta_videos": "",
    "lotes": str(BASE / "lotes.geojson") if (BASE / "lotes.geojson").exists() else "",
    "camara": "principal",          # principal | tele | personalizada
    "hfov": pv.HFOV_PRINCIPAL,
    "cada_m": 15.0,
    "alt_min": 10.0,
    "conf": 0.2,
    "yaw_modo": "auto",             # auto | fijo
    "yaw": 0.0,
}
HFOV_CAMARAS = {"principal": pv.HFOV_PRINCIPAL, "tele": 27.8}


def leer_config():
    try:
        return {**DEFAULTS, **json.loads(CONFIG.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def guardar_config(nueva):
    c = leer_config()
    for k, v in nueva.items():
        if k not in DEFAULTS:
            continue
        c[k] = float(v) if isinstance(DEFAULTS[k], float) else str(v)
    CONFIG.write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    return c


# ───────────────────────── Procesamiento en segundo plano ─────────────────────────

class Trabajo:
    def __init__(self):
        self.lock = threading.Lock()
        self.log = collections.deque(maxlen=400)
        self.ocupado = False
        self.actual = None
        self.hechos = 0
        self.total = 0
        self.errores = []

    def write(self, texto):  # destino de print() mientras se procesa
        with self.lock:
            for linea in texto.splitlines():
                if linea.strip():
                    self.log.append(linea)
        return len(texto)

    def flush(self):
        pass

    def estado(self):
        with self.lock:
            return {"ocupado": self.ocupado, "actual": self.actual, "hechos": self.hechos,
                    "total": self.total, "errores": self.errores, "log": list(self.log)[-80:]}

    def iniciar(self, videos):
        with self.lock:
            if self.ocupado:
                return False
            self.ocupado, self.hechos, self.total, self.errores = True, 0, len(videos), []
            self.log.clear()
        threading.Thread(target=self._correr, args=(videos, leer_config()), daemon=True).start()
        return True

    def _correr(self, videos, c):
        hfov = HFOV_CAMARAS.get(c["camara"], c["hfov"])
        yaw = c["yaw"] if c["yaw_modo"] == "fijo" else None
        with contextlib.redirect_stdout(self):
            for ruta in videos:
                with self.lock:
                    self.actual = Path(ruta).name
                print(f"── {Path(ruta).name} ──")
                try:
                    pv.video(ruta, lotes_path=c["lotes"] or None, conf=c["conf"], cada_m=c["cada_m"],
                             alt_min=c["alt_min"], hfov=hfov, yaw_fijo=yaw)
                except SystemExit as e:
                    print(f"✖ {e}")
                    self.errores.append(f"{Path(ruta).name}: {e}")
                except Exception as e:  # noqa: BLE001 — se muestra en pantalla en vez de cortar el servidor
                    print(f"✖ Error inesperado: {e!r}")
                    self.errores.append(f"{Path(ruta).name}: {e!r}")
                with self.lock:
                    self.hechos += 1
        with self.lock:
            self.ocupado, self.actual = False, None


TRABAJO = Trabajo()


# ───────────────────────── Datos para la interfaz ─────────────────────────

def listar_videos():
    carpeta = Path(leer_config()["carpeta_videos"] or ".")
    if not leer_config()["carpeta_videos"] or not carpeta.is_dir():
        return {"carpeta_ok": False, "videos": []}
    videos = []
    for p in sorted(carpeta.rglob("*")):
        if p.suffix.lower() not in EXT_VIDEO:
            continue
        srt = any(p.with_suffix(s).exists() for s in (".SRT", ".srt"))
        salida = SALIDA / f"video_{p.stem}"
        videos.append({"ruta": str(p), "nombre": p.name, "carpeta": str(p.parent.relative_to(carpeta)),
                       "mb": round(p.stat().st_size / 1e6), "srt": srt,
                       "fecha": dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%d/%m/%Y %H:%M"),
                       "procesado": (salida / "mapa.html").exists(), "resultado": salida.name})
    return {"carpeta_ok": True, "videos": videos}


def listar_resultados():
    res = []
    if not SALIDA.is_dir():
        return res
    for d in SALIDA.iterdir():
        if not (d / "mapa.html").exists():
            continue
        n_autos = n_tags = 0
        try:
            with open(d / "autos.csv", encoding="utf-8") as f:
                for fila in csv.DictReader(f):
                    n_autos += 1
                    n_tags += bool(fila.get("tag"))
        except OSError:
            pass
        mt = (d / "mapa.html").stat().st_mtime
        res.append({"nombre": d.name, "tipo": "video" if d.name.startswith("video_") else "fotos",
                    "autos": n_autos, "tags": n_tags, "orden": mt,
                    "fecha": dt.datetime.fromtimestamp(mt).strftime("%d/%m/%Y %H:%M"),
                    "url": f"/salida/{d.name}/mapa.html"})
    return sorted(res, key=lambda r: -r["orden"])


def elegir(tipo):
    """Abre el diálogo nativo de Windows para elegir carpeta o archivo (el servidor corre en esta PC)."""
    import tkinter as tk
    from tkinter import filedialog
    raiz = tk.Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    try:
        if tipo == "lotes":
            r = filedialog.askopenfilename(title="Archivo de lotes", parent=raiz,
                                           filetypes=[("GeoJSON", "*.geojson *.json"), ("Todos", "*.*")])
        else:
            r = filedialog.askdirectory(title="Carpeta de los videos", parent=raiz)
    finally:
        raiz.destroy()
    return str(Path(r)) if r else ""


# ───────────────────────── Servidor HTTP ─────────────────────────

class Manejador(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silencioso
        pass

    def _json(self, datos, codigo=200):
        cuerpo = json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _archivo(self, ruta):
        try:
            cuerpo = ruta.read_bytes()
        except OSError:
            return self.send_error(404)
        tipo = mimetypes.guess_type(ruta.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", tipo + ("; charset=utf-8" if tipo.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _cuerpo(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        ruta = unquote(urlparse(self.path).path)
        if ruta == "/":
            return self._archivo(BASE / "interfaz.html")
        if ruta == "/api/config":
            return self._json(leer_config())
        if ruta == "/api/videos":
            return self._json(listar_videos())
        if ruta == "/api/resultados":
            return self._json(listar_resultados())
        if ruta == "/api/estado":
            return self._json(TRABAJO.estado())
        if ruta.startswith("/salida/"):
            destino = (SALIDA / ruta[len("/salida/"):]).resolve()
            if destino.is_relative_to(SALIDA.resolve()) and destino.is_file():
                return self._archivo(destino)
        self.send_error(404)

    def do_POST(self):
        ruta = urlparse(self.path).path
        try:
            datos = self._cuerpo()
        except ValueError:
            return self._json({"error": "JSON inválido"}, 400)
        if ruta == "/api/config":
            return self._json(guardar_config(datos))
        if ruta == "/api/elegir":
            return self._json({"ruta": elegir(datos.get("tipo"))})
        if ruta == "/api/procesar":
            videos = [v for v in datos.get("videos", []) if Path(v).is_file()]
            if not videos:
                return self._json({"error": "No hay videos seleccionados"}, 400)
            if not TRABAJO.iniciar(videos):
                return self._json({"error": "Ya hay un procesamiento en curso"}, 409)
            return self._json({"ok": True})
        self.send_error(404)


class Servidor(ThreadingHTTPServer):
    request_queue_size = 128  # el mapa pide muchas imágenes juntas; con la cola por defecto (5) Windows corta conexiones
    daemon_threads = True
    allow_reuse_address = False  # en Windows permitiría DOS servidores en el mismo puerto


def main():
    url = f"http://127.0.0.1:{PUERTO}"
    try:
        servidor = Servidor(("127.0.0.1", PUERTO), Manejador)
    except OSError:
        print(f"La interfaz ya está abierta en {url}")
        webbrowser.open(url)
        return
    print(f"Interfaz en {url}  (cerrá esta ventana para salir)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
