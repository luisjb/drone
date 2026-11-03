"""
Inventario de autos en playón con drone DJI (probado pensando en Air 3S).

Toma las fotos de un vuelo (cámara a -90°, mirando al piso), detecta los autos,
calcula la coordenada GPS de cada uno a partir de los metadatos de la foto
(GPS, altura, orientación del gimbal y focal), elimina duplicados entre fotos
superpuestas, lee AprilTags en el techo (si los hay), asigna el lote y guarda
todo en una base SQLite + un mapa HTML.

Comandos:
  python procesar_vuelo.py procesar CARPETA_FOTOS [--lotes lotes.geojson]
  python procesar_vuelo.py vigilar  E:\\DCIM          # procesa solo al insertar la microSD
  python procesar_vuelo.py video    DJI_0001.MP4      # video + su .SRT (subtítulos de DJI Fly)
  python procesar_vuelo.py asignar  --tag 17 --vin 8AJ... --modelo "Hilux SRX" --color Blanco
  python procesar_vuelo.py buscar   --tag 17        (o --vin ...)

Requisitos:  pip install ultralytics opencv-python
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "inventario.db"
MODELO = BASE / "modelos" / "yolo11s-obb.pt"   # YOLO11 OBB entrenado en DOTA (imágenes aéreas)
CLASES_VEHICULO = [9, 10]                       # 9 = large vehicle, 10 = small vehicle
GSD_OBJETIVO = (0.16, 0.22)                     # m/píxel a los que el modelo detecta mejor (2 escalas)
METROS_POR_GRADO = 111_320.0
EXTENSIONES = {".jpg", ".jpeg"}


# ───────────────────────────── Metadatos DJI ─────────────────────────────

def _num(v):
    try:
        return float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _gps_exif_a_grados(valor, ref):
    g, m, s = (float(x) for x in valor)
    d = g + m / 60 + s / 3600
    return -d if ref in ("S", "W") else d


def leer_metadatos(ruta):
    """Devuelve dict con lat, lon, alt (relativa al despegue), yaw, pitch, f35, fecha, ancho, alto."""
    im = Image.open(ruta)
    ancho, alto = im.size
    exif = im.getexif()
    exif_ifd = exif.get_ifd(0x8769)
    gps_ifd = exif.get_ifd(0x8825)

    datos = open(ruta, "rb").read(256 * 1024)  # el XMP de DJI está al principio del archivo
    xmp = {}
    ini, fin = datos.find(b"<x:xmpmeta"), datos.find(b"</x:xmpmeta")
    if ini != -1 and fin != -1:
        texto = datos[ini:fin].decode("utf-8", "ignore")
        for k, v in re.findall(r'drone-dji:(\w+)="([^"]*)"', texto):
            xmp[k] = v
        for k, v in re.findall(r"<drone-dji:(\w+)>([^<]*)</drone-dji:\1>", texto):
            xmp[k] = v

    lat = _num(xmp.get("GpsLatitude"))
    lon = _num(xmp.get("GpsLongitude") or xmp.get("GpsLongtitude"))  # algunos modelos lo escriben mal
    if (lat is None or lon is None) and gps_ifd:
        try:
            lat = _gps_exif_a_grados(gps_ifd[2], gps_ifd.get(1, "N"))
            lon = _gps_exif_a_grados(gps_ifd[4], gps_ifd.get(3, "E"))
        except (KeyError, TypeError, ValueError):
            pass

    f35 = exif_ifd.get(0xA405)  # FocalLengthIn35mmFilm
    fecha = exif_ifd.get(0x9003) or exif.get(0x0132)
    return {
        "lat": lat, "lon": lon,
        "alt": _num(xmp.get("RelativeAltitude")),
        "yaw": _num(xmp.get("GimbalYawDegree")) if "GimbalYawDegree" in xmp else _num(xmp.get("FlightYawDegree")),
        "pitch": _num(xmp.get("GimbalPitchDegree")),
        "f35": float(f35) if f35 else None,
        "fecha": str(fecha) if fecha else None,
        "ancho": ancho, "alto": alto,
    }


# ─────────────────────── Geometría: píxel ↔ suelo ───────────────────────

class Camara:
    """Modelo pinhole. Ejes del mundo: E (este), N (norte), U (arriba). Cámara a altura H sobre el piso."""

    def __init__(self, m):
        self.lat0, self.lon0, self.H = m["lat"], m["lon"], m["alt"]
        self.w, self.h = m["ancho"], m["alto"]
        diag = math.hypot(self.w, self.h)
        self.f = (m["f35"] or 24.0) * diag / 43.267  # focal en píxeles (43.267 mm = diagonal 35 mm)
        self.cx, self.cy = self.w / 2, self.h / 2
        psi = math.radians(m["yaw"] or 0.0)
        th = math.radians(m["pitch"] if m["pitch"] is not None else -90.0)
        self.F = np.array([math.sin(psi) * math.cos(th), math.cos(psi) * math.cos(th), math.sin(th)])
        self.R = np.array([math.cos(psi), -math.sin(psi), 0.0])
        self.U = np.cross(self.R, self.F)
        self.gsd = self.H / self.f  # metros por píxel en el centro de la imagen

    def pixel_a_metros(self, u, v):
        """Offset (este, norte) en metros desde el punto bajo el drone. None si el rayo no toca el piso."""
        rayo = self.R * ((u - self.cx) / self.f) - self.U * ((v - self.cy) / self.f) + self.F
        if rayo[2] > -0.05:
            return None
        t = -self.H / rayo[2]
        return rayo[0] * t, rayo[1] * t

    def metros_a_pixel(self, e, n):
        vec = np.array([e, n, -self.H])
        z = vec @ self.F
        if z <= 0:
            return None
        return self.cx + self.f * (vec @ self.R) / z, self.cy + self.f * (vec @ -self.U) / z

    def metros_a_latlon(self, e, n):
        lat = self.lat0 + n / METROS_POR_GRADO
        lon = self.lon0 + e / (METROS_POR_GRADO * math.cos(math.radians(self.lat0)))
        return lat, lon

    def latlon_a_metros(self, lat, lon):
        n = (lat - self.lat0) * METROS_POR_GRADO
        e = (lon - self.lon0) * METROS_POR_GRADO * math.cos(math.radians(self.lat0))
        return e, n

    def pixel_a_latlon(self, u, v):
        r = self.pixel_a_metros(u, v)
        return None if r is None else self.metros_a_latlon(*r)

    def contiene(self, lat, lon, margen=0.03):
        p = self.metros_a_pixel(*self.latlon_a_metros(lat, lon))
        if p is None:
            return False
        mx, my = self.w * margen, self.h * margen
        return mx <= p[0] <= self.w - mx and my <= p[1] <= self.h - my


def distancia_m(a, b):
    dn = (a[0] - b[0]) * METROS_POR_GRADO
    de = (a[1] - b[1]) * METROS_POR_GRADO * math.cos(math.radians(a[0]))
    return math.hypot(de, dn)


# ───────────────────────────── Detección ─────────────────────────────

_modelo = None


def detectar_autos(img, gsd, conf):
    """Corre el modelo en 2 escalas (ajustadas al GSD real) y une resultados. Devuelve lista de dicts."""
    global _modelo
    if _modelo is None:
        from ultralytics import YOLO
        _modelo = YOLO(str(MODELO) if MODELO.exists() else "yolo11s-obb.pt")
    h, w = img.shape[:2]
    dets = []
    for objetivo in GSD_OBJETIVO:
        imgsz = int(round(w * gsd / objetivo / 32) * 32) if gsd else (512 if objetivo < 0.2 else 448)
        imgsz = max(320, min(imgsz, 2048))
        r = _modelo.predict(img, imgsz=imgsz, conf=conf, classes=CLASES_VEHICULO, agnostic_nms=True, verbose=False)[0]
        for (xc, yc, bw, bh, ang), c, poly in zip(r.obb.xywhr.tolist(), r.obb.conf.tolist(),
                                                   r.obb.xyxyxyxy.tolist()):
            dets.append({"u": xc, "v": yc, "largo_px": max(bw, bh), "ancho_px": min(bw, bh),
                         "ang": ang if bw >= bh else ang + math.pi / 2, "conf": c, "poly": poly})
    # unir duplicados entre escalas (centros muy cercanos) quedándose con el de mayor confianza
    dets.sort(key=lambda d: -d["conf"])
    finales = []
    for d in dets:
        if all(math.hypot(d["u"] - f["u"], d["v"] - f["v"]) > 0.45 * min(d["ancho_px"], f["ancho_px"]) * 2
               for f in finales):
            finales.append(d)
    return finales


def detectar_tags(img):
    """AprilTag 36h11. Devuelve lista de (id, u, v)."""
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    # por defecto OpenCV descarta marcadores chicos respecto de la imagen; en una foto de 48 MP un tag
    # de 26 cm a 50 m mide ~28 px, así que bajamos el mínimo a ~16 px de lado
    params.minMarkerPerimeterRate = 64 / max(img.shape[:2])
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(diccionario, params)
    esquinas, ids, _ = det.detectMarkers(gris)
    if ids is None:
        return []
    return [(int(i), float(c[0][:, 0].mean()), float(c[0][:, 1].mean())) for c, i in zip(esquinas, ids.flatten())]


# ───────────────────────────── Lotes ─────────────────────────────

def cargar_lotes(ruta):
    if not ruta:
        return []
    if not Path(ruta).exists():
        print(f"  aviso: no encontré {ruta}; los autos quedan sin lote asignado")
        return []
    gj = json.load(open(ruta, encoding="utf-8"))
    lotes = []
    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        props = f.get("properties") or {}
        nombre = props.get("nombre") or props.get("name") or props.get("lote") or f"Lote {len(lotes) + 1}"
        polis = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) if g.get("type") == "MultiPolygon" else []
        for p in polis:
            lotes.append((nombre, p[0]))  # anillo exterior, en [lon, lat]
    return lotes


def lote_de(lat, lon, lotes):
    for nombre, anillo in lotes:
        dentro = False
        j = len(anillo) - 1
        for i in range(len(anillo)):
            xi, yi = anillo[i][0], anillo[i][1]
            xj, yj = anillo[j][0], anillo[j][1]
            if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
                dentro = not dentro
            j = i
        if dentro:
            return nombre
    return None


# ───────────────────────────── Base de datos ─────────────────────────────

def abrir_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS vuelos(id INTEGER PRIMARY KEY, nombre TEXT, fecha TEXT, carpeta TEXT,
                                      n_fotos INT, n_autos INT, procesado TEXT);
    CREATE TABLE IF NOT EXISTS autos(vuelo_id INT, auto_id INT, lat REAL, lon REAL, lote TEXT, tag INT,
                                     conf REAL, largo_m REAL, ancho_m REAL, rumbo REAL,
                                     foto TEXT, recorte TEXT);
    CREATE TABLE IF NOT EXISTS vehiculos(tag INT PRIMARY KEY, vin TEXT, modelo TEXT, color TEXT, ingreso TEXT);
    CREATE TABLE IF NOT EXISTS fotos_procesadas(ruta TEXT PRIMARY KEY, tam INT, vuelo_id INT);
    CREATE INDEX IF NOT EXISTS ix_autos_tag ON autos(tag);
    """)
    return con


# ───────────────────────────── Proceso principal ─────────────────────────────

def procesar(carpeta, lotes_path=None, conf=0.2, meta_json=None, salida=None, nombre=None, con=None):
    carpeta = Path(carpeta)
    fotos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not fotos:
        sys.exit(f"No hay fotos JPG en {carpeta}")
    manual = json.load(open(meta_json, encoding="utf-8")) if meta_json else {}
    lotes = cargar_lotes(lotes_path)

    # 1) metadatos y cámaras
    camaras, sin_meta = {}, []
    for p in fotos:
        m = leer_metadatos(p)
        m.update({k: v for k, v in manual.get(p.name, {}).items()})
        if None in (m["lat"], m["lon"], m["alt"]):
            sin_meta.append(p.name)
            continue
        if m["pitch"] is not None and m["pitch"] > -60:
            print(f"  aviso: {p.name} tiene la cámara a {m['pitch']:.0f}° (lo ideal es -90°); la posición será menos precisa")
        camaras[p] = (Camara(m), m)
    if sin_meta:
        print(f"  {len(sin_meta)} foto(s) sin GPS/altura en los metadatos, se omiten: {', '.join(sin_meta[:5])}"
              f"{'...' if len(sin_meta) > 5 else ''}")
    if not camaras:
        sys.exit("Ninguna foto tiene metadatos de GPS. ¿Son los JPG originales del drone?")

    fecha0 = min((m["fecha"] or "") for _, m in camaras.values()) or dt.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    nombre = nombre or "vuelo_" + fecha0.replace(":", "").replace(" ", "_")
    salida = Path(salida or BASE / "salida" / nombre)
    (salida / "anotadas").mkdir(parents=True, exist_ok=True)
    (salida / "recortes").mkdir(parents=True, exist_ok=True)

    # 2) detección en cada foto
    candidatos, huellas = [], []
    for i, (p, (cam, m)) in enumerate(camaras.items(), 1):
        img = cv2.imread(str(p))
        dets = detectar_autos(img, cam.gsd, conf)
        tags = detectar_tags(img)
        print(f"  [{i}/{len(camaras)}] {p.name}: {len(dets)} autos, {len(tags)} tags  "
              f"(altura {cam.H:.0f} m, {cam.gsd * 100:.1f} cm/px)")

        # asociar tags a autos (tag dentro de la caja)
        tag_usado = set()
        for d in dets:
            poly = np.array(d["poly"], dtype=np.float32)
            for tid, tu, tv in tags:
                if cv2.pointPolygonTest(poly, (tu, tv), False) >= 0:
                    d["tag"] = tid
                    tag_usado.add(tid)
        for tid, tu, tv in tags:  # tags sin auto detectado: igual se registran
            if tid not in tag_usado:
                dets.append({"u": tu, "v": tv, "largo_px": 0, "ancho_px": 0, "ang": 0, "conf": 1.0,
                             "poly": None, "tag": tid})

        for d in dets:
            ll = cam.pixel_a_latlon(d["u"], d["v"])
            if ll is None:
                continue
            d.update({"lat": ll[0], "lon": ll[1], "foto": p,
                      "dist_centro": math.hypot(d["u"] - cam.cx, d["v"] - cam.cy)})
            # rumbo del auto respecto al norte
            ext = cam.pixel_a_latlon(d["u"] + 10 * math.cos(d["ang"]), d["v"] + 10 * math.sin(d["ang"]))
            if ext:
                d["rumbo"] = (math.degrees(math.atan2((ext[1] - ll[1]) * math.cos(math.radians(ll[0])),
                                                      ext[0] - ll[0])) + 360) % 180
            candidatos.append(d)
        c = cam.pixel_a_latlon(cam.cx, cam.cy)
        esquinas = [cam.pixel_a_latlon(u, v) for u, v in ((0, 0), (cam.w, 0), (cam.w, cam.h), (0, cam.h))]
        orto = _ortofoto(img, cam, salida / "ortofotos" / (p.stem + ".png"))
        huellas.append({"foto": p.name, "centro": c, "esquinas": esquinas, "orto": orto})
        _guardar_anotada(img, dets, salida / "anotadas" / p.name)

    # 3) duplicados entre fotos: cada punto del piso lo "cuenta" la foto en la que queda más cerca del centro
    centros = {p: cam.pixel_a_latlon(cam.cx, cam.cy) for p, (cam, _) in camaras.items()}
    unicos, descartados = [], []
    for d in candidatos:
        propia = distancia_m((d["lat"], d["lon"]), centros[d["foto"]])
        mejor_otra = any(distancia_m((d["lat"], d["lon"]), centros[q]) < propia - 0.5
                         and camaras[q][0].contiene(d["lat"], d["lon"])
                         for q in camaras if q != d["foto"])
        (descartados if mejor_otra else unicos).append(d)
    # si un tag solo se pudo leer en la foto descartada, se lo pasamos al auto conservado
    for d in descartados:
        if d.get("tag") is None or any(u.get("tag") == d["tag"] for u in unicos):
            continue
        cerca = [u for u in unicos if u.get("tag") is None and distancia_m((d["lat"], d["lon"]), (u["lat"], u["lon"])) < 2.0]
        if cerca:
            min(cerca, key=lambda u: distancia_m((d["lat"], d["lon"]), (u["lat"], u["lon"])))["tag"] = d["tag"]
        else:
            unicos.append(d)
    # red de seguridad: unir lo que quedó a < 1.2 m (bordes entre fotos) y todo lo que comparte tag
    unicos.sort(key=lambda d: (d.get("tag") is None, d["dist_centro"]))
    autos = []
    for d in unicos:
        dup = next((a for a in autos if (d.get("tag") is not None and d.get("tag") == a.get("tag"))
                    or distancia_m((d["lat"], d["lon"]), (a["lat"], a["lon"])) < 1.2), None)
        if dup is None:
            autos.append(d)
        elif dup.get("tag") is None and d.get("tag") is not None:
            dup["tag"] = d["tag"]

    # 4) recortes, lotes y salida
    cache = {}
    for n, a in enumerate(sorted(autos, key=lambda a: (-a["lat"], a["lon"])), 1):
        a["id"] = n
        a["lote"] = lote_de(a["lat"], a["lon"], lotes)
        cam = camaras[a["foto"]][0]
        a["largo_m"] = round(a["largo_px"] * cam.gsd, 1)
        a["ancho_m"] = round(a["ancho_px"] * cam.gsd, 1)
        img = cache.setdefault(a["foto"], cv2.imread(str(a["foto"])))
        r = int(max(a["largo_px"], 60) * 0.75)
        u, v = int(a["u"]), int(a["v"])
        rec = img[max(0, v - r):v + r, max(0, u - r):u + r]
        if max(rec.shape[:2]) > 320:
            f = 320 / max(rec.shape[:2])
            rec = cv2.resize(rec, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        a["recorte"] = f"recortes/auto_{n:04d}.jpg"
        cv2.imwrite(str(salida / a["recorte"]), rec, [cv2.IMWRITE_JPEG_QUALITY, 85])
    autos.sort(key=lambda a: a["id"])

    con = con or abrir_db()
    cur = con.execute("INSERT INTO vuelos(nombre, fecha, carpeta, n_fotos, n_autos, procesado) VALUES(?,?,?,?,?,?)",
                      (nombre, fecha0, str(carpeta), len(camaras), len(autos), dt.datetime.now().isoformat(" ", "seconds")))
    vuelo_id = cur.lastrowid
    con.executemany("INSERT INTO autos VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(vuelo_id, a["id"], a["lat"], a["lon"], a["lote"], a.get("tag"), round(a["conf"], 3),
                      a["largo_m"], a["ancho_m"], round(a.get("rumbo", 0), 1), a["foto"].name,
                      str(salida / a["recorte"])) for a in autos])
    con.commit()
    vehiculos = {r[0]: {"vin": r[1], "modelo": r[2], "color": r[3]}
                 for r in con.execute("SELECT tag, vin, modelo, color FROM vehiculos")}

    _exportar(salida, nombre, autos, huellas, lotes, vehiculos)
    print(f"\n✔ {len(autos)} autos únicos en {len(camaras)} fotos  →  {salida / 'mapa.html'}")
    return vuelo_id, salida


def _ortofoto(img, cam, ruta, lado_max=1800):
    """Endereza la foto al norte (proyección sobre el piso) para superponerla en el mapa.
    Devuelve {"archivo", "sw", "ne"} o None si la cámara no mira al piso."""
    pts_img = [(0, 0), (cam.w, 0), (cam.w, cam.h), (0, cam.h)]
    pts_m = [cam.pixel_a_metros(u, v) for u, v in pts_img]
    if None in pts_m:
        return None
    es, ns = [p[0] for p in pts_m], [p[1] for p in pts_m]
    res = max(max(es) - min(es), max(ns) - min(ns)) / lado_max
    dst = np.float32([((e - min(es)) / res, (max(ns) - n) / res) for e, n in pts_m])
    Hm = cv2.getPerspectiveTransform(np.float32(pts_img), dst)
    tam = (int(math.ceil((max(es) - min(es)) / res)), int(math.ceil((max(ns) - min(ns)) / res)))
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    out = cv2.warpPerspective(rgba, Hm, tam, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ruta), out, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return {"archivo": f"ortofotos/{ruta.name}",
            "sw": cam.metros_a_latlon(min(es), min(ns)), "ne": cam.metros_a_latlon(max(es), max(ns))}


def _guardar_anotada(img, dets, ruta):
    out = img.copy()
    for d in dets:
        if d["poly"] is not None:
            cv2.polylines(out, [np.array(d["poly"], dtype=np.int32)], True, (0, 220, 255), 3)
        if d.get("tag") is not None:
            cv2.putText(out, f"#{d['tag']}", (int(d["u"]) - 20, int(d["v"])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
            cv2.putText(out, f"#{d['tag']}", (int(d["u"]) - 20, int(d["v"])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(out, f"{len(dets)} autos", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
    cv2.putText(out, f"{len(dets)} autos", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    if out.shape[1] > 3000:
        f = 3000 / out.shape[1]
        out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(ruta), out, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _exportar(salida, nombre, autos, huellas, lotes, vehiculos):
    filas = []
    for a in autos:
        v = vehiculos.get(a.get("tag"), {})
        filas.append({"id": a["id"], "lat": round(a["lat"], 7), "lon": round(a["lon"], 7), "lote": a["lote"],
                      "tag": a.get("tag"), "vin": v.get("vin"), "modelo": v.get("modelo"), "color": v.get("color"),
                      "conf": round(a["conf"], 2), "largo_m": a["largo_m"], "rumbo": round(a.get("rumbo", 0)),
                      "foto": a["foto"].name, "recorte": a["recorte"]})
    # CSV
    import csv
    with open(salida / "autos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()) if filas else ["id"])
        w.writeheader()
        w.writerows(filas)
    # GeoJSON
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
         "properties": {k: v for k, v in r.items() if k not in ("lat", "lon")}} for r in filas]}
    json.dump(gj, open(salida / "autos.geojson", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # Mapa
    datos = {"nombre": nombre, "autos": filas,
             "huellas": [{"foto": h["foto"], "esquinas": [e for e in h["esquinas"] if e], "orto": h["orto"]}
                         for h in huellas],
             "lotes": [{"nombre": n, "anillo": [[p[1], p[0]] for p in a]} for n, a in lotes]}
    html = (BASE / "plantilla_mapa.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATOS__*/null", json.dumps(datos, ensure_ascii=False))
    (salida / "mapa.html").write_text(html, encoding="utf-8")


# ───────────────────────────── Vigilar microSD / carpeta ─────────────────────────────

def vigilar(origen, lotes_path=None, conf=0.2, cada=15, espera=20):
    """Revisa ORIGEN cada N segundos. Cuando aparecen fotos nuevas y dejan de cambiar durante
    ESPERA segundos (tarjeta montada / copia terminada), las copia a vuelos/<fecha> y las procesa."""
    origen = Path(origen)
    con = abrir_db()
    print(f"Vigilando {origen}  (Ctrl+C para salir)")
    ultimo_estado, estable_desde = None, None
    while True:
        try:
            nuevas = []
            if origen.exists():
                hechas = {r[0] for r in con.execute("SELECT ruta FROM fotos_procesadas")}
                for p in origen.rglob("*"):
                    if p.suffix.lower() in EXTENSIONES and str(p.resolve()) not in hechas:
                        nuevas.append(p)
            estado = tuple(sorted((str(p), p.stat().st_size) for p in nuevas))
            if nuevas and estado == ultimo_estado:
                if time.time() - estable_desde >= espera:
                    for grupo in _agrupar_por_vuelo(nuevas):
                        fecha = (leer_metadatos(grupo[0])["fecha"] or "").replace(":", "").replace(" ", "_")
                        destino = BASE / "vuelos" / (fecha or dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
                        destino.mkdir(parents=True, exist_ok=True)
                        for p in grupo:
                            shutil.copy2(p, destino / p.name)
                        print(f"\n▶ Vuelo nuevo: {len(grupo)} fotos → {destino}")
                        try:
                            vid, _ = procesar(destino, lotes_path, conf, con=con)
                        except SystemExit as e:
                            print(f"  no se pudo procesar: {e}")
                            vid = None
                        con.executemany("INSERT OR REPLACE INTO fotos_procesadas VALUES(?,?,?)",
                                        [(str(p.resolve()), p.stat().st_size, vid) for p in grupo])
                        con.commit()
                    ultimo_estado = None
                    continue
            else:
                ultimo_estado, estable_desde = estado, time.time()
            time.sleep(cada)
        except KeyboardInterrupt:
            print("\nListo.")
            return
        except OSError as e:  # tarjeta retirada a mitad de lectura
            print(f"  aviso: {e}")
            ultimo_estado = None
            time.sleep(cada)


def _agrupar_por_vuelo(fotos, corte_min=10):
    """Separa vuelos distintos: si entre dos fotos pasan más de 10 minutos, es otro vuelo."""
    def ts(p):
        f = leer_metadatos(p)["fecha"]
        try:
            return dt.datetime.strptime(f, "%Y:%m:%d %H:%M:%S").timestamp()
        except (TypeError, ValueError):
            return p.stat().st_mtime
    orden = sorted(fotos, key=ts)
    grupos, previo = [], None
    for p in orden:
        t = ts(p)
        if previo is None or t - previo > corte_min * 60:
            grupos.append([])
        grupos[-1].append(p)
        previo = t
    return grupos


# ───────────────────── Video: MP4 + subtítulos .SRT de DJI ─────────────────────
# DJI Fly graba un .SRT junto a cada video (activar Cámara → "Subtítulos de video") con la posición
# de cada cuadro. Con eso se sacan cuadros cada N metros y se procesan igual que las fotos.

HFOV_PRINCIPAL = 71.6  # campo de visión horizontal (°) de la cámara principal de 24 mm del Air 3S


def leer_srt(ruta):
    """Telemetría por cuadro del .SRT de DJI. Devuelve lista ordenada de dicts:
    t (s desde el inicio del video), lat, lon, alt (relativa al despegue), yaw, pitch (None si no vienen), fecha."""
    texto = Path(ruta).read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    filas = []
    for bloque in re.split(r"\n\s*\n", texto):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->", bloque)
        if not m:
            continue
        t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
        c = {k.lower(): float(v) for k, v in re.findall(r"([A-Za-z_]+)\s*:\s*([-+]?\d+(?:\.\d+)?)", bloque)}
        lat, lon = c.get("latitude"), c.get("longitude", c.get("longtitude"))
        if lat is None and (g := re.search(r"GPS\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)", bloque)):  # formato viejo: GPS(lon, lat, ...)
            lon, lat = float(g[1]), float(g[2])
        alt = c.get("rel_alt")
        if alt is None and (h := re.search(r"\bH\s*[:]?\s*([-\d.]+)\s*m", bloque)):
            alt = float(h[1])
        if lat is None or lon is None or alt is None or (abs(lat) < 1e-6 and abs(lon) < 1e-6):
            continue  # sin GPS en ese cuadro
        f = re.search(r"(\d{4})[-.:/](\d{2})[-.:/](\d{2})\s+(\d{2}):(\d{2}):(\d{2})", bloque)
        filas.append({"t": t, "lat": lat, "lon": lon, "alt": alt, "yaw": c.get("gb_yaw"), "pitch": c.get("gb_pitch"),
                      "fecha": f"{f[1]}:{f[2]}:{f[3]} {f[4]}:{f[5]}:{f[6]}" if f else None})
    filas.sort(key=lambda r: r["t"])
    return filas


def _rumbo_por_trayectoria(filas, ventana_s=1.5, min_m=2.0):
    """Si el .SRT no trae el yaw del gimbal, se estima por la dirección de avance del drone
    (sirve cuando la misión vuela 'mirando hacia adelante', que es lo normal en waypoints)."""
    j0 = j1 = 0
    ultimo = None
    for r in filas:
        while filas[j0]["t"] < r["t"] - ventana_s:
            j0 += 1
        while j1 + 1 < len(filas) and filas[j1 + 1]["t"] <= r["t"] + ventana_s:
            j1 += 1
        a, b = filas[j0], filas[j1]
        dn = (b["lat"] - a["lat"]) * METROS_POR_GRADO
        de = (b["lon"] - a["lon"]) * METROS_POR_GRADO * math.cos(math.radians(a["lat"]))
        if math.hypot(de, dn) >= min_m:
            ultimo = (math.degrees(math.atan2(de, dn)) + 360) % 360
        r["yaw_trayectoria"] = ultimo


def video(ruta_video, srt=None, lotes_path=None, conf=0.2, cada_m=15.0, alt_min=10.0, hfov=HFOV_PRINCIPAL,
          yaw_fijo=None, destino=None, salida=None):
    """Saca un cuadro cada CADA_M metros recorridos (el más nítido alrededor de ese momento), le asigna
    la posición del .SRT y procesa el conjunto con el mismo motor que las fotos."""
    ruta_video = Path(ruta_video)
    srt = Path(srt) if srt else next((p for p in (ruta_video.with_suffix(".SRT"), ruta_video.with_suffix(".srt"))
                                      if p.exists()), None)
    if not srt or not srt.exists():
        sys.exit(f"No encontré el .SRT de {ruta_video.name}. Activá 'Subtítulos de video' en DJI Fly "
                 f"o indicá el archivo con --srt.")
    filas = leer_srt(srt)
    if not filas:
        sys.exit(f"{srt.name} no tiene posiciones GPS válidas.")
    tiene_yaw = any(r["yaw"] is not None for r in filas)
    if not tiene_yaw and yaw_fijo is None:
        _rumbo_por_trayectoria(filas)
        print("  el .SRT no trae el yaw del gimbal: se estima por la dirección de avance "
              "(si la misión usa un rumbo fijo, indicalo con --yaw)")

    # 1) momentos a extraer: cada CADA_M metros, en altura de trabajo
    objetivos, previo = [], None
    for r in filas:
        if r["alt"] < alt_min:
            continue
        if previo is None or distancia_m((r["lat"], r["lon"]), (previo["lat"], previo["lon"])) >= cada_m:
            objetivos.append(r)
            previo = r
    if not objetivos:
        sys.exit(f"El video nunca supera {alt_min:.0f} m de altura.")

    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        sys.exit(f"OpenCV no pudo abrir {ruta_video.name} (si es H.265, convertilo a H.264 o actualizá opencv-python).")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_px = (w / 2) / math.tan(math.radians(hfov) / 2)
    f35 = f_px * 43.267 / math.hypot(w, h)  # Camara() convierte f35 → píxeles con la diagonal

    destino = Path(destino or BASE / "vuelos" / f"video_{ruta_video.stem}")
    destino.mkdir(parents=True, exist_ok=True)
    print(f"▶ {ruta_video.name}: {w}x{h}, {len(filas)} posiciones, {len(objetivos)} cuadros a extraer → {destino}")

    # 2) lectura secuencial: de los cuadros a ±0.3 s de cada objetivo se queda con el más nítido
    meta, k, mejor = {}, 0, None
    tiempos = [r["t"] for r in filas]

    def guardar(m):
        t, img, _ = m
        i = int(np.searchsorted(tiempos, t))
        r = min(filas[max(i - 1, 0):i + 1], key=lambda f: abs(f["t"] - t))  # posición más cercana en el tiempo
        nombre = f"{ruta_video.stem}_{int(t * 1000):08d}.jpg"
        cv2.imwrite(str(destino / nombre), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        yaw = yaw_fijo if yaw_fijo is not None else r["yaw"] if tiene_yaw else r.get("yaw_trayectoria")
        meta[nombre] = {"lat": r["lat"], "lon": r["lon"], "alt": r["alt"], "yaw": yaw,
                        "pitch": r["pitch"] if r["pitch"] is not None else -90.0, "f35": f35, "fecha": r["fecha"]}

    while k < len(objetivos) and cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t < objetivos[k]["t"] - 0.3:
            continue
        if t <= objetivos[k]["t"] + 0.3:
            ok, img = cap.retrieve()
            if ok:
                chico = cv2.cvtColor(cv2.resize(img, (w // 4, h // 4)), cv2.COLOR_BGR2GRAY)
                nitidez = cv2.Laplacian(chico, cv2.CV_64F).var()
                if mejor is None or nitidez > mejor[2]:
                    mejor = (t, img, nitidez)
            continue
        if mejor:
            guardar(mejor)
        mejor = None
        k += 1
        while k < len(objetivos) and objetivos[k]["t"] + 0.3 < t:
            k += 1  # objetivos salteados (video más corto que el .SRT, etc.)
    if mejor:
        guardar(mejor)
    cap.release()
    if not meta:
        sys.exit("No se pudo extraer ningún cuadro del video.")
    sin_yaw = sum(1 for m in meta.values() if m["yaw"] is None)
    if sin_yaw:
        print(f"  aviso: {sin_yaw} cuadro(s) sin rumbo conocido (drone quieto); se asume norte arriba")
    ruta_meta = destino / "metadatos_video.json"
    ruta_meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return procesar(destino, lotes_path, conf, meta_json=ruta_meta, salida=salida, nombre=f"video_{ruta_video.stem}")


# ───────────────────────────── Asignar / buscar ─────────────────────────────

def asignar(tag, vin, modelo, color):
    con = abrir_db()
    con.execute("INSERT OR REPLACE INTO vehiculos VALUES(?,?,?,?,?)",
                (tag, vin, modelo, color, dt.date.today().isoformat()))
    con.commit()
    print(f"Tag #{tag} → {vin} {modelo or ''} {color or ''}")


def buscar(tag=None, vin=None):
    con = abrir_db()
    if vin:
        fila = con.execute("SELECT tag FROM vehiculos WHERE vin LIKE ?", (f"%{vin}%",)).fetchone()
        if not fila:
            sys.exit(f"No hay ningún tag asignado al VIN {vin}")
        tag = fila[0]
    fila = con.execute("""SELECT a.lat, a.lon, a.lote, v.nombre, v.fecha, a.foto, a.recorte
                          FROM autos a JOIN vuelos v ON v.id = a.vuelo_id
                          WHERE a.tag = ? ORDER BY v.fecha DESC, v.id DESC LIMIT 1""", (tag,)).fetchone()
    if not fila:
        sys.exit(f"El tag #{tag} no aparece en ningún vuelo")
    lat, lon, lote, vuelo, fecha, foto, recorte = fila
    print(f"Tag #{tag}  —  Lote: {lote or 'sin lote'}\n  {lat:.7f}, {lon:.7f}\n"
          f"  https://www.google.com/maps?q={lat:.7f},{lon:.7f}\n  Visto en {vuelo} ({fecha}), foto {foto}\n  {recorte}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("procesar", help="Procesa una carpeta de fotos de un vuelo")
    p.add_argument("carpeta")
    p.add_argument("--lotes", help="GeoJSON con los polígonos de los lotes (propiedad 'nombre')")
    p.add_argument("--conf", type=float, default=0.2)
    p.add_argument("--meta-json", help="JSON {foto: {lat, lon, alt, yaw, pitch, f35}} para completar/forzar metadatos")
    p.add_argument("--salida")
    v = sub.add_parser("vigilar", help="Procesa automáticamente las fotos nuevas que aparezcan (p. ej. la microSD)")
    v.add_argument("origen")
    v.add_argument("--lotes")
    v.add_argument("--conf", type=float, default=0.2)
    v.add_argument("--cada", type=int, default=15, help="segundos entre revisiones")
    vd = sub.add_parser("video", help="Procesa un video MP4 usando la telemetría de su archivo .SRT")
    vd.add_argument("video")
    vd.add_argument("--srt", help="por defecto, el .SRT con el mismo nombre que el video")
    vd.add_argument("--lotes")
    vd.add_argument("--conf", type=float, default=0.2)
    vd.add_argument("--cada-m", type=float, default=15.0, help="metros recorridos entre cuadros extraídos")
    vd.add_argument("--alt-min", type=float, default=10.0, help="ignora despegue/aterrizaje por debajo de esta altura")
    vd.add_argument("--hfov", type=float, default=HFOV_PRINCIPAL,
                    help="campo de visión horizontal en grados (principal 24 mm ≈ 71.6; tele 70 mm ≈ 27.8)")
    vd.add_argument("--yaw", type=float, help="rumbo fijo de la cámara (°, 0 = norte) si la misión no gira")
    vd.add_argument("--salida")
    a = sub.add_parser("asignar", help="Asocia un número de tag a un vehículo")
    a.add_argument("--tag", type=int, required=True)
    a.add_argument("--vin", required=True)
    a.add_argument("--modelo")
    a.add_argument("--color")
    b = sub.add_parser("buscar", help="Última ubicación conocida de un vehículo")
    b.add_argument("--tag", type=int)
    b.add_argument("--vin")
    args = ap.parse_args()

    if args.cmd == "procesar":
        procesar(args.carpeta, args.lotes, args.conf, args.meta_json, args.salida)
    elif args.cmd == "vigilar":
        vigilar(args.origen, args.lotes, args.conf, args.cada)
    elif args.cmd == "video":
        video(args.video, args.srt, args.lotes, args.conf, args.cada_m, args.alt_min, args.hfov, args.yaw,
              salida=args.salida)
    elif args.cmd == "asignar":
        asignar(args.tag, args.vin, args.modelo, args.color)
    elif args.cmd == "buscar":
        if not (args.tag or args.vin):
            sys.exit("Indicá --tag o --vin")
        buscar(args.tag, args.vin)


if __name__ == "__main__":
    main()
