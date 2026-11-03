"""
Genera AprilTags (familia 36h11) para imprimir en hojas A4, un tag por hoja, cada uno con un
vehículo de EJEMPLO con datos aleatorios (modelo, color, VIN, patente, fecha de ingreso).

  python generar_tags.py                       → 10 tags (#0 a #9)   → tags_0-9.pdf + tags_0-9.csv
  python generar_tags.py --cantidad 20         → 20 tags (#0 a #19)
  python generar_tags.py --cantidad 20 --registrar   además los carga en inventario.db (buscador del mapa)
  python generar_tags.py --semilla 7           → otros datos aleatorios (misma semilla = mismos datos)

Tamaño: el cuadrado negro mide 18 cm (lo máximo que entra en A4 con el borde blanco).
Para leerlo hay que volar más bajo que con tags de 26 cm: ≈ 35 m con fotos de 48 MP, ≈ 20 m con video 4K.

Imprimir en imán vinilo MATE (sin brillo) y no recortar el borde blanco: el detector lo necesita.
"""
import argparse
import csv
import datetime as dt
import random
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

HOJA_CM = (21.0, 29.7)  # A4
LADO_MAX_CM = 18.0
DPI = 150

# (marca, modelo, prefijo VIN del fabricante)
MODELOS = [
    ("Toyota", "Hilux SRX", "8AJ"), ("Toyota", "Corolla Cross XEI", "9BR"), ("Toyota", "Yaris XLS", "9BR"),
    ("Ford", "Ranger XLT", "8AF"), ("Ford", "Territory Titanium", "LVS"), ("Volkswagen", "Amarok V6", "8AW"),
    ("Volkswagen", "Taos Highline", "8AW"), ("Fiat", "Cronos Drive", "8AP"), ("Peugeot", "208 Allure", "8AD"),
    ("Chevrolet", "Onix Premier", "9BG"), ("Renault", "Kangoo Stepway", "93Y"), ("Nissan", "Frontier PRO-4X", "8AN"),
]
COLORES = ["Blanco", "Gris Plata", "Negro", "Gris Oscuro", "Rojo", "Azul", "Bordó", "Beige"]


def vin_aleatorio(rng, prefijo):
    """VIN de 17 caracteres con dígito verificador válido (norma ISO 3779 / NHTSA)."""
    chars = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # sin I, O, Q
    vin = list(prefijo + "".join(rng.choice(chars) for _ in range(5)) + "0" + rng.choice("PRSTV")
               + rng.choice(chars) + "".join(rng.choice("0123456789") for _ in range(6)))
    val = {**{str(i): i for i in range(10)},
           **dict(zip("ABCDEFGH", range(1, 9))), **dict(zip("JKLMNPR", (1, 2, 3, 4, 5, 7, 9))),
           **dict(zip("STUVWXYZ", range(2, 10)))}
    pesos = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)
    resto = sum(val[c] * p for c, p in zip(vin, pesos)) % 11
    vin[8] = "X" if resto == 10 else str(resto)
    return "".join(vin)


def vehiculo_aleatorio(rng, tag):
    marca, modelo, prefijo = rng.choice(MODELOS)
    letras = "ABCDEFGHJKLMNPRSTUVWXYZ"
    return {
        "tag": tag,
        "vin": vin_aleatorio(rng, prefijo),
        "modelo": f"{marca} {modelo}",
        "color": rng.choice(COLORES),
        "patente": f"A{rng.choice('ABCDEFGH')} {rng.randint(100, 999)} {rng.choice(letras)}{rng.choice(letras)}",
        "ingreso": (dt.date.today() - dt.timedelta(days=rng.randint(0, 60))).isoformat(),
    }


def tag_imagen(tag_id, lado_px):
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    base = cv2.aruco.generateImageMarker(dic, tag_id, 8, borderBits=1)  # 6x6 de datos + borde negro
    return cv2.resize(base, (lado_px, lado_px), interpolation=cv2.INTER_NEAREST)


def fuente(tam, negrita=True):
    nombres = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if negrita else ("arial.ttf", "DejaVuSans.ttf")
    for nombre in nombres:
        try:
            return ImageFont.truetype(nombre, tam)
        except OSError:
            pass
    return ImageFont.load_default()


def hoja(v, lado_cm):
    px = lambda cm: int(round(cm / 2.54 * DPI))
    W, H, lado = px(HOJA_CM[0]), px(HOJA_CM[1]), px(lado_cm)
    img = Image.new("L", (W, H), 255)
    x0, y0 = (W - lado) // 2, px(2)
    img.paste(Image.fromarray(tag_imagen(v["tag"], lado)), (x0, y0))
    d = ImageDraw.Draw(img)
    # guía de corte: deja 1 cm de blanco alrededor del cuadrado negro
    d.rectangle([x0 - px(1), y0 - px(1), x0 + lado + px(1), y0 + lado + px(1)], outline=190, width=2)

    y = y0 + lado + px(2.4)
    d.text((W // 2, y), f"#{v['tag']}", fill=0, font=fuente(px(2.2)), anchor="mm")
    y += px(2.0)
    d.text((W // 2, y), v["modelo"], fill=0, font=fuente(px(0.8)), anchor="mm")
    y += px(1.0)
    d.text((W // 2, y), f"{v['color']}  ·  Patente {v['patente']}", fill=40, font=fuente(px(0.55), False), anchor="mm")
    y += px(0.8)
    d.text((W // 2, y), f"VIN {v['vin']}  ·  Ingreso {v['ingreso']}", fill=40, font=fuente(px(0.55), False), anchor="mm")
    d.text((W // 2, H - px(1.2)),
           f"DATOS DE EJEMPLO (aleatorios)  ·  AprilTag 36h11  ·  {lado_cm:g} cm  ·  no recortar el borde blanco",
           fill=120, font=fuente(px(0.4), False), anchor="mm")
    return img.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cantidad", type=int, default=10, help="cuántos tags generar (por defecto 10)")
    ap.add_argument("--desde", type=int, default=0, help="número del primer tag")
    ap.add_argument("--cm", type=float, default=LADO_MAX_CM, help=f"lado del cuadrado negro (máx. {LADO_MAX_CM:g} cm)")
    ap.add_argument("--semilla", type=int, help="semilla de los datos aleatorios (para repetir los mismos)")
    ap.add_argument("--registrar", action="store_true",
                    help="cargar los vehículos de ejemplo en inventario.db (reemplaza lo que tengan esos tags)")
    ap.add_argument("--salida", help="nombre del PDF")
    a = ap.parse_args()

    hasta = a.desde + a.cantidad - 1
    if a.cantidad < 1 or not 0 <= a.desde <= hasta <= 586:
        raise SystemExit("La familia 36h11 tiene ids del 0 al 586")
    if not 5 <= a.cm <= LADO_MAX_CM:
        raise SystemExit(f"En A4 el tag puede medir entre 5 y {LADO_MAX_CM:g} cm (hace falta 1 cm de borde blanco)")

    rng = random.Random(a.semilla)
    vehiculos = [vehiculo_aleatorio(rng, t) for t in range(a.desde, hasta + 1)]

    salida = Path(a.salida or f"tags_{a.desde}-{hasta}.pdf")
    paginas = [hoja(v, a.cm) for v in vehiculos]
    paginas[0].save(salida, save_all=True, append_images=paginas[1:], resolution=DPI)
    with open(salida.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(vehiculos[0]))
        w.writeheader()
        w.writerows(vehiculos)
    print(f"{len(paginas)} tags (#{a.desde} a #{hasta}) → {salida}  (A4, cuadrado de {a.cm:g} cm)")
    print(f"Datos de ejemplo → {salida.with_suffix('.csv')}")

    if a.registrar:
        import procesar_vuelo as pv
        con = pv.abrir_db()
        con.executemany("INSERT OR REPLACE INTO vehiculos VALUES(?,?,?,?,?)",
                        [(v["tag"], v["vin"], v["modelo"], v["color"], v["ingreso"]) for v in vehiculos])
        con.commit()
        print(f"Registrados en {pv.DB_PATH.name}: ahora se pueden buscar por tag o VIN.")


if __name__ == "__main__":
    main()
