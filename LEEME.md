# Inventario de autos en playón con drone DJI

Proceso: **vuelo automático → fotos → programa → mapa + base de datos con la posición de cada auto.**

```
 microSD del drone ──► procesar_vuelo.py vigilar ──► detecta autos + lee tags ──► GPS de cada auto
                                                                                   │
                         inventario.db  ◄────────────────────────────────────────────┤
                         salida/<vuelo>/mapa.html, autos.csv, autos.geojson ◄─────────┘
```

## 1. Instalación (una sola vez)

```
pip install -r requirements.txt
```
La primera vez se descarga solo el modelo `yolo11s-obb.pt` (detector entrenado con imágenes aéreas).
Si tenés placa NVIDIA, instalá antes PyTorch con CUDA (pytorch.org) y va mucho más rápido.

## 2. Configuración del vuelo (DJI Air 3S)

| Parámetro | Valor recomendado | Por qué |
|---|---|---|
| Cámara | **Principal (24 mm)**, modo **48 MP** | Los tags necesitan resolución |
| Inclinación del gimbal | **−90°** (mirando al piso) | Así el cálculo de posición es exacto |
| Altura | **≈ 35 m** (máx. ~50 m) | Un tag de 18 cm (A4) tiene que medir ≥ 28 px en la foto |
| Superposición entre fotos | 25–40 % | Alcanza: no es fotogrametría, el programa elimina duplicados |
| Formato | JPG (puede ser JPG+RAW) | El programa lee los JPG |

A 35 m y 48 MP, cada foto cubre unos **50 × 38 m**: separá las pasadas unos 30 m y sacá una foto cada ~22 m.

**Misión de waypoints en DJI Fly:** armá una vez la grilla sobre el playón con la acción *Tomar foto* en cada
punto y el gimbal a −90°. Guardala y reutilizala: el operador solo la carga y aprieta *Go*. El despegue
y la vuelta son automáticos (dejá configurado RTH).

## 3. Interfaz (la forma más simple)

Doble clic en **`iniciar_interfaz.bat`** (o `python interfaz.py`). Se abre el navegador con tres pasos:

1. **Configuración:** carpeta de los videos (puede ser la microSD), archivo de lotes, cámara usada,
   cada cuántos metros sacar un cuadro, altura mínima, confianza de detección y orientación de la cámara.
   Se guarda en `config.json`.
2. **Videos:** lista los MP4 de la carpeta con su estado (*nuevo*, *procesado* o *sin .SRT*). Elegís cuáles
   procesar y ves el avance en vivo.
3. **Resultados:** todos los vuelos procesados (de video y de fotos) con su cantidad de autos y tags; al
   elegir uno se muestra su mapa a la derecha.

La interfaz funciona solo en esa computadora (no se abre a la red). Para cerrarla, cerrá la ventana negra.

## 3.b Procesamiento automático por línea de comandos

Dejá corriendo en la notebook (o hacé doble clic en `iniciar_vigilancia.bat`):

```
python procesar_vuelo.py vigilar E:\DCIM --lotes lotes.geojson
```
Al terminar el vuelo, poné la microSD en la notebook. El programa detecta las fotos nuevas, espera a que
terminen de copiarse, las guarda en `vuelos/<fecha>/` y genera `salida/vuelo_<fecha>/mapa.html`.
Si en una tarjeta hay varios vuelos (más de 10 minutos entre fotos), los separa solo.
Una foto ya procesada no se vuelve a procesar.

Para procesar una carpeta a mano:  `python procesar_vuelo.py procesar CARPETA --lotes lotes.geojson`

### Con video en vez de fotos

```
python procesar_vuelo.py video E:\DCIM\100MEDIA\DJI_0001.MP4 --lotes lotes.geojson
```
* En DJI Fly activá **Cámara → Subtítulos de video**: así graba un `.SRT` junto al `.MP4` con la posición
  GPS y la altura de cada cuadro. Sin ese archivo el video no se puede ubicar.
* El programa saca un cuadro cada 15 m recorridos (`--cada-m`), el más nítido de ese momento, y lo procesa
  igual que una foto.
* El `.SRT` del Air 3S no trae el rumbo de la cámara: se calcula por la dirección de avance. Para eso, la misión
  tiene que volar **mirando hacia adelante**. Si usa un rumbo fijo, indicalo: `--yaw 90`.
* El video tiene mucha menos resolución que la foto de 48 MP: en 4K, para leer los tags hay que volar a
  **≈ 20 m** (a 40 m un tag de 18 cm mide ~12 px y no se lee). Los autos se detectan bien a 40 m.
* No uses zoom. Con la cámara tele de 70 mm, agregá `--hfov 27.8`.

## 4. Tags en el techo (AprilTag)

* `python generar_tags.py` → PDF con 10 tags, uno por hoja **A4** (cuadrado de 18 cm), y un CSV.
  `--cantidad 20` genera 20; `--desde 50` empieza en otro número.
* Cada hoja trae un **vehículo de ejemplo con datos aleatorios** (modelo, color, VIN, patente, ingreso),
  marcado como "DATOS DE EJEMPLO". Sirve para demostraciones; `--semilla 7` da otros datos y repite
  siempre los mismos con la misma semilla.
* `--registrar` carga esos vehículos de ejemplo en `inventario.db`, así el buscador del mapa los encuentra
  por tag o VIN. Reemplaza lo que hubiera asignado a esos números de tag.
* Imprimir en **imán vinilo mate** y respetar el **borde blanco** alrededor del cuadrado negro.
* Al ingresar un auto al playón, se le pone un imán y se registra:
  `python procesar_vuelo.py asignar --tag 17 --vin 8AJ... --modelo "Hilux SRX" --color Blanco`
* Para encontrarlo: `python procesar_vuelo.py buscar --vin 8AJ...` (o `--tag 17`), o desde el buscador del mapa.

## 5. Lotes

Dibujá una vez los polígonos de los lotes en **geojson.io** (o QGIS), poné a cada uno la propiedad
`"nombre": "Lote A"` y guardalo como `lotes.geojson`. `lotes_ejemplo.geojson` muestra el formato.

## 6. Qué genera cada vuelo

* `mapa.html`: tus fotos del vuelo enderezadas sobre el satélite, un punto por auto (verde = con tag) y un buscador.
* `autos.csv` / `autos.geojson`: id, lat, lon, lote, tag, VIN, modelo, largo, orientación y foto de origen.
* `anotadas/`: cada foto con los autos marcados.
* `recortes/`: una miniatura por auto.
* `inventario.db` (SQLite): todos los vuelos, para el historial y las búsquedas.

## 7. Precisión y límites

* **Posición:** depende del GPS del drone (±1–3 m) y de la brújula. Alcanza para lote y fila. Para precisión
  de cajón, poné algunos tags fijos en el piso como referencia.
* **Altura:** se usa la altura relativa al punto de despegue. Despegá desde el nivel del playón.
* **Detección:** con el modelo genérico se detectan ~88 % de los autos; los que fallan suelen estar bajo
  árboles. Hacer *fine-tuning* con 200–300 fotos propias etiquetadas lo lleva por encima del 98 %.
  Un auto con tag se registra aunque el detector no lo encuentre.
* **Tags:** se leen desde ~20 px de lado. Apuntá a 28 px o más (ver la tabla de vuelo).
