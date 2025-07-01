import os
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import streamlit as st
import requests
import time
import xml.etree.ElementTree as ET
import gpxpy
import pandas as pd

st.set_page_config(page_title="Análisis de Rutas", layout="centered")
st.title("🗺️ Análisis de Rutas y Terreno")
st.markdown("Sube uno o más archivos `.kml`, `.gpx` o `.geojson` para generar mapas de calor o comparar perfiles de elevación.")

st.info("""
**📌 Instrucciones para exportar desde Google Earth Pro:**

Al guardar tu ruta como archivo `.kml`, asegúrate de elegir la opción adecuada según tu caso:

- **Clamped to ground** (recomendado si trazaste el Path manualmente):  
  El archivo no incluirá altitud. La app obtendrá automáticamente la elevación real desde un modelo topográfico.

- **Absolute** (recomendado si ya tienes datos de altitud precisos):  
  El archivo incluirá coordenadas 3D reales y se usará directamente sin corrección.

Puedes usar cualquiera de las dos opciones. La app detectará si falta altitud y la completará automáticamente.
""")

# Selector de modo
modo = st.radio("¿Qué deseas hacer?", ["🌄 Generar mapa de calor", "📊 Comparar perfiles"], index=0)

# Carga de archivos
uploaded_files = st.file_uploader("📂 Subir archivos", type=["kml", "gpx", "geojson"], accept_multiple_files=True)

# Función para obtener ubicación desde coordenadas
def obtener_ubicacion(lat: float, lon: float) -> str:
    api_key = "f8e557c5994849b1b46c41abc3095126"
    url = f"https://api.geoapify.com/v1/geocode/reverse?lat={lat}&lon={lon}&apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            props = data["features"][0]["properties"]
            ciudad = props.get("city", "")
            estado = props.get("state", "")
            pais = props.get("country", "")
            return f"{ciudad}, {estado}, {pais}".strip(", ")
        else:
            return "Ubicación desconocida"
    except Exception:
        return "Ubicación desconocida"

# Funciones para extraer coordenadas
def extraer_coords_kml(text: str):
    try:
        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        root = ET.fromstring(text)
        coords_total = []
        for coord_elem in root.findall(".//kml:LineString/kml:coordinates", ns):
            raw_coords = coord_elem.text.strip().split()
            for triplet in raw_coords:
                parts = triplet.strip().split(",")
                if len(parts) == 3:
                    x, y, z = map(float, parts)
                    coords_total.append((x, y, z))
        return coords_total if coords_total else None
    except Exception:
        return None

def extraer_coords_gpx(text: str):
    try:
        gpx = gpxpy.parse(text)
        coords = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    coords.append((point.longitude, point.latitude, point.elevation or 0))
        return coords if coords else None
    except Exception:
        return None

def extraer_coords_geojson(text: str):
    try:
        data = json.loads(text)
        coords = []
        for feature in data["features"]:
            geom = feature["geometry"]
            if geom["type"] == "LineString":
                for pt in geom["coordinates"]:
                    if len(pt) == 3:
                        coords.append(tuple(pt))
                    elif len(pt) == 2:
                        coords.append((pt[0], pt[1], 0))
        return coords if coords else None
    except Exception:
        return None

def obtener_altitudes(coords_2d: list[tuple[float, float]]):
    batch_size = 100
    coords_con_altura = []
    for i in range(0, len(coords_2d), batch_size):
        batch = coords_2d[i:i+batch_size]
        loc_str = "|".join([f"{lat},{lon}" for lon, lat in batch])
        url = f"https://api.opentopodata.org/v1/srtm90m?locations={loc_str}"
        response = requests.get(url)
        if response.status_code != 200:
            return None
        data = response.json()
        for j, result in enumerate(data["results"]):
            lon, lat = batch[j]
            z = result["elevation"]
            coords_con_altura.append((lon, lat, z))
        time.sleep(1)
    return coords_con_altura

def calcular_distancias(coords: list[tuple[float, float, float]]):
    dists = [0]
    for i in range(1, len(coords)):
        x0, y0 = coords[i-1][:2]
        x1, y1 = coords[i][:2]
        dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        dists.append(dists[-1] + dist)
    return dists

def calcular_pendientes(coords: list[tuple[float, float, float]]):
    pendientes = []
    for i in range(1, len(coords)):
        x0, y0, z0 = coords[i - 1]
        x1, y1, z1 = coords[i]
        dz = z1 - z0
        dx = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if dx > 0:
            pendientes.append(100 * dz / dx)
    return pendientes

def estimar_volumen(grid_z, grid_x, grid_y, ref_altura: float):
    dx = (grid_x[1, 0] - grid_x[0, 0])
    dy = (grid_y[0, 1] - grid_y[0, 0])
    area_celda = dx * dy
    diferencia = grid_z - ref_altura
    volumen_corte = np.sum(diferencia[diferencia > 0]) * area_celda
    volumen_relleno = np.sum(np.abs(diferencia[diferencia < 0])) * area_celda
    return volumen_corte, volumen_relleno

if uploaded_files:
    if modo == "🌄 Generar mapa de calor":
        archivo = uploaded_files[0]
        nombre_archivo = os.path.splitext(archivo.name)[0]
        contenido = archivo.read().decode("utf-8")
        extension = os.path.splitext(archivo.name)[1].lower()

        if extension == ".kml":
            coords_3d = extraer_coords_kml(contenido)
        elif extension == ".gpx":
            coords_3d = extraer_coords_gpx(contenido)
        elif extension == ".geojson":
            coords_3d = extraer_coords_geojson(contenido)
        else:
            st.error("❌ Tipo de archivo no soportado.")
            st.stop()

        if not coords_3d:
            st.error("❌ No se encontraron coordenadas válidas.")
            st.stop()

        if all(z == 0 for _, _, z in coords_3d):
            st.warning("⚠️ El archivo no contiene altitud. Obteniendo elevación real automáticamente...")
            coords_2d = [(x, y) for x, y, _ in coords_3d]
            coords_3d = obtener_altitudes(coords_2d)
            if not coords_3d:
                st.error("❌ No se pudo obtener altitud.")
                st.stop()
            st.success("✅ Altitud obtenida correctamente.")

        x, y, z = zip(*coords_3d)
        grid_res = st.slider("📏 Resolución de la grilla", 50, 500, 200, step=50)
        num_curvas = st.slider("🌀 Número de curvas de nivel", 5, 50, 15)

        grid_x, grid_y = np.mgrid[min(x):max(x):complex(grid_res), min(y):max(y):complex(grid_res)]
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        lat0, lon0 = coords_3d[0][1], coords_3d[0][0]
        ubicacion = obtener_ubicacion(lat0, lon0)

        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)),
                            origin='lower', cmap='terrain', alpha=0.8)
        contours = ax.contour(grid_x, grid_y, grid_z, levels=num_curvas, colors='black', linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8)
        plt.colorbar(heatmap, ax=ax, label="Altitud (m)")
        ax.set_title(f"Mapa de Elevación Interpolado\nUbicación: {ubicacion}", fontsize=12)
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("📥 Descargar imagen PNG", data=buf.getvalue(), file_name="mapa_calor.png", mime="image/png")

    elif modo == "📊 Comparar perfiles":
        if st.button("🚀 Comparar rutas"):
            fig, ax = plt.subplots(figsize=(10, 6))
            stats: list[dict] = []
            ref_altura: float = st.number_input("📏 Altitud de referencia para volumen (m)", value=0.0)
            ubicacion_global = "Ubicación desconocida"

            for archivo in uploaded_files:
                nombre_archivo = os.path.splitext(archivo.name)[0]
                contenido = archivo.read().decode("utf-8")
                extension = os.path.splitext(archivo.name)[1].lower()

                if extension == ".kml":
                    coords_3d = extraer_coords_kml(contenido)
                elif extension == ".gpx":
                    coords_3d = extraer_coords_gpx(contenido)
                elif extension == ".geojson":
                    coords_3d = extraer_coords_geojson(contenido)
                else:
                    st.warning(f"❌ Tipo de archivo no soportado: {archivo.name}")
                    continue

                if not coords_3d:
                    st.warning(f"⚠️ No se encontraron coordenadas en {archivo.name}")
                    continue

                if all(z == 0 for _, _, z in coords_3d):
                    coords_2d = [(x, y) for x, y, _ in coords_3d]
                    coords_3d = obtener_altitudes(coords_2d)
                    if not coords_3d:
                        st.warning(f"⚠️ No se pudo obtener altitud para {archivo.name}")
                        continue
                    st.success(f"✅ Altitud obtenida para {archivo.name}")

                lat0, lon0 = coords_3d[0][1], coords_3d[0][0]
                ubicacion = obtener_ubicacion(lat0, lon0)
                if ubicacion_global == "Ubicación desconocida":
                    ubicacion_global = ubicacion

                distancias = calcular_distancias(coords_3d)
                elevaciones = [z for _, _, z in coords_3d]
                ax.plot(distancias, elevaciones, label=f"{nombre_archivo} ({ubicacion})")

                pendientes = calcular_pendientes(coords_3d)
                pendiente_media = round(np.mean(np.abs(pendientes)), 2)
                pendiente_max = round(np.max(np.abs(pendientes)), 2)

                x, y, z = zip(*coords_3d)
                grid_x, grid_y = np.mgrid[min(x):max(x):complex(100), min(y):max(y):complex(100)]
                grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
                vol_corte, vol_relleno = estimar_volumen(grid_z, grid_x, grid_y, ref_altura)

                stats.append({
                    "Ruta": nombre_archivo,
                    "Puntos": len(coords_3d),
                    "Distancia (u)": round(distancias[-1], 2),
                    "Altitud mín (m)": round(min(elevaciones), 2),
                    "Altitud máx (m)": round(max(elevaciones), 2),
                    "Ganancia total (m)": round(sum(np.diff(elevaciones)[np.diff(elevaciones) > 0]), 2),
                    "Pendiente media (%)": pendiente_media,
                    "Pendiente máx (%)": pendiente_max,
                    "Volumen corte (u³)": round(vol_corte, 2),
                    "Volumen relleno (u³)": round(vol_relleno, 2)
                })

            if stats:
                ax.set_title(f"Comparación de Perfiles de Elevación\nUbicación: {ubicacion_global}", fontsize=12)
                ax.set_xlabel("Distancia acumulada (unidades)")
                ax.set_ylabel("Altitud (m)")
                ax.legend()
                st.pyplot(fig)

                df_stats = pd.DataFrame(stats)
                st.markdown("### 📊 Estadísticas comparativas")
                st.dataframe(df_stats)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300)
                st.download_button("📥 Descargar gráfico PNG", data=buf.getvalue(), file_name="comparacion_perfiles.png", mime="image/png")