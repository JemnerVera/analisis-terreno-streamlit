import os
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import streamlit as st
import requests
import time
import xml.etree.ElementTree as ET
import gpxpy
import pandas as pd
from matplotlib.lines import Line2D

st.set_page_config(page_title="Análisis de Rutas", layout="centered")
st.title("🗺️ Análisis de Rutas y Terreno")
st.markdown("Sube uno o más archivos `.kml`, `.gpx` o `.geojson` para generar mapas de calor o comparar perfiles de elevación.")

st.info("""
**📌 Instrucciones para exportar desde Google Earth Pro:**

Al guardar tu ruta como archivo `.kml`, asegúrate de elegir la opción adecuada según tu caso:

- **Clamped to ground**: recomendado si trazaste el Path manualmente. La app calculará la altitud automáticamente.
- **Absolute**: recomendado si ya tienes datos de altitud precisos. La app usará las altitudes del archivo.

Puedes usar cualquiera de las dos opciones. La app detectará si falta altitud y la completará automáticamente.
""")

modo = st.radio("¿Qué deseas hacer?", ["🌄 Generar mapa de calor", "📊 Comparar perfiles"], index=0)
uploaded_files = st.file_uploader("📂 Subir archivos", type=["kml", "gpx", "geojson"], accept_multiple_files=True)

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

def filtrar_puntos_cercanos(coords, umbral_metros=1.0):
    coords_filtrados = [coords[0]]
    for pt in coords[1:]:
        x0, y0 = coords_filtrados[-1][:2]
        x1, y1 = pt[:2]
        dx = (x1 - x0) * 111320
        dy = (y1 - y0) * 110540
        dist = np.sqrt(dx**2 + dy**2)
        if dist >= umbral_metros:
            coords_filtrados.append(pt)
    return coords_filtrados

def crear_mascara_validez(grid_z):
    mask = ~np.isnan(grid_z)
    return gaussian_filter(mask.astype(float), sigma=2) > 0.1

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

        coords_3d = filtrar_puntos_cercanos(coords_3d, umbral_metros=1.0)

        x, y, z = zip(*coords_3d)
        grid_res = st.slider("📏 Resolución de la grilla", 50, 500, 200, step=50)
        num_curvas = st.slider("🌀 Número de curvas de nivel", 5, 50, 15)

        grid_x, grid_y = np.mgrid[min(x):max(x):complex(grid_res), min(y):max(y):complex(grid_res)]
        grid_z_raw = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        grid_z = gaussian_filter(grid_z_raw, sigma=1)
        mascara_valida = crear_mascara_validez(grid_z_raw)

        # Calcular pendiente local y umbral dinámico
        grad_y, grad_x = np.gradient(grid_z)
        pendiente = np.sqrt(grad_x**2 + grad_y**2)
        umbral_dinamico = max(np.percentile(pendiente[mascara_valida], 60), 0.00005)
        mascara_pendiente = pendiente > umbral_dinamico

        lat0, lon0 = coords_3d[0][1], coords_3d[0][0]
        ubicacion = obtener_ubicacion(lat0, lon0)

        ancho = abs(max(x) - min(x)) * 111320
        alto = abs(max(y) - min(y)) * 110540
        area_m2 = ancho * alto
        area_ha = area_m2 / 10000

        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.imshow(np.where(mascara_valida, grid_z, np.nan).T,
                            extent=(min(x), max(x), min(y), max(y)),
                            origin='lower', cmap='terrain', alpha=0.9)

        legend_elements = []

        if num_curvas > 0:
            curva_color = 'black'
            curva_alpha = 0.8
            curva_linewidth = 0.8
            curva_label = "Curvas de nivel (zonas relevantes)"

            curvas_z = np.where(mascara_valida & (mascara_pendiente | (pendiente > 0.00001)), grid_z, np.nan)
            contours = ax.contour(grid_x, grid_y, curvas_z,
                                  levels=num_curvas,
                                  colors=curva_color,
                                  linewidths=curva_linewidth,
                                  alpha=curva_alpha)
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
            legend_elements.append(Line2D([0], [0], color=curva_color, lw=1, label=curva_label))

        plt.colorbar(heatmap, ax=ax, label="Altitud (m)")
        ax.set_title(f"Mapa de Elevación Interpolado\nUbicación: {ubicacion}\nÁrea aproximada: {area_ha:.2f} ha", fontsize=12)
        ax.set_xlabel("Longitud (°)")
        ax.set_ylabel("Latitud (°)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

        escala_m = 100
        legend_elements.append(Line2D([0], [0], color='black', lw=2, label=f'Escala: {escala_m} m'))
        legend_elements.append(Line2D([0], [0], color='none', label=f'Área: {area_ha:.2f} ha'))
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8, frameon=True)

        ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.85),
                    arrowprops=dict(facecolor='black', width=2, headwidth=8),
                    ha='center', va='center', fontsize=10, xycoords='axes fraction')

        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        st.download_button("📥 Descargar imagen PNG", data=buf.getvalue(), file_name="mapa_calor.png", mime="image/png")

    elif modo == "📊 Comparar perfiles":
        st.warning("⚠️ Esta sección aún no ha sido modificada. ¿Quieres que la mejoremos también?")
