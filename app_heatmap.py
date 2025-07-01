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
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

st.set_page_config(page_title="Análisis de Terreno", layout="centered")
st.title("🌍 Análisis de Terreno y Perfiles")
st.markdown("Sube uno o más archivos `.kml`, `.gpx` o `.geojson` para generar mapas topográficos, analizar perfiles o estimar movimiento de tierra.")

modo = st.radio("¿Qué deseas hacer?", [
    "🗺️ Generar mapa topográfico",
    "📈 Analizar perfiles",
    "⛏️ Movimiento de tierra aproximado"
], index=0)

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
    # 🗺️ GENERAR MAPA TOPOGRÁFICO
    if modo == "🗺️ Generar mapa topográfico":
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
        num_curvas = st.slider("🌀 Número de curvas de nivel", 5, 50, 20)

        grid_x, grid_y = np.mgrid[min(x):max(x):complex(grid_res), min(y):max(y):complex(grid_res)]
        grid_z_raw = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        grid_z = gaussian_filter(grid_z_raw, sigma=1)
        mascara_valida = crear_mascara_validez(grid_z_raw)

        grad_y, grad_x = np.gradient(grid_z)
        pendiente = np.sqrt(grad_x**2 + grad_y**2)
        umbral_dinamico = max(np.percentile(pendiente[mascara_valida], 60), 0.00005)
        mascara_pendiente = pendiente > umbral_dinamico

        curvas_z = np.where(mascara_valida & (mascara_pendiente | (pendiente > 0.00001)), grid_z, np.nan)

        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.imshow(np.where(mascara_valida, grid_z, np.nan).T,
                            extent=(min(x), max(x), min(y), max(y)),
                            origin='lower', cmap='terrain', alpha=0.9)

        if num_curvas > 0:
            contours = ax.contour(grid_x, grid_y, curvas_z,
                                  levels=num_curvas, colors='black', linewidths=0.8, alpha=0.8)
            ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

        plt.colorbar(heatmap, ax=ax, label="Altitud (m)")
        ax.set_title("Mapa Topográfico Interpolado")
        ax.set_xlabel("Longitud (°)")
        ax.set_ylabel("Latitud (°)")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    # 📈 ANALIZAR PERFILES
    elif modo == "📈 Analizar perfiles":
        fig = go.Figure()
        resumen = []
        colores = px.colors.qualitative.Plotly

        for idx, archivo in enumerate(uploaded_files):
            nombre = os.path.splitext(archivo.name)[0]
            contenido = archivo.read().decode("utf-8")
            extension = os.path.splitext(archivo.name)[1].lower()

            if extension == ".kml":
                coords_3d = extraer_coords_kml(contenido)
            elif extension == ".gpx":
                coords_3d = extraer_coords_gpx(contenido)
            elif extension == ".geojson":
                coords_3d = extraer_coords_geojson(contenido)
            else:
                continue

            if not coords_3d:
                continue

            if all(z == 0 for _, _, z in coords_3d):
                coords_2d = [(x, y) for x, y, _ in coords_3d]
                coords_3d = obtener_altitudes(coords_2d)
                if not coords_3d:
                    continue

            coords_3d = filtrar_puntos_cercanos(coords_3d, umbral_metros=1.0)
            x, y, z = zip(*coords_3d)
            dists = [0]
            for i in range(1, len(x)):
                dx = (x[i] - x[i-1]) * 111320
                dy = (y[i] - y[i-1]) * 110540
                d = np.sqrt(dx**2 + dy**2)
                dists.append(dists[-1] + d)

            elev = np.array(z)
            dist = np.array(dists)
            desnivel = np.sum(np.abs(np.diff(elev)))
            pendiente_max = np.max(np.abs(np.diff(elev) / np.diff(dist + 1e-6))) * 100

            fig.add_trace(go.Scatter(
                x=dist,
                y=elev,
                mode='lines',
                name=nombre,
                line=dict(color=colores[idx % len(colores)]),
                hovertemplate=f"<b>{nombre}</b><br>Distancia: %{{x:.0f}} m<br>Altitud: %{{y:.1f}} m"
            ))

            resumen.append({
                "Ruta": nombre,
                "Distancia (km)": dist[-1] / 1000,
                "Desnivel acumulado (m)": desnivel,
                "Pendiente máxima (%)": pendiente_max
            })

        fig.update_layout(
            title="Comparación de Perfiles de Elevación",
            xaxis_title="Distancia (m)",
            yaxis_title="Altitud (m)",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        df_resumen = pd.DataFrame(resumen)
        st.markdown("### 📊 Resumen por ruta")
        st.dataframe(df_resumen.style.format({
            "Distancia (km)": "{:.2f}",
            "Desnivel acumulado (m)": "{:.1f}",
            "Pendiente máxima (%)": "{:.1f}"
        }))

        csv_buffer = StringIO()
        df_resumen.to_csv(csv_buffer, index=False)
        st.download_button("📥 Descargar resumen CSV", data=csv_buffer.getvalue(),
                           file_name="resumen_perfiles.csv", mime="text/csv")
        
    # ⛏️ MOVIMIENTO DE TIERRA APROXIMADO (pendiente real + visualización corregida)
    elif modo == "⛏️ Movimiento de tierra aproximado":
        tipo_cultivo = st.selectbox("🌱 Tipo de cultivo", ["Uva", "Arándano"])
        pendiente_max = 10 if tipo_cultivo == "Uva" else 5  # en %

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
            coords_2d = [(x, y) for x, y, _ in coords_3d]
            coords_3d = obtener_altitudes(coords_2d)
            if not coords_3d:
                st.error("❌ No se pudo obtener altitud.")
                st.stop()

        coords_3d = filtrar_puntos_cercanos(coords_3d, umbral_metros=1.0)
        x, y, z = zip(*coords_3d)

        grid_res = st.slider("📏 Resolución de la grilla", 50, 500, 200, step=50)
        grid_x, grid_y = np.mgrid[min(x):max(x):complex(grid_res), min(y):max(y):complex(grid_res)]
        grid_z_raw = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        grid_z = gaussian_filter(grid_z_raw, sigma=1)
        mascara_valida = crear_mascara_validez(grid_z_raw)

        z_ref = np.nanmean(grid_z[mascara_valida])
        delta_z = grid_z - z_ref

        # Cálculo de pendiente real en % usando distancias en metros
        dx_m = (max(x) - min(x)) * 111320 / grid_res
        dy_m = (max(y) - min(y)) * 110540 / grid_res
        grad_y, grad_x = np.gradient(grid_z, dy_m, dx_m)
        pendiente = np.sqrt(grad_x**2 + grad_y**2) * 100  # en %

        # Clasificación
        mapa = np.full_like(grid_z, np.nan)
        zona_nivelada = (pendiente <= pendiente_max)
        zona_corte = (~zona_nivelada) & (delta_z > 0)
        zona_relleno = (~zona_nivelada) & (delta_z < 0)
        mapa[zona_nivelada] = 0
        mapa[zona_corte] = 2
        mapa[zona_relleno] = -2

        # Cálculo de volúmenes
        area_celda = dx_m * dy_m
        volumen_corte = np.nansum(np.where(zona_corte, delta_z * area_celda, 0))
        volumen_relleno = np.nansum(np.where(zona_relleno, -delta_z * area_celda, 0))

        # Cálculo de áreas en hectáreas
        area_nivelado = np.count_nonzero(zona_nivelada) * area_celda / 10_000
        area_corte = np.count_nonzero(zona_corte) * area_celda / 10_000
        area_relleno = np.count_nonzero(zona_relleno) * area_celda / 10_000

        # Visualización
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(["blue", "gray", "red"])  # relleno, nivelado, corte
        bounds = [-2.5, -0.5, 0.5, 2.5]  # 3 regiones → 4 límites
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mapa.T, extent=(min(x), max(x), min(y), max(y)),
                       origin='lower', cmap=cmap, norm=norm, alpha=0.85)
        ax.set_title(f"Zonas de Corte, Relleno y Nivelado (Pendiente ≤ {pendiente_max}% para {tipo_cultivo})")
        ax.set_xlabel("Longitud (°)")
        ax.set_ylabel("Latitud (°)")
        ax.grid(True, linestyle='--', alpha=0.5)

        legend_elements = [
            Line2D([0], [0], color='red', lw=4, label='Corte severo'),
            Line2D([0], [0], color='blue', lw=4, label='Relleno severo'),
            Line2D([0], [0], color='gray', lw=4, label=f'Nivelado (pendiente ≤ {pendiente_max}%)'),
            Line2D([0], [0], color='none', label=f'Cota de referencia: {z_ref:.2f} m'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8, frameon=True)

        st.pyplot(fig)

        # Métricas
        st.markdown("### 📊 Estimación de movimiento de tierra")
        col1, col2 = st.columns(2)
        col1.metric("🟥 Volumen de corte", f"{volumen_corte:,.0f} m³")
        col2.metric("🟦 Volumen de relleno", f"{volumen_relleno:,.0f} m³")

        st.markdown("### 📐 Área por tipo de zona (hectáreas)")
        col3, col4, col5 = st.columns(3)
        col3.metric("🟦 Área de relleno", f"{area_relleno:,.2f} ha")
        col4.metric("🟥 Área de corte", f"{area_corte:,.2f} ha")
        col5.metric("⚪ Área nivelada", f"{area_nivelado:,.2f} ha")

        # Gráfico circular
        labels = ['Nivelado', 'Corte', 'Relleno']
        areas = [area_nivelado, area_corte, area_relleno]
        colors = ['gray', 'red', 'blue']

        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(areas, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax_pie.axis('equal')
        ax_pie.set_title("Distribución de áreas por tipo de terreno")
        st.pyplot(fig_pie)
