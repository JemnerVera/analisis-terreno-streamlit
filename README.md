# Análisis de Rutas y Terreno

Esta aplicación web permite generar mapas de calor interpolados y comparar perfiles de elevación a partir de archivos `.kml`, `.gpx` o `.geojson`. Es ideal para evaluar rutas, analizar pendientes y estimar volúmenes de corte y relleno en terrenos.

## ¿Cómo usar la app?

1. Abre la app en línea: [https://analisis-terreno.streamlit.app](https://analisis-terreno.streamlit.app) *(reemplaza con tu URL real)*
2. Sube uno o más archivos de ruta en formato `.kml`, `.gpx` o `.geojson`
3. Elige entre dos modos:
   - **Generar mapa de calor**: visualiza la elevación interpolada con curvas de nivel
   - **Comparar perfiles**: analiza altitud, pendiente y volumen entre rutas
4. Descarga los resultados como imagen `.png`

## Exportar desde Google Earth Pro

Al guardar tu ruta como archivo `.kml`, elige una de estas opciones:

- **Clamped to ground**: recomendado si trazaste el Path manualmente. La app calculará la altitud automáticamente.
- **Absolute**: recomendado si ya tienes datos de altitud precisos. La app usará las altitudes del archivo.

## Tecnologías utilizadas

- [Streamlit](https://streamlit.io/)
- Python (NumPy, Matplotlib, Pandas, SciPy)
- [OpenTopoData](https://www.opentopodata.org/) para altitud
- [Geoapify](https://www.geoapify.com/) para geolocalización inversa

## Licencia

Este proyecto es de uso libre para fines educativos y de análisis. Puedes adaptarlo según tus necesidades.

---