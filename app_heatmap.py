        # 📍 Obtener ubicación real desde la primera coordenada
        lat0, lon0 = y[0], x[0]
        nombre_lugar = obtener_ubicacion(lat0, lon0)

        # 🧮 Calcular área aproximada (en hectáreas)
        from shapely.geometry import LineString
        linea = LineString([(x_, y_) for x_, y_ in zip(x, y)])
        area_aprox_ha = linea.convex_hull.area * 111320 * 110540 / 10_000

        # 🏷️ Nombre del proyecto (puedes personalizarlo)
        nombre_proyecto = "Agrícola Andrea"

        # 🧭 Título completo dentro del mapa (parte superior)
        titulo_mapa = f"Mapa Topográfico Interpolado – {nombre_proyecto}\n{nombre_lugar}\nÁrea aproximada: {area_aprox_ha:,.2f} ha"
        ax.text(0.5, 1.05, titulo_mapa, transform=ax.transAxes,
                fontsize=11, ha='center', va='bottom', weight='bold')

        # 🗂️ Leyenda textual dentro del mapa (esquina superior izquierda)
        leyenda = f"Curvas de nivel\nEscala: {escala_m:.0f} m\nÁrea: {area_aprox_ha:,.2f} ha"
        ax.text(0.01, 0.99, leyenda, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

        # 🎨 Estética final
        plt.colorbar(heatmap, ax=ax, label="Altitud (m)")
        ax.set_xlabel("Longitud (°)")
        ax.set_ylabel("Latitud (°)")
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)
