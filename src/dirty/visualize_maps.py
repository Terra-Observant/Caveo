"""
visualize_maps.py - Создание карт для визуализации гексагональной сетки H3

Обновлённая версия с увеличенным масштабом для лучшей видимости ячеек.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import os
import sys
import numpy as np

# Константы
CRS_WGS84 = "EPSG:4326"
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
PROJ_DOCS = os.path.join(PROJECT_ROOT, "proj_docs")

# Цвета районов
DISTRICT_COLORS = {
    1: "#3498db",  # Адлерский
    2: "#27ae60",  # Хостинский
    3: "#e67e22",  # Центральный
    4: "#9b59b6",  # Лазаревский
}

DISTRICT_NAMES = {
    1: "Адлерский",
    2: "Хостинский",
    3: "Центральный",
    4: "Лазаревский",
}

# Настройки для увеличенного масштаба
FIG_SIZE_OVERVIEW = (24, 18)  # Увеличенный размер для общей карты
FIG_SIZE_DISTRICT = (20, 16)  # Увеличенный размер для карт районов
DPI = 300  # Высокое разрешение


def load_grid_data():
    """Загружает сетку в WGS84."""
    grid_path = os.path.join(DATA_PROCESSED, "sochi_hex_grid_wgs84.geojson")
    print(f"Загрузка сетки из {grid_path}...")
    gdf = gpd.read_file(grid_path)
    print(f"  Загружено {len(gdf)} ячеек")
    return gdf


def load_districts():
    """Загружает границы районов."""
    districts_path = os.path.join(
        PROJECT_ROOT, "data", "external", "sochi_districts.geojson"
    )
    print(f"Загрузка границ районов...")
    gdf = gpd.read_file(districts_path)
    return gdf


def plot_overview_map(grid_gdf, districts_gdf, output_path=None):
    """
    Создаёт общую карту территории с увеличенным масштабом.
    """
    if output_path is None:
        output_path = os.path.join(PROJ_DOCS, "map_districts_overview.png")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_OVERVIEW)

    # Рисуем ячейки сетки с более тонкими границами
    grid_gdf.plot(
        ax=ax,
        column="color",
        linewidth=0.2,
        edgecolor="#666666",
        legend=False,
        alpha=0.9,
    )

    # Рисуем границы районов
    districts_gdf.to_crs(CRS_WGS84).plot(
        ax=ax, facecolor="none", edgecolor="black", linewidth=3
    )

    # Легенда
    legend_patches = []
    for district_id, color in DISTRICT_COLORS.items():
        patch = mpatches.Patch(
            color=color,
            label=f"{DISTRICT_NAMES.get(district_id, 'Не определён')}",
            alpha=0.8,
            edgecolor="#333333",
            linewidth=1.5,
        )
        legend_patches.append(patch)

    legend = ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=16,
        title="Районы",
        title_fontsize=18,
        framealpha=0.95,
        edgecolor="#333333",
    )

    total_cells = len(grid_gdf)
    ax.set_title(
        f"Большой Сочи: гексагональная сетка H3 (resolution {grid_gdf['h3_resolution'].iloc[0] if 'h3_resolution' in grid_gdf.columns else 9})\n"
        f"Всего ячеек: {total_cells:,}",
        fontsize=22,
        fontweight="bold",
        pad=30,
    )

    ax.set_xlabel("Долгота", fontsize=16, labelpad=10)
    ax.set_ylabel("Широта", fontsize=16, labelpad=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Общая карта сохранена: {output_path}")
    print(f"  Размер: {FIG_SIZE_OVERVIEW}, DPI: {DPI}")


def plot_district_map(grid_gdf, districts_gdf, district_id, output_path=None):
    """
    Создаёт увеличенную карту отдельного района.
    """
    district_name = DISTRICT_NAMES.get(district_id, f"Район {district_id}")

    if output_path is None:
        # Английские имена для файлов
        eng_names = {1: "adler", 2: "khosta", 3: "central", 4: "lazarevsky"}
        filename = f"map_district_{eng_names.get(district_id, district_id)}.png"
        output_path = os.path.join(PROJ_DOCS, filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_DISTRICT)

    # Фильтруем ячейки района
    district_mask = grid_gdf["district_id"] == district_id
    district_grid = grid_gdf[district_mask].copy()

    # Соседние районы - приглушённые
    other_mask = grid_gdf["district_id"] != district_id
    other_grid = grid_gdf[other_mask].copy()

    # Рисуем другие районы серым
    other_grid["color"] = "#d5d5d5"
    other_grid.plot(
        ax=ax, column="color", linewidth=0.3, edgecolor="#999999", alpha=0.4
    )

    # Рисуем ячейки выбранного района
    district_color = DISTRICT_COLORS.get(district_id, "#95a5a6")
    district_grid["color"] = district_color
    district_grid.plot(
        ax=ax, column="color", linewidth=0.5, edgecolor="#444444", alpha=0.9
    )

    # Граница района
    district_boundary = districts_gdf[
        districts_gdf["district_id"] == district_id
    ].to_crs(CRS_WGS84)

    if len(district_boundary) > 0:
        district_boundary.plot(
            ax=ax, facecolor="none", edgecolor="black", linewidth=3.5
        )

    # Статистика
    cell_count = len(district_grid)
    area_km2 = district_grid["area_km2"].sum()

    ax.set_title(
        f"{district_name} район: {cell_count:,} ячеек H3\n"
        f"Площадь: {area_km2:.2f} км²",
        fontsize=22,
        fontweight="bold",
        pad=30,
    )

    ax.set_xlabel("Долгота", fontsize=16, labelpad=10)
    ax.set_ylabel("Широта", fontsize=16, labelpad=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.15, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"Карта района '{district_name}' сохранена: {output_path}")
    print(f"  Размер: {FIG_SIZE_DISTRICT}, DPI: {DPI}")


def create_all_maps():
    """Создаёт все карты."""
    # Загрузка данных
    grid_gdf = load_grid_data()
    districts_gdf = load_districts()

    # Добавляем district_id к районам если нет
    if "district_id" not in districts_gdf.columns:
        district_mapping = {}
        for idx, row in districts_gdf.iterrows():
            name = str(row.get("name", "")).lower()
            if "адлер" in name:
                district_mapping[idx] = 1
            elif "хост" in name:
                district_mapping[idx] = 2
            elif "центр" in name or "central" in name:
                district_mapping[idx] = 3
            elif "лазарев" in name:
                district_mapping[idx] = 4
            else:
                district_mapping[idx] = None
        districts_gdf["district_id"] = districts_gdf.index.map(district_mapping)

    # Общая карта
    print("\nСоздание общей карты...")
    plot_overview_map(grid_gdf, districts_gdf)

    # Карты отдельных районов
    for district_id in [1, 2, 3, 4]:
        print(f"\nСоздание карты района {district_id}...")
        plot_district_map(grid_gdf, districts_gdf, district_id)

    print("\nВсе карты созданы!")


if __name__ == "__main__":
    create_all_maps()
