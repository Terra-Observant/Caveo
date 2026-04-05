"""
create_hex_grid.py - Создание гексагональной сетки H3 для Большого Сочи

Использует библиотеку H3 (Uber) для построения сетки.
Resolution 9: площадь ячейки ~0.105 км² (~105 000 м²)
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
from shapely import wkt
import h3
import os
import sys

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dirty"))
from geo_utils import load_and_validate, reproject_gdf, validate_geometry


# Константы H3
H3_RESOLUTION = 9  # Площадь ячейки ~0.105 км² (~105 000 м²)
CRS_UTM = "EPSG:32637"  # UTM Zone 37N для метрических расчётов
CRS_WGS84 = "EPSG:4326"  # WGS84 для хранения

# Цвета районов
DISTRICT_COLORS = {
    1: "#3498db",  # Адлерский - синий
    2: "#27ae60",  # Хостинский - зелёный
    3: "#e67e22",  # Центральный - оранжевый
    4: "#9b59b6",  # Лазаревский - фиолетовый
    None: "#95a5a6",  # Не определён - серый
}

# Пути к данным
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_EXTERNAL = os.path.join(PROJECT_ROOT, "data", "external")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
PROJ_DOCS = os.path.join(PROJECT_ROOT, "proj_docs")


def h3_cell_to_polygon(cell_id: str) -> Polygon:
    """
    Конвертирует H3 cell_id в Shapely Polygon.

    Parameters
    ----------
    cell_id : str
        H3 cell identifier

    Returns
    -------
    Polygon
        Shapely Polygon geometry
    """
    boundary = h3.cell_to_boundary(cell_id)
    # boundary возвращает список [lat, lng] пар, нужно конвертировать в (x, y) = (lng, lat)
    coords = [(lng, lat) for lat, lng in boundary]
    return Polygon(coords)


def get_h3_cells_in_polygon(boundary_gdf, resolution: int) -> list:
    """
    Получает все H3 ячейки, покрывающие полигон территории.

    Parameters
    ----------
    boundary_gdf : gpd.GeoDataFrame
        Граница территории в WGS84
    resolution : int
        H3 resolution level

    Returns
    -------
    list
        Список H3 cell_id
    """
    # Объединяем все полигоны в один
    boundary = (
        boundary_gdf.union_all()
        if hasattr(boundary_gdf, "union_all")
        else boundary_gdf.unary_union
    )

    # Получаем bounding box
    minx, miny, maxx, maxy = boundary.bounds

    # Получаем H3 ячейки для bounding box
    # Используем polygon_to_cells для получения ячеек внутри полигона
    # Сначала конвертируем boundary в формат H3 (список контуров)

    # Метод: покрываем bounding box ячейками, затем фильтруем по пересечению с boundary
    bbox_polygon = Polygon(
        [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    )

    # Получаем центроид bounding box для начальной ячейки
    center_lat = (miny + maxy) / 2
    center_lng = (minx + maxx) / 2
    start_cell = h3.latlng_to_cell(center_lat, center_lng, resolution)

    # Вычисляем примерное количество ячеек для покрытия
    # Используем k-ring для покрытия всей территории
    # Для Большого Сочи (~145 км в длину) при res 9 нужно k примерно 200-300
    # Но лучше использовать grid_disk или grid_ring

    # Более надёжный метод: используем polygon_to_cells если доступна
    # Или используем grid_disk с достаточным k

    # Рассчитываем k на основе размера территории
    bbox_diag_km = (
        (maxx - minx) ** 2 + (maxy - miny) ** 2
    ) ** 0.5 * 111  # примерно в км
    # При res 9 ячейка ~0.33 км в диаметре
    k = int(bbox_diag_km / 0.33) + 50

    # Это может быть слишком много, используем более умный подход
    # Используем polyfill (polygon fill)
    try:
        # Для H3 v4 используем polygon_to_cells
        # boundary должен быть в формате geo boundary
        if hasattr(boundary, "exterior"):
            # Single polygon
            coords = [(lat, lng) for lng, lat in boundary.exterior.coords]
            cells = h3.polyfill(coords, resolution, geo_json_conformant=True)
        else:
            raise ValueError("Unknown boundary type")
    except Exception:
        # Fallback: grid_disk с фильтрацией
        cells = set(h3.grid_disk(start_cell, min(k, 500)))

    return list(cells)


def filter_cells_by_boundary(cell_ids: list, boundary_gdf) -> list:
    """
    Фильтрует H3 ячейки, оставляя только те, что пересекаются с территорией.

    Parameters
    ----------
    cell_ids : list
        Список H3 cell_id
    boundary_gdf : gpd.GeoDataFrame
        Граница территории в WGS84

    Returns
    -------
    list
        Отфильтрованный список H3 cell_id
    """
    boundary = (
        boundary_gdf.union_all()
        if hasattr(boundary_gdf, "union_all")
        else boundary_gdf.unary_union
    )

    filtered = []
    coverage_data = {}

    for cell_id in cell_ids:
        poly = h3_cell_to_polygon(cell_id)

        # Проверяем пересечение
        if boundary.intersects(poly):
            # Вычисляем покрытие
            intersection = poly.intersection(boundary)
            if intersection.is_empty:
                continue

            coverage_pct = (intersection.area / poly.area) * 100 if poly.area > 0 else 0

            # Критерий: центроид внутри ИЛИ покрытие >= 10%
            centroid = poly.centroid
            if boundary.contains(centroid) or coverage_pct >= 10:
                filtered.append(cell_id)
                coverage_data[cell_id] = coverage_pct

    return filtered, coverage_data


def assign_districts_to_cells(cell_ids: list, districts_gdf) -> dict:
    """
    Привязывает H3 ячейки к районам по центроиду.

    Parameters
    ----------
    cell_ids : list
        Список H3 cell_id
    districts_gdf : gpd.GeoDataFrame
        Границы районов в WGS84

    Returns
    -------
    dict
        Словарь {cell_id: {'district_id': ..., 'district_name': ...}}
    """
    assignments = {}

    for cell_id in cell_ids:
        # Получаем центроид ячейки
        cell_center = h3.cell_to_latlng(cell_id)
        centroid = Point(cell_center[1], cell_center[0])  # (lng, lat)

        district_id = None
        district_name = None

        # Проверяем каждый район
        for idx, row in districts_gdf.iterrows():
            if row.geometry.contains(centroid):
                district_id = row.get("district_id", None)
                district_name = row.get("name", row.get("district_name", None))

                # Если district_id нет, определяем по имени
                if district_id is None and district_name:
                    name_lower = str(district_name).lower()
                    if "адлер" in name_lower:
                        district_id = 1
                    elif "хост" in name_lower:
                        district_id = 2
                    elif "центр" in name_lower or "central" in name_lower:
                        district_id = 3
                    elif "лазарев" in name_lower:
                        district_id = 4

                break

        assignments[cell_id] = {
            "district_id": district_id,
            "district_name": district_name,
            "color": DISTRICT_COLORS.get(district_id, DISTRICT_COLORS[None]),
        }

    return assignments


def create_hex_grid(boundary_path, districts_path, resolution=H3_RESOLUTION):
    """
    Основная функция создания гексагональной сетки H3.

    Parameters
    ----------
    boundary_path : str
        Путь к файлу границы Большого Сочи
    districts_path : str
        Путь к файлу границ районов
    resolution : int
        H3 resolution level

    Returns
    -------
    tuple
        (hex_gdf, hex_gdf_wgs84) - сетки в UTM и WGS84
    """
    print("Загрузка границ Большого Сочи...")
    boundary_gdf = load_and_validate(boundary_path, CRS_WGS84)

    print("Загрузка границ районов...")
    districts_gdf = load_and_validate(districts_path, CRS_WGS84)

    # Добавляем district_id к районам если его нет
    if "district_id" not in districts_gdf.columns:
        print("Добавление district_id к районам...")
        district_mapping = {}
        for idx, row in districts_gdf.iterrows():
            name = str(row.get("name", row.get("district_name", ""))).lower()
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
        print(f"  Сопоставление районов: {district_mapping}")

    # Получаем H3 ячейки
    print(f"Получение H3 ячеек resolution {resolution}...")
    cell_ids = get_h3_cells_in_polygon(boundary_gdf, resolution)
    print(f"  Получено {len(cell_ids)} ячеек (до фильтрации)")

    # Фильтрация по границе
    print("Фильтрация ячеек по границе территории...")
    filtered_cells, coverage_data = filter_cells_by_boundary(cell_ids, boundary_gdf)
    print(f"  Осталось {len(filtered_cells)} ячеек")

    # Привязка к районам
    print("Привязка ячеек к районам...")
    assignments = assign_districts_to_cells(filtered_cells, districts_gdf)

    # Создание GeoDataFrame
    print("Создание GeoDataFrame...")
    records = []

    for cell_id in filtered_cells:
        # Полигон в WGS84
        poly_wgs84 = h3_cell_to_polygon(cell_id)

        # Центроид
        lat, lng = h3.cell_to_latlng(cell_id)

        # Площадь ячейки в м²
        area_m2 = h3.cell_area(cell_id, unit="m^2")
        area_km2 = area_m2 / 1_000_000

        assignment = assignments.get(cell_id, {})

        records.append(
            {
                "cell_id": cell_id,
                "district_id": assignment.get("district_id"),
                "district_name": assignment.get("district_name"),
                "color": assignment.get("color", DISTRICT_COLORS[None]),
                "centroid_lon": lng,
                "centroid_lat": lat,
                "area_m2": area_m2,
                "area_km2": area_km2,
                "coverage_pct": coverage_data.get(cell_id, 100.0),
                "h3_resolution": resolution,
                "geometry_wgs84": poly_wgs84,
            }
        )

    df = pd.DataFrame(records)

    # GeoDataFrame в WGS84
    hex_gdf_wgs84 = gpd.GeoDataFrame(df, geometry="geometry_wgs84", crs=CRS_WGS84)
    hex_gdf_wgs84 = hex_gdf_wgs84.rename_geometry("geometry")

    # Перепроецируем в UTM для метрических расчётов
    hex_gdf = hex_gdf_wgs84.to_crs(CRS_UTM)
    hex_gdf["centroid_x_utm"] = hex_gdf.geometry.centroid.x
    hex_gdf["centroid_y_utm"] = hex_gdf.geometry.centroid.y
    hex_gdf["hex_side_m"] = (
        np.sqrt(hex_gdf["area_m2"] / (3 * np.sqrt(3) / 2)) / 2
    )  # Примерная сторона

    print(f"\nИтого создано {len(hex_gdf)} ячеек H3 resolution {resolution}")

    return hex_gdf, hex_gdf_wgs84


def save_results(hex_gdf, hex_gdf_wgs84, output_dir=None):
    """
    Сохраняет результаты в различных форматах.
    """
    if output_dir is None:
        output_dir = DATA_PROCESSED

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nСохранение результатов в {output_dir}...")

    # GeoJSON (UTM)
    utm_path = os.path.join(output_dir, "sochi_hex_grid.geojson")
    hex_gdf.to_file(utm_path, driver="GeoJSON")
    print(f"  ✓ {utm_path}")

    # GeoJSON (WGS84)
    wgs84_path = os.path.join(output_dir, "sochi_hex_grid_wgs84.geojson")
    hex_gdf_wgs84.to_file(wgs84_path, driver="GeoJSON")
    print(f"  ✓ {wgs84_path}")

    # GeoPackage
    gpkg_path = os.path.join(output_dir, "sochi_hex_grid.gpkg")
    hex_gdf.to_file(gpkg_path, driver="GPKG")
    print(f"  ✓ {gpkg_path}")

    # CSV (атрибуты без геометрии)
    csv_path = os.path.join(output_dir, "sochi_hex_grid_attributes.csv")
    cols_to_save = [
        "cell_id",
        "h3_resolution",
        "district_id",
        "district_name",
        "color",
        "centroid_lon",
        "centroid_lat",
        "centroid_x_utm",
        "centroid_y_utm",
        "area_m2",
        "area_km2",
        "coverage_pct",
        "hex_side_m",
    ]
    hex_gdf[cols_to_save].to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")


def print_statistics(hex_gdf):
    """
    Выводит статистику по районам.
    """
    print("\n" + "=" * 70)
    print("СВОДНАЯ СТАТИСТИКА ПО РАЙОНАМ")
    print("=" * 70)

    stats = []
    for district_id, district_name in [
        (1, "Адлерский"),
        (2, "Хостинский"),
        (3, "Центральный"),
        (4, "Лазаревский"),
    ]:
        mask = hex_gdf["district_id"] == district_id
        count = mask.sum()
        area_km2 = hex_gdf.loc[mask, "area_km2"].sum() if count > 0 else 0
        stats.append(
            {"district_name": district_name, "cell_count": count, "area_km2": area_km2}
        )

    total_cells = sum(s["cell_count"] for s in stats)
    total_area = sum(s["area_km2"] for s in stats)

    print(
        f"{'Район':<20} {'Кол-во ячеек':>15} {'Площадь (км²)':>15} {'% от общей':>12}"
    )
    print("-" * 70)

    for s in stats:
        pct = (s["area_km2"] / total_area * 100) if total_area > 0 else 0
        print(
            f"{s['district_name']:<20} {s['cell_count']:>15} {s['area_km2']:>15.2f} {pct:>11.1f}%"
        )

    print("-" * 70)
    print(f"{'ВСЕГО':<20} {total_cells:>15} {total_area:>15.2f} {'100.0%':>12}")
    print("=" * 70)


if __name__ == "__main__":
    # Пути к данным
    boundary_path = os.path.join(DATA_EXTERNAL, "sochi_boundary.geojson")
    districts_path = os.path.join(DATA_EXTERNAL, "sochi_districts.geojson")

    # Проверка существования файлов
    if not os.path.exists(boundary_path):
        print(f"ОШИБКА: Не найден файл {boundary_path}")
        sys.exit(1)

    if not os.path.exists(districts_path):
        print(f"ОШИБКА: Не найден файл {districts_path}")
        sys.exit(1)

    # Создание сетки
    hex_gdf, hex_gdf_wgs84 = create_hex_grid(boundary_path, districts_path)

    # Сохранение
    save_results(hex_gdf, hex_gdf_wgs84)

    # Статистика
    print_statistics(hex_gdf)

    print("\n✓ Создание гексагональной сетки H3 завершено!")
