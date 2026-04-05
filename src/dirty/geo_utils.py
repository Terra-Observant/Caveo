"""
geo_utils.py - Утилиты для работы с геоданными

Функции:
- Конвертация CSV в GeoDataFrame
- Перепроецирование между CRS
- Валидация геометрии
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.validation import make_valid
import warnings


def csv_to_geodataframe(
    csv_path: str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Конвертирует CSV с координатами в GeoDataFrame.
    
    Parameters
    ----------
    csv_path : str
        Путь к CSV файлу
    lon_col : str
        Название колонки долготы
    lat_col : str
        Название колонки широты
    crs : str
        Система координат (по умолчанию WGS84)
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame с геометрией точек
    """
    df = pd.read_csv(csv_path)
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    return gdf


def reproject_gdf(
    gdf: gpd.GeoDataFrame,
    target_crs: str = "EPSG:32637"
) -> gpd.GeoDataFrame:
    """
    Перепроецирует GeoDataFrame в указанную CRS.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Входной GeoDataFrame
    target_crs : str
        Целевая CRS (по умолчанию UTM Zone 37N)
        
    Returns
    -------
    gpd.GeoDataFrame
        Перепроецированный GeoDataFrame
    """
    return gdf.to_crs(target_crs)


def validate_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Проверяет и исправляет валидность геометрии.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame для проверки
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame с валидной геометрией
    """
    # Исправляем невалидные геометрии
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: make_valid(geom) if not geom.is_valid else geom
    )
    
    # Удаляем пустые геометрии
    gdf = gdf[~gdf["geometry"].is_empty]
    gdf = gdf[gdf["geometry"].notna()]
    
    return gdf


def load_and_validate(
    geojson_path: str,
    target_crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Загружает GeoJSON и валидирует геометрию.
    
    Parameters
    ----------
    geojson_path : str
        Путь к GeoJSON файлу
    target_crs : str
        Целевая CRS
        
    Returns
    -------
    gpd.GeoDataFrame
        Валидный GeoDataFrame
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf = gpd.read_file(geojson_path)
    
    gdf = validate_geometry(gdf)
    
    if gdf.crs != target_crs:
        gdf = reproject_gdf(gdf, target_crs)
    
    return gdf


def calculate_area(gdf: gpd.GeoDataFrame, crs_utm: str = "EPSG:32637") -> gpd.GeoDataFrame:
    """
    Добавляет колонки с площадью в м² и км².
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame с геометрией
    crs_utm : str
        UTM CRS для расчёта площади
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame с добавленными колонками area_m2 и area_km2
    """
    gdf_utm = gdf.to_crs(crs_utm)
    gdf["area_m2"] = gdf_utm.geometry.area
    gdf["area_km2"] = gdf["area_m2"] / 1_000_000
    return gdf
