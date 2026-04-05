"""
cell_aggregator.py - Модуль агрегации данных по ячейкам гексагональной сетки

Функции:
- Агрегация растровых данных (зональная статистика)
- Агрегация точечных данных (подсчёт, расстояния)
- Буферная статистика
- Общий интерфейс для добавления признаков
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from typing import Union, List, Optional, Dict, Any
import os


class CellAggregator:
    """
    Класс для агрегации различных типов данных по ячейкам гексагональной сетки.
    
    Parameters
    ----------
    grid_path : str
        Путь к файлу гексагональной сетки (GeoJSON)
    """
    
    def __init__(self, grid_path: str):
        self.grid = gpd.read_file(grid_path)
        self.features = {}
        print(f"Загружена сетка: {len(self.grid)} ячеек")
    
    def add_raster_features(
        self,
        raster_path: str,
        feature_prefix: str,
        stats: Optional[List[str]] = None,
        nodata_value: float = -9999
    ) -> 'CellAggregator':
        """
        Добавляет признаки из растровых данных (зональная статистика).
        
        Parameters
        ----------
        raster_path : str
            Путь к растровому файлу (GeoTIFF)
        feature_prefix : str
            Префикс для названий признаков (например, 'elev')
        stats : list, optional
            Список статистик для вычисления
            По умолчанию: ['mean', 'min', 'max', 'std']
        nodata_value : float
            Значение nodata для растра
            
        Returns
        -------
        CellAggregator
            Self для цепочки вызовов
        """
        try:
            from rasterstats import zonal_stats
        except ImportError:
            print("Установите rasterstats: pip install rasterstats")
            return self
        
        if stats is None:
            stats = ['mean', 'min', 'max', 'std']
        
        print(f"Добавление растровых признаков: {feature_prefix}...")
        
        # Зональная статистика
        results = zonal_stats(
            vectors=self.grid.geometry,
            raster=raster_path,
            stats=stats,
            nodata=nodata_value,
            all_touched=False
        )
        
        # Добавление результатов в DataFrame
        for stat in stats:
            col_name = f"{feature_prefix}_{stat}"
            self.grid[col_name] = [r.get(stat, np.nan) for r in results]
            self.features[col_name] = {
                'source': 'raster',
                'raster': raster_path,
                'stat': stat
            }
        
        print(f"  ✓ Добавлено {len(stats)} признаков: {[f'{feature_prefix}_{s}' for s in stats]}")
        return self
    
    def add_point_features(
        self,
        points_gdf: gpd.GeoDataFrame,
        feature_prefix: str,
        count_col: Optional[str] = None
    ) -> 'CellAggregator':
        """
        Добавляет признаки из точечных данных (подсчёт в ячейках).
        
        Parameters
        ----------
        points_gdf : gpd.GeoDataFrame
            GeoDataFrame с точками
        feature_prefix : str
            Префикс для названий признаков (например, 'ls')
        count_col : str, optional
            Название колонки для подсчёта (если None, подсчитывает все точки)
            
        Returns
        -------
        CellAggregator
            Self для цепочки вызовов
        """
        print(f"Добавление точечных признаков: {feature_prefix}...")
        
        # Убедимся, что CRS совпадает
        if points_gdf.crs != self.grid.crs:
            points_gdf = points_gdf.to_crs(self.grid.crs)
        
        # Пространственное соединение
        joined = gpd.sjoin(points_gdf, self.grid[['cell_id', 'geometry']], 
                          how='inner', predicate='within')
        
        # Подсчёт точек в каждой ячейке
        if count_col and count_col in joined.columns:
            counts = joined.groupby('cell_id')[count_col].sum()
        else:
            counts = joined.groupby('cell_id').size()
        
        # Добавление в сетку
        count_col_name = f"{feature_prefix}_count"
        self.grid[count_col_name] = self.grid['cell_id'].map(counts).fillna(0).astype(int)
        
        # Флаг наличия события
        has_event_col = f"{feature_prefix}_has_event"
        self.grid[has_event_col] = (self.grid[count_col_name] > 0).astype(int)
        
        self.features[count_col_name] = {
            'source': 'point',
            'method': 'spatial_join_count'
        }
        
        print(f"  ✓ Добавлено 2 признака: {count_col_name}, {has_event_col}")
        return self
    
    def add_nearest_distance(
        self,
        points_gdf: gpd.GeoDataFrame,
        feature_prefix: str
    ) -> 'CellAggregator':
        """
        Добавляет расстояние до ближайшей точки.
        
        Parameters
        ----------
        points_gdf : gpd.GeoDataFrame
            GeoDataFrame с точками
        feature_prefix : str
            Префикс для названия признака
            
        Returns
        -------
        CellAggregator
            Self для цепочки вызовов
        """
        print(f"Добавление расстояний до ближайших точек: {feature_prefix}...")
        
        # Убедимся, что CRS совпадает (должна быть проекция в метрах)
        if points_gdf.crs != self.grid.crs:
            points_gdf = points_gdf.to_crs(self.grid.crs)
        
        # Координаты точек
        points_coords = np.array([(p.x, p.y) for p in points_gdf.geometry])
        
        # Координаты центроидов ячеек
        centroids = self.grid.geometry.centroid
        centroid_coords = np.array([(c.x, c.y) for c in centroids])
        
        # KD-дерево для быстрого поиска
        tree = cKDTree(points_coords)
        distances, _ = tree.query(centroid_coords, k=1)
        
        # Добавление в DataFrame
        dist_col = f"{feature_prefix}_nearest_m"
        self.grid[dist_col] = distances
        
        self.features[dist_col] = {
            'source': 'point',
            'method': 'nearest_neighbor_distance'
        }
        
        print(f"  ✓ Добавлен признак: {dist_col}")
        return self
    
    def add_buffer_stats(
        self,
        points_gdf: gpd.GeoDataFrame,
        feature_prefix: str,
        buffer_radius: float = 5000
    ) -> 'CellAggregator':
        """
        Добавляет статистику в буфере вокруг центроида.
        
        Parameters
        ----------
        points_gdf : gpd.GeoDataFrame
            GeoDataFrame с точками
        feature_prefix : str
            Префикс для названия признаков
        buffer_radius : float
            Радиус буфера в метрах (по умолчанию 5000 м = 5 км)
            
        Returns
        -------
        CellAggregator
            Self для цепочки вызовов
        """
        print(f"Добавление буферной статистики: {feature_prefix} (радиус {buffer_radius} м)...")
        
        # Убедимся, что CRS совпадает
        if points_gdf.crs != self.grid.crs:
            points_gdf = points_gdf.to_crs(self.grid.crs)
        
        buffer_counts = []
        
        for idx, row in self.grid.iterrows():
            # Создаём буфер вокруг центроида
            centroid = row.geometry.centroid
            buffer = centroid.buffer(buffer_radius)
            
            # Подсчёт точек в буфере
            points_in_buffer = points_gdf[points_gdf.within(buffer)]
            buffer_counts.append(len(points_in_buffer))
        
        # Добавление в DataFrame
        count_col = f"{feature_prefix}_count_{int(buffer_radius/1000)}km"
        self.grid[count_col] = buffer_counts
        
        # Плотность
        buffer_area_km2 = np.pi * (buffer_radius / 1000) ** 2
        density_col = f"{feature_prefix}_density_{int(buffer_radius/1000)}km"
        self.grid[density_col] = buffer_counts / buffer_area_km2
        
        self.features[count_col] = {
            'source': 'point',
            'method': f'buffer_count_{buffer_radius}m'
        }
        
        print(f"  ✓ Добавлено 2 признака: {count_col}, {density_col}")
        return self
    
    def add_value_from_column(
        self,
        source_gdf: gpd.GeoDataFrame,
        source_col: str,
        target_col: str
    ) -> 'CellAggregator':
        """
        Добавляет значения из колонки другого GeoDataFrame по spatial join.
        
        Parameters
        ----------
        source_gdf : gpd.GeoDataFrame
            GeoDataFrame с исходными данными
        source_col : str
            Название колонки в source_gdf
        target_col : str
            Название колонки для создания в сетке
            
        Returns
        -------
        CellAggregator
            Self для цепочки вызовов
        """
        print(f"Добавление данных из колонки: {source_col} -> {target_col}...")
        
        if source_gdf.crs != self.grid.crs:
            source_gdf = source_gdf.to_crs(self.grid.crs)
        
        # Пространственное соединение
        joined = gpd.sjoin(self.grid, source_gdf[[source_col, 'geometry']], 
                          how='left', predicate='within')
        
        self.grid[target_col] = joined[source_col].values
        
        self.features[target_col] = {
            'source': 'vector',
            'method': 'spatial_join'
        }
        
        print(f"  ✓ Добавлен признак: {target_col}")
        return self
    
    def get_features_df(self) -> pd.DataFrame:
        """
        Возвращает DataFrame со всеми признаками.
        
        Returns
        -------
        pd.DataFrame
            Таблица с признаками (без геометрии)
        """
        cols = ['cell_id'] + list(self.features.keys())
        return self.grid[cols].copy()
    
    def save_features(self, output_path: str):
        """
        Сохраняет таблицу признаков.
        
        Parameters
        ----------
        output_path : str
            Путь для сохранения (CSV)
        """
        features_df = self.get_features_df()
        features_df.to_csv(output_path, index=False)
        print(f"✓ Признаки сохранены: {output_path}")
    
    def get_grid(self) -> gpd.GeoDataFrame:
        """
        Возвращает GeoDataFrame со всеми признаками.
        
        Returns
        -------
        gpd.GeoDataFrame
            Сетка с добавленными признаками
        """
        return self.grid.copy()
    
    def summary(self):
        """Выводит сводку добавленных признаков."""
        print("\n" + "="*60)
        print("СВОДКА ДОБАВЛЕННЫХ ПРИЗНАКОВ")
        print("="*60)
        
        for feature_name, feature_info in self.features.items():
            source = feature_info.get('source', 'unknown')
            method = feature_info.get('method', 'unknown')
            print(f"  {feature_name:<30} [{source:6s}] {method}")
        
        print(f"\nВсего признаков: {len(self.features)}")
        print("="*60)


def create_aggregator_from_file(grid_path: str) -> CellAggregator:
    """
    Фабричная функция для создания агрегатора.
    
    Parameters
    ----------
    grid_path : str
        Путь к файлу сетки
        
    Returns
    -------
    CellAggregator
        Экземпляр агрегатора
    """
    return CellAggregator(grid_path)
