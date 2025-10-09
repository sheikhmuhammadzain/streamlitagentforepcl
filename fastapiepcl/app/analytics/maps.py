import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster


def add_coordinates_to_df(df: pd.DataFrame, location_coords: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    lat = pd.Series(pd.NA, index=df.index)
    lon = pd.Series(pd.NA, index=df.index)
    for col in ['location.1', 'sublocation', 'location']:
        if col in df.columns:
            lat = lat.fillna(df[col].astype(str).map(lambda v: location_coords.get(v, {}).get('lat')))
            lon = lon.fillna(df[col].astype(str).map(lambda v: location_coords.get(v, {}).get('lon')))
    df['latitude'] = lat
    df['longitude'] = lon
    # Deterministic jitter to prevent overlaps and reruns
    mask = df['latitude'].notna() & df['longitude'].notna()
    if mask.any():
        key = df['incident_id'].astype(str) if 'incident_id' in df.columns else (
            df['title'].astype(str) if 'title' in df.columns else (
            df['location.1'].astype(str) if 'location.1' in df.columns else (
            df['location'].astype(str) if 'location' in df.columns else pd.Series(df.index.astype(str), index=df.index))))
        h = pd.util.hash_pandas_object(key, index=False).astype(np.uint32)
        lat_j = ((h % 1000) / 1000.0 - 0.5) * 0.00003
        lon_j = (((h // 1000) % 1000) / 1000.0 - 0.5) * 0.00003
        df.loc[mask, 'latitude'] = df.loc[mask, 'latitude'].astype(float) + lat_j[mask].values
        df.loc[mask, 'longitude'] = df.loc[mask, 'longitude'].astype(float) + lon_j[mask].values
    return df


def build_combined_map_html(inc_df: pd.DataFrame, haz_df: pd.DataFrame) -> str:
    inc = inc_df.copy()
    haz = haz_df.copy()
    # Try to infer latitude/longitude from common alternative column names
    def _ensure_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
        if 'latitude' not in df.columns:
            for cand in ['lat', 'Lat', 'LAT', 'Latitude', 'LATITUDE']:
                if cand in df.columns:
                    df['latitude'] = pd.to_numeric(df[cand], errors='coerce')
                    break
        if 'longitude' not in df.columns:
            for cand in ['lon', 'Lon', 'LON', 'lng', 'Lng', 'LNG', 'long', 'Long', 'LONG', 'Longitude', 'LONGITUDE']:
                if cand in df.columns:
                    df['longitude'] = pd.to_numeric(df[cand], errors='coerce')
                    break
        # Ensure expected coordinate columns exist to avoid KeyError
        if 'latitude' not in df.columns:
            df['latitude'] = pd.NA
        if 'longitude' not in df.columns:
            df['longitude'] = pd.NA
        return df

    inc = _ensure_lat_lon(inc)
    haz = _ensure_lat_lon(haz)

    # If coordinates are still missing, synthesize deterministic fallback positions
    def _synthesize_coords(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        base_lat, base_lon = 24.8607, 67.0011  # Karachi (EPCL vicinity) as default center
        # choose a key column to group by location semantics
        key_col = 'location.1' if 'location.1' in df.columns else (
            'sublocation' if 'sublocation' in df.columns else (
            'location' if 'location' in df.columns else (
            'department' if 'department' in df.columns else (
            'Department' if 'Department' in df.columns else (
            'title' if 'title' in df.columns else (
            'incident_id' if 'incident_id' in df.columns else None))))))
        if key_col is None:
            # fallback to index values as grouping key
            df = df.copy()
            df['_fallback_index_key'] = df.index.astype(str)
            key_col = '_fallback_index_key'
        lat_nan = df['latitude'].isna()
        lon_nan = df['longitude'].isna()
        mask = lat_nan | lon_nan
        if not mask.any():
            return df
        keys = df.loc[mask, key_col].astype(str).fillna('Unknown')
        h = pd.util.hash_pandas_object(keys, index=False).astype(np.uint32)
        # spread within roughly +/- ~220 meters (0.002 degrees) from base
        lat_off = ((h % 1000) / 1000.0 - 0.5) * 0.004
        lon_off = (((h // 1000) % 1000) / 1000.0 - 0.5) * 0.004
        df.loc[mask, 'latitude'] = pd.to_numeric(df.loc[mask, 'latitude'], errors='coerce').fillna(base_lat) + lat_off.values
        df.loc[mask, 'longitude'] = pd.to_numeric(df.loc[mask, 'longitude'], errors='coerce').fillna(base_lon) + lon_off.values
        return df

    inc = _synthesize_coords(inc)
    haz = _synthesize_coords(haz)
    inc_coords = inc.dropna(subset=['latitude', 'longitude']) if not inc.empty else pd.DataFrame()
    haz_coords = haz.dropna(subset=['latitude', 'longitude']) if not haz.empty else pd.DataFrame()

    if not inc_coords.empty:
        center_lat = float(inc_coords['latitude'].mean())
        center_lon = float(inc_coords['longitude'].mean())
    elif not haz_coords.empty:
        center_lat = float(haz_coords['latitude'].mean())
        center_lon = float(haz_coords['longitude'].mean())
    else:
        center_lat, center_lon = 24.8607, 67.0011

    m = folium.Map(location=[center_lat, center_lon], zoom_start=17, max_zoom=22, tiles='CartoDB Positron', control_scale=True, prefer_canvas=True)

    if not inc_coords.empty:
        w = pd.to_numeric(inc_coords.get('severity_score', pd.Series(1.0, index=inc_coords.index)), errors='coerce').fillna(1.0) / 5.0
        if len(inc_coords) > 3000:
            inc_coords = inc_coords.sample(3000, random_state=42)
            w = w.loc[inc_coords.index]
        fg_inc = folium.FeatureGroup(name='Incidents Heat', show=True)
        HeatMap(list(zip(inc_coords['latitude'].astype(float), inc_coords['longitude'].astype(float), w.astype(float))), min_opacity=0.25, radius=18, blur=12, gradient={0.0: '#3b82f6', 0.5: '#fde047', 0.75: '#f59e0b', 1.0: '#dc2626'}).add_to(fg_inc)
        fg_inc.add_to(m)

    if not haz_coords.empty:
        w = pd.to_numeric(haz_coords.get('severity_score', pd.Series(1.0, index=haz_coords.index)), errors='coerce').fillna(1.0) / 5.0
        if len(haz_coords) > 3000:
            haz_coords = haz_coords.sample(3000, random_state=42)
            w = w.loc[haz_coords.index]
        fg_haz = folium.FeatureGroup(name='Hazards Heat', show=True)
        HeatMap(list(zip(haz_coords['latitude'].astype(float), haz_coords['longitude'].astype(float), w.astype(float))), min_opacity=0.25, radius=18, blur=12, gradient={0.0: '#16a34a', 0.5: '#facc15', 0.75: '#f59e0b', 1.0: '#7f1d1d'}).add_to(fg_haz)
        fg_haz.add_to(m)

    def _labels_layer(df: pd.DataFrame, name: str, color: str, max_labels: int = 20):
        if df.empty:
            return
        key_col = 'location.1' if 'location.1' in df.columns else ('sublocation' if 'sublocation' in df.columns else ('location' if 'location' in df.columns else None))
        if not key_col:
            return
        grp = df.groupby(key_col).agg(lat=('latitude','mean'), lon=('longitude','mean'), count=(key_col,'size')).reset_index()
        grp = grp.sort_values('count', ascending=False).head(max_labels)
        fg = folium.FeatureGroup(name=name, show=True)
        css = """
        <style>
            .map-label {background: rgba(255,255,255,0.9); padding: 4px 6px; border-radius: 4px; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,.08); font-size: 12px; color: #111827;}
            .map-label .count {color: %s; font-weight: 700;}
        </style>
        """ % color
        m.get_root().header.add_child(folium.Element(css))
        for _, r in grp.iterrows():
            html = f"<div class='map-label'>{r[key_col]}<br><span class='count'>{int(r['count'])}</span></div>"
            folium.Marker(location=[float(r['lat']), float(r['lon'])], icon=folium.DivIcon(html=html)).add_to(fg)
        fg.add_to(m)

    _labels_layer(inc_coords, 'Incident Labels', '#dc2626')
    _labels_layer(haz_coords, 'Hazard Labels', '#b45309')

    # Compact detail clusters
    if not inc_coords.empty:
        inc_cluster = MarkerCluster(name='Incident Details', show=False)
        for _, row in inc_coords.head(600).iterrows():
            popup = f"<b>Incident</b><br>Location: {row.get('location.1', row.get('sublocation', 'Unknown'))}<br>Date: {row.get('occurrence_date','N/A')}<br>Status: {row.get('status','N/A')}<br>Severity: {row.get('severity_score','N/A')}<br>Risk: {row.get('risk_score','N/A')}"
            folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=3, color='red', fill=True, fill_opacity=0.6, popup=popup).add_to(inc_cluster)
        inc_cluster.add_to(m)
    if not haz_coords.empty:
        haz_cluster = MarkerCluster(name='Hazard Details', show=False)
        for _, row in haz_coords.head(600).iterrows():
            popup = f"<b>Hazard</b><br>Location: {row.get('location.1', row.get('sublocation', 'Unknown'))}<br>Date: {row.get('occurrence_date','N/A')}<br>Status: {row.get('status','N/A')}<br>Severity: {row.get('severity_score','N/A')}<br>Risk: {row.get('risk_score','N/A')}"
            folium.CircleMarker([float(row['latitude']), float(row['longitude'])], radius=3, color='orange', fill=True, fill_opacity=0.6, popup=popup).add_to(haz_cluster)
        haz_cluster.add_to(m)

    if not inc_coords.empty or not haz_coords.empty:
        lat_concat = pd.concat([inc_coords['latitude'], haz_coords['latitude']]).astype(float)
        lon_concat = pd.concat([inc_coords['longitude'], haz_coords['longitude']]).astype(float)
        m.fit_bounds([[lat_concat.min(), lon_concat.min()], [lat_concat.max(), lon_concat.max()]])

    folium.LayerControl().add_to(m)
    return m.get_root().render()


