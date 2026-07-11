from __future__ import annotations

import io
import os
import re
import json
import math
import zipfile
import tempfile
import heapq
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests

import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import features
from affine import Affine

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box, shape
from shapely.ops import transform as shp_transform, unary_union
from shapely.validation import make_valid
from pyproj import CRS, Transformer, Geod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =============================================================================
# HIDROSED_V1 · Base modular
# Cuenca morfométrica + eje del cauce + secciones
# =============================================================================

st.set_page_config(page_title="HIDROSED_V1", page_icon="💧", layout="wide")

# ----------------------------- Data classes ----------------------------------
@dataclass
class BasinMorphometry:
    name: str
    area_km2: float
    area_ha: float
    perimeter_km: float
    compactness_kc: float
    equivalent_diameter_km: float
    bbox_length_km: float
    form_factor: float
    elongation_ratio: float
    z_min: float
    z_mean: float
    z_max: float
    relief_m: float
    mean_slope_pct: float
    centroid_x: float
    centroid_y: float
    touches_dem_edge: bool
    confidence_pct: float
    warnings: List[str] = field(default_factory=list)

@dataclass
class AxisInfo:
    length_full_m: float
    length_useful_m: float
    km_pc_support: float
    km_pc_hydro: float
    orientation: str
    z_upstream: Optional[float]
    z_downstream: Optional[float]
    warnings: List[str] = field(default_factory=list)

@dataclass
class CrossSection:
    section_id: str
    name: str
    km: float
    line_xy: List[Tuple[float, float]]
    stations: List[float]
    elevations: List[float]
    origin: str = "DEM"
    source_topography: str = "DEM"
    section_type: str = "natural"
    status_qa: str = "aceptada"
    notes: str = ""
    talweg_station: Optional[float] = None
    talweg_elevation: Optional[float] = None
    left_bank_station: Optional[float] = None
    right_bank_station: Optional[float] = None
    total_width_m: Optional[float] = None
    main_channel_width_m: Optional[float] = None
    length_downstream_m: Optional[float] = None
    length_lob_m: Optional[float] = None
    length_channel_m: Optional[float] = None
    length_rob_m: Optional[float] = None
    manning_lob: float = 0.040
    manning_channel: float = 0.035
    manning_rob: float = 0.040
    contraction_coeff: float = 0.10
    expansion_coeff: float = 0.30
    plan: str = "Base"
    change_log: List[Dict[str, Any]] = field(default_factory=list)
    original_stations: List[float] = field(default_factory=list)
    original_elevations: List[float] = field(default_factory=list)

# ------------------------------- KML/KMZ -------------------------------------
def uploaded_bytes(uploaded) -> bytes:
    if uploaded is None:
        raise ValueError("Archivo no cargado.")
    uploaded.seek(0)
    return uploaded.read()

def read_kml_text(uploaded) -> str:
    data = uploaded_bytes(uploaded)
    name = (uploaded.name or "").lower()
    if name.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            kmls = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kmls:
                raise ValueError("El KMZ no contiene KML.")
            return z.read(kmls[0]).decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")

def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1]

def parse_coord_text(text: str) -> List[Tuple[float, float, Optional[float]]]:
    coords = []
    for item in re.split(r"\s+", (text or "").strip()):
        if not item:
            continue
        parts = item.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0]); lat = float(parts[1])
                z = float(parts[2]) if len(parts) > 2 and parts[2] != "" else None
                coords.append((lon, lat, z))
            except Exception:
                pass
    return coords

def extract_kml_geometries(uploaded) -> Dict[str, List[Tuple[str, Any]]]:
    kml = read_kml_text(uploaded)
    root = ET.fromstring(kml.encode("utf-8"))
    out = {"points": [], "lines": [], "polygons": []}
    for pm in root.iter():
        if _strip_ns(pm.tag) != "Placemark":
            continue
        name = "Sin nombre"
        for child in pm:
            if _strip_ns(child.tag) == "name" and child.text:
                name = child.text.strip()
                break
        for node in pm.iter():
            tag = _strip_ns(node.tag)
            if tag == "Point":
                cnode = next((c for c in node.iter() if _strip_ns(c.tag) == "coordinates"), None)
                coords = parse_coord_text(cnode.text if cnode is not None else "")
                if coords:
                    lon, lat, _ = coords[0]
                    out["points"].append((name, Point(lon, lat)))
            elif tag == "LineString":
                cnode = next((c for c in node.iter() if _strip_ns(c.tag) == "coordinates"), None)
                coords = parse_coord_text(cnode.text if cnode is not None else "")
                if len(coords) >= 2:
                    out["lines"].append((name, LineString([(lon, lat) for lon, lat, _ in coords])))
            elif tag == "Polygon":
                cnode = next((c for c in node.iter() if _strip_ns(c.tag) == "coordinates"), None)
                coords = parse_coord_text(cnode.text if cnode is not None else "")
                if len(coords) >= 4:
                    out["polygons"].append((name, Polygon([(lon, lat) for lon, lat, _ in coords])))
    return out

def first_point(uploaded, label: str) -> Tuple[str, Point]:
    geoms = extract_kml_geometries(uploaded)
    if not geoms["points"]:
        raise ValueError(f"{label}: el archivo no contiene un punto válido.")
    return geoms["points"][0]

def longest_line(uploaded, label: str) -> Tuple[str, LineString]:
    geoms = extract_kml_geometries(uploaded)
    if not geoms["lines"]:
        raise ValueError(f"{label}: el archivo no contiene una línea válida.")
    return max(geoms["lines"], key=lambda t: t[1].length)

# ----------------------------- Projection ------------------------------------
def utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = 32700 + zone if lat < 0 else 32600 + zone
    return CRS.from_epsg(epsg)

def tx(src, dst) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)

def project_geom(geom, src, dst):
    tr = tx(src, dst)
    return shp_transform(lambda x, y, z=None: tr.transform(x, y), geom)

# ----------------------------- DEM download/load -----------------------------
def bbox_from_inputs(pc_support: Point, pc_hydro: Point, axis: LineString, margin_km: float) -> Tuple[float, float, float, float]:
    geom = unary_union([pc_support, pc_hydro, axis])
    minx, miny, maxx, maxy = geom.bounds
    # degrees approximated at latitude
    lat0 = (miny + maxy) / 2
    dlat = margin_km / 111.0
    dlon = margin_km / (111.0 * max(math.cos(math.radians(lat0)), 0.15))
    return (minx - dlon, miny - dlat, maxx + dlon, maxy + dlat)

def split_bbox(bounds: Tuple[float, float, float, float], n: int) -> List[Tuple[float, float, float, float]]:
    west, south, east, north = bounds
    n = max(1, int(n))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    boxes = []
    dx = (east - west) / cols
    dy = (north - south) / rows
    for r in range(rows):
        for c in range(cols):
            if len(boxes) >= n:
                break
            boxes.append((west + c*dx, south + r*dy, west + (c+1)*dx, south + (r+1)*dy))
    return boxes

def download_opentopo_dem(api_key: str, demtype: str, bounds: Tuple[float, float, float, float], tmpdir: str, n_tiles: int = 1) -> str:
    west, south, east, north = bounds
    tile_paths = []
    for i, b in enumerate(split_bbox(bounds, n_tiles), start=1):
        w, s, e, n = b
        url = "https://portal.opentopography.org/API/globaldem"
        params = dict(demtype=demtype, south=s, north=n, west=w, east=e, outputFormat="GTiff", API_Key=api_key)
        r = requests.get(url, params=params, timeout=240)
        if r.status_code != 200 or not r.content or len(r.content) < 1000:
            raise RuntimeError(f"OpenTopography falló en tile {i}/{n_tiles}: código {r.status_code}. {r.text[:300]}")
        path = os.path.join(tmpdir, f"dem_tile_{i:02d}.tif")
        with open(path, "wb") as f:
            f.write(r.content)
        tile_paths.append(path)
    if len(tile_paths) == 1:
        return tile_paths[0]
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})
    out = os.path.join(tmpdir, "dem_merged.tif")
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    return out

def save_uploaded_dem(uploaded, tmpdir: str) -> str:
    path = os.path.join(tmpdir, "dem_manual.tif")
    with open(path, "wb") as f:
        f.write(uploaded_bytes(uploaded))
    return path

def reproject_dem_to_utm(src_path: str, dst_crs: CRS, resolution_m: float, tmpdir: str) -> Tuple[np.ndarray, Affine, CRS, np.ndarray, str]:
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution_m)
        kwargs = src.meta.copy()
        kwargs.update(crs=dst_crs, transform=transform, width=width, height=height, count=1, dtype="float32", nodata=np.nan)
        out_path = os.path.join(tmpdir, "dem_utm.tif")
        arr = np.full((height, width), np.nan, dtype="float32")
        reproject(source=rasterio.band(src, 1), destination=arr, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.bilinear, dst_nodata=np.nan)
    valid = np.isfinite(arr)
    with rasterio.open(out_path, "w", **kwargs) as dst:
        dst.write(arr, 1)
    return arr, transform, dst_crs, valid, out_path

# ---------------------------- Hydrology raster -------------------------------
def priority_flood_fill(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # Lightweight priority flood; enough for DEM conditioning without scipy.
    h, w = dem.shape
    filled = dem.copy().astype("float64")
    visited = np.zeros((h, w), dtype=bool)
    pq: List[Tuple[float, int, int]] = []
    for r in range(h):
        for c in (0, w-1):
            if valid[r, c] and not visited[r, c]:
                visited[r, c] = True; heapq.heappush(pq, (filled[r, c], r, c))
    for c in range(w):
        for r in (0, h-1):
            if valid[r, c] and not visited[r, c]:
                visited[r, c] = True; heapq.heappush(pq, (filled[r, c], r, c))
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    eps = 1e-4
    while pq:
        z, r, c = heapq.heappop(pq)
        for dr, dc in neigh:
            rr, cc = r+dr, c+dc
            if 0 <= rr < h and 0 <= cc < w and valid[rr, cc] and not visited[rr, cc]:
                visited[rr, cc] = True
                if filled[rr, cc] <= z:
                    filled[rr, cc] = z + eps
                heapq.heappush(pq, (filled[rr, cc], rr, cc))
    filled[~valid] = np.nan
    return filled

def compute_d8(filled: np.ndarray, valid: np.ndarray, transform: Affine) -> Tuple[np.ndarray, np.ndarray]:
    h, w = filled.shape
    rec = np.full(h*w, -1, dtype=np.int64)
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    csx = abs(transform.a); csy = abs(transform.e)
    dist = [math.hypot(dc*csx, dr*csy) for dr, dc in dirs]
    for r in range(h):
        for c in range(w):
            if not valid[r,c]: continue
            z = filled[r,c]
            best_slope = 0.0; best = -1
            for k,(dr,dc) in enumerate(dirs):
                rr, cc = r+dr, c+dc
                if 0 <= rr < h and 0 <= cc < w and valid[rr,cc]:
                    slope = (z - filled[rr,cc]) / dist[k]
                    if slope > best_slope:
                        best_slope = slope; best = rr*w + cc
            rec[r*w+c] = best
    return rec, valid.ravel()

def compute_accumulation(rec: np.ndarray, valid_flat: np.ndarray) -> np.ndarray:
    n = rec.size
    indeg = np.zeros(n, dtype=np.int32)
    for i in range(n):
        j = rec[i]
        if valid_flat[i] and j >= 0:
            indeg[j] += 1
    acc = np.ones(n, dtype=np.float64)
    acc[~valid_flat] = 0
    q = [i for i in range(n) if valid_flat[i] and indeg[i] == 0]
    head = 0
    while head < len(q):
        i = q[head]; head += 1
        j = rec[i]
        if j >= 0:
            acc[j] += acc[i]
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    return acc

def world_to_rowcol(transform: Affine, x: float, y: float) -> Tuple[int,int]:
    inv = ~transform
    c, r = inv * (x, y)
    return int(round(r)), int(round(c))

def rowcol_to_world(transform: Affine, r: int, c: int) -> Tuple[float,float]:
    x, y = transform * (c + 0.5, r + 0.5)
    return x, y

def snap_to_acc(point: Point, acc2d: np.ndarray, valid: np.ndarray, transform: Affine, radius_m: float, axis: Optional[LineString] = None, axis_buffer_m: float = 500.0) -> Tuple[int,int,float,float]:
    h, w = acc2d.shape
    r0, c0 = world_to_rowcol(transform, point.x, point.y)
    px = abs(transform.a)
    rad = max(1, int(math.ceil(radius_m / px)))
    best = None
    axis_buf = axis.buffer(axis_buffer_m) if axis is not None and axis_buffer_m > 0 else None
    for r in range(max(0,r0-rad), min(h,r0+rad+1)):
        for c in range(max(0,c0-rad), min(w,c0+rad+1)):
            if not valid[r,c]: continue
            x, y = rowcol_to_world(transform, r, c)
            d = math.hypot(x - point.x, y - point.y)
            if d > radius_m: continue
            if axis_buf is not None and not axis_buf.contains(Point(x,y)):
                continue
            val = acc2d[r,c]
            score = (val, -d)
            if best is None or score > best[0]:
                best = (score, r, c, d, val)
    if best is None:
        raise RuntimeError("No se pudo ajustar el punto al drenaje en el radio indicado.")
    _, r, c, d, val = best
    return r, c, float(d), float(val)

def upstream_mask(rec: np.ndarray, valid_flat: np.ndarray, outlet_idx: int) -> np.ndarray:
    children: List[List[int]] = [[] for _ in range(rec.size)]
    for i, j in enumerate(rec):
        if valid_flat[i] and j >= 0:
            children[j].append(i)
    mask = np.zeros(rec.size, dtype=bool)
    stack = [outlet_idx]
    mask[outlet_idx] = True
    while stack:
        j = stack.pop()
        for ch in children[j]:
            if not mask[ch]:
                mask[ch] = True; stack.append(ch)
    return mask

def polygonize_mask(mask2d: np.ndarray, transform: Affine):
    shapes = []
    for geom, val in features.shapes(mask2d.astype("uint8"), mask=mask2d, transform=transform):
        if val == 1:
            shapes.append(shape(geom))
    if not shapes:
        raise RuntimeError("No se generó polígono de cuenca.")
    geom = unary_union(shapes)
    geom = make_valid(geom)
    if isinstance(geom, MultiPolygon):
        geom = max(list(geom.geoms), key=lambda g: g.area)
    return geom

def mask_touches_edge(mask2d: np.ndarray) -> bool:
    return bool(mask2d[0,:].any() or mask2d[-1,:].any() or mask2d[:,0].any() or mask2d[:,-1].any())

def delineate_basin(pc: Point, name: str, dem: np.ndarray, transform: Affine, valid: np.ndarray, rec: np.ndarray, acc: np.ndarray, crs_utm: CRS, snap_radii: List[float], axis: Optional[LineString], axis_buffer_m: float) -> Dict[str, Any]:
    acc2d = acc.reshape(dem.shape)
    best = None
    valid_flat = valid.ravel()
    for rad in snap_radii:
        try:
            r, c, dist, acc_cells = snap_to_acc(pc, acc2d, valid, transform, rad, axis=axis, axis_buffer_m=axis_buffer_m)
            idx = r * dem.shape[1] + c
            mask_flat = upstream_mask(rec, valid_flat, idx)
            mask2d = mask_flat.reshape(dem.shape)
            poly = polygonize_mask(mask2d, transform)
            area_km2 = poly.area / 1e6
            if area_km2 <= 0:
                continue
            touches = mask_touches_edge(mask2d)
            shape_idx = poly.length / (2 * math.sqrt(math.pi * max(poly.area, 1)))
            score = 100.0
            if touches: score -= 25
            if dist > 0.8*rad: score -= 10
            if shape_idx > 4: score -= 10
            if area_km2 < 0.02: score -= 30
            if axis is not None:
                frac = axis.intersection(poly).length / max(axis.length, 1)
                if frac < 0.3: score -= 15
            cand = dict(name=name, polygon=poly, mask=mask2d, snapped_rc=(r,c), snap_distance_m=dist, snap_radius_m=rad, acc_cells=acc_cells, touches_edge=touches, confidence_pct=max(0,min(100,score)), shape_index=shape_idx)
            if best is None or cand["confidence_pct"] > best["confidence_pct"]:
                best = cand
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"No se pudo delimitar cuenca para {name}.")
    return best

# ---------------------------- Morphometry ------------------------------------
def sample_dem_in_geom(dem: np.ndarray, transform: Affine, geom, valid: np.ndarray) -> np.ndarray:
    mask = features.geometry_mask([geom.__geo_interface__], out_shape=dem.shape, transform=transform, invert=True)
    vals = dem[mask & valid]
    return vals[np.isfinite(vals)]

def slope_mean_pct(dem: np.ndarray, transform: Affine, geom, valid: np.ndarray) -> float:
    arr = dem.astype(float)
    dx = abs(transform.a); dy = abs(transform.e)
    gy, gx = np.gradient(arr, dy, dx)
    slope = np.sqrt(gx*gx + gy*gy) * 100.0
    mask = features.geometry_mask([geom.__geo_interface__], out_shape=dem.shape, transform=transform, invert=True)
    vals = slope[mask & valid & np.isfinite(slope)]
    return float(np.nanmean(vals)) if vals.size else float("nan")

def morphometry(name: str, basin: Dict[str, Any], dem: np.ndarray, transform: Affine, valid: np.ndarray) -> BasinMorphometry:
    poly = basin["polygon"]
    vals = sample_dem_in_geom(dem, transform, poly, valid)
    area_km2 = poly.area / 1e6
    perim_km = poly.length / 1000
    minx, miny, maxx, maxy = poly.bounds
    bbox_len_km = max(maxx-minx, maxy-miny) / 1000
    kc = perim_km / (2 * math.sqrt(math.pi * max(area_km2, 1e-9)))
    eqd = 2 * math.sqrt(area_km2 / math.pi)
    ff = area_km2 / max(bbox_len_km*bbox_len_km, 1e-9)
    elong = 2 * math.sqrt(area_km2 / math.pi) / max(bbox_len_km, 1e-9)
    zmin = float(np.nanmin(vals)) if vals.size else float("nan")
    zmean = float(np.nanmean(vals)) if vals.size else float("nan")
    zmax = float(np.nanmax(vals)) if vals.size else float("nan")
    centroid = poly.centroid
    warnings = []
    if basin["touches_edge"]: warnings.append("La cuenca toca el borde del DEM; ampliar DEM antes de usar como definitiva.")
    if basin["confidence_pct"] < 80: warnings.append("Confianza geométrica media/baja; revisar PC, eje y DEM.")
    return BasinMorphometry(
        name=name, area_km2=float(area_km2), area_ha=float(area_km2*100), perimeter_km=float(perim_km), compactness_kc=float(kc), equivalent_diameter_km=float(eqd), bbox_length_km=float(bbox_len_km), form_factor=float(ff), elongation_ratio=float(elong), z_min=zmin, z_mean=zmean, z_max=zmax, relief_m=float(zmax-zmin) if np.isfinite(zmax-zmin) else float("nan"), mean_slope_pct=slope_mean_pct(dem, transform, poly, valid), centroid_x=float(centroid.x), centroid_y=float(centroid.y), touches_dem_edge=bool(basin["touches_edge"]), confidence_pct=float(basin["confidence_pct"]), warnings=warnings)

def hypsometric_curve(dem: np.ndarray, transform: Affine, geom, valid: np.ndarray, bins: int = 40) -> pd.DataFrame:
    vals = sample_dem_in_geom(dem, transform, geom, valid)
    if vals.size < 5:
        return pd.DataFrame(columns=["elevacion_m", "area_relativa_pct"])
    vals = np.sort(vals)
    elev = np.linspace(vals.min(), vals.max(), bins)
    # area fraction above elevation
    frac = [(vals >= z).sum() / vals.size * 100 for z in elev]
    return pd.DataFrame({"elevacion_m": elev, "area_relativa_pct": frac})

# ------------------------------- Axis/sections -------------------------------
def substring_line(line: LineString, d0: float, d1: float) -> LineString:
    d0, d1 = sorted([max(0,min(line.length,d0)), max(0,min(line.length,d1))])
    if abs(d1-d0) < 1e-6:
        p = line.interpolate(d0)
        return LineString([p.coords[0], p.coords[0]])
    pts = [line.interpolate(d0)]
    for x,y in line.coords:
        d = line.project(Point(x,y))
        if d0 < d < d1:
            pts.append(Point(x,y))
    pts.append(line.interpolate(d1))
    # unique
    coords=[]
    for p in pts:
        xy=(float(p.x),float(p.y))
        if not coords or math.hypot(coords[-1][0]-xy[0], coords[-1][1]-xy[1]) > 1e-6:
            coords.append(xy)
    return LineString(coords) if len(coords)>=2 else LineString([line.interpolate(d0).coords[0], line.interpolate(d1).coords[0]])

def dem_sample(dem: np.ndarray, transform: Affine, valid: np.ndarray, x: float, y: float) -> Optional[float]:
    r, c = world_to_rowcol(transform, x, y)
    if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1] and valid[r,c]:
        return float(dem[r,c])
    return None

def orient_axis(axis: LineString, dem: np.ndarray, transform: Affine, valid: np.ndarray, manual_invert: bool) -> Tuple[LineString, str, Optional[float], Optional[float]]:
    start = Point(axis.coords[0]); end = Point(axis.coords[-1])
    zs = dem_sample(dem, transform, valid, start.x, start.y)
    ze = dem_sample(dem, transform, valid, end.x, end.y)
    coords = list(axis.coords)
    reason = "sentido original"
    # Want upstream to downstream: first point higher elevation
    if zs is not None and ze is not None and zs < ze:
        coords = list(reversed(coords)); zs, ze = ze, zs; reason = "invertido automáticamente por DEM"
    if manual_invert:
        coords = list(reversed(coords)); zs, ze = ze, zs; reason = "invertido manualmente"
    return LineString(coords), reason, zs, ze

def useful_axis_between_pcs(axis: LineString, pc_support: Point, pc_hydro: Point) -> Tuple[LineString, float, float]:
    d1 = axis.project(pc_support)
    d2 = axis.project(pc_hydro)
    seg = substring_line(axis, d1, d2)
    return seg, d1/1000, d2/1000

def tangent_at(line: LineString, d: float, eps: float = 1.0) -> Tuple[float,float]:
    d0=max(0,d-eps); d1=min(line.length,d+eps)
    p0=line.interpolate(d0); p1=line.interpolate(d1)
    vx=p1.x-p0.x; vy=p1.y-p0.y
    norm=math.hypot(vx,vy) or 1
    return vx/norm, vy/norm

def section_line(line: LineString, d: float, half_width: float) -> LineString:
    p=line.interpolate(d); tx_,ty_=tangent_at(line,d)
    # left looking downstream = rotate tangent CCW
    nx=-ty_; ny=tx_
    return LineString([(p.x+nx*half_width,p.y+ny*half_width),(p.x-nx*half_width,p.y-ny*half_width)])

def generate_profile(line: LineString, dem: np.ndarray, transform: Affine, valid: np.ndarray, step_m: float) -> pd.DataFrame:
    ds = np.arange(0, line.length + 0.1, max(step_m, 1.0))
    rows=[]
    for d in ds:
        p=line.interpolate(float(min(d,line.length)))
        z=dem_sample(dem,transform,valid,p.x,p.y)
        rows.append({"km": d/1000, "x": p.x, "y": p.y, "cota_dem_m": z})
    df=pd.DataFrame(rows)
    df["pendiente_local_pct"] = df["cota_dem_m"].diff(-1) / (df["km"].diff(-1)*1000) * 100
    return df

def make_natural_sections(line: LineString, dem: np.ndarray, transform: Affine, valid: np.ndarray, spacing_m: float, half_width: float, station_step: float) -> List[CrossSection]:
    ds=np.arange(0,line.length+0.01,max(spacing_m,1.0))
    secs=[]
    for idx,d in enumerate(ds):
        d=float(min(d,line.length))
        sl=section_line(line,d,half_width)
        coords=list(sl.coords)
        length=sl.length
        ss=np.arange(-half_width, half_width+0.01, max(station_step,0.5))
        xs=[]; ys=[]; zs=[]
        for s in ss:
            frac=(s+half_width)/max(2*half_width,1e-9)
            x=coords[0][0]+(coords[1][0]-coords[0][0])*frac
            y=coords[0][1]+(coords[1][1]-coords[0][1])*frac
            xs.append(x); ys.append(y); zs.append(dem_sample(dem,transform,valid,x,y))
        zs_arr=np.array([np.nan if z is None else z for z in zs],dtype=float)
        status="aceptada" if np.isfinite(zs_arr).sum()>=max(3,len(zs_arr)//2) else "revisar"
        sec=CrossSection(section_id=f"XS_{idx:04d}", name=f"XS km {d/1000:.3f}", km=d/1000,
                         line_xy=[(coords[0][0],coords[0][1]),(coords[1][0],coords[1][1])],
                         stations=[float(s) for s in ss], elevations=[float(z) if np.isfinite(z) else np.nan for z in zs_arr],
                         status_qa=status, notes="Sección natural desde DEM")
        finalize_section(sec)
        secs.append(sec)
    return secs

def make_prismatic_sections(line: LineString, profile: pd.DataFrame, spacing_m: float, kind: str, bottom_width: float, height: float, z_l: float, z_r: float, n: float) -> List[CrossSection]:
    ds=np.arange(0,line.length+0.01,max(spacing_m,1.0))
    secs=[]
    prof_km=profile["km"].values if not profile.empty else np.array([0,line.length/1000])
    prof_z=profile["cota_dem_m"].ffill().bfill().values if not profile.empty else np.array([0,0])
    for idx,d in enumerate(ds):
        km=d/1000
        bed=float(np.interp(km, prof_km, prof_z)) if len(prof_km)>0 else 0.0
        if kind.lower().startswith("rect"):
            stations=[-bottom_width/2,bottom_width/2]
            elev=[bed,bed]
            left=-bottom_width/2; right=bottom_width/2
        else:
            top_l=-(bottom_width/2+z_l*height); bot_l=-bottom_width/2; bot_r=bottom_width/2; top_r=bottom_width/2+z_r*height
            stations=[top_l,bot_l,bot_r,top_r]
            elev=[bed+height,bed,bed,bed+height]
            left=bot_l; right=bot_r
        sl=section_line(line,float(min(d,line.length)),max(abs(min(stations)),abs(max(stations))))
        sec=CrossSection(section_id=f"PR_{idx:04d}", name=f"Prismática km {km:.3f}", km=km,
                         line_xy=list(sl.coords), stations=[float(s) for s in stations], elevations=[float(z) for z in elev],
                         origin="modificada", source_topography="prismática", section_type=kind, manning_channel=n,
                         left_bank_station=left, right_bank_station=right, status_qa="aceptada", notes="Sección prismática")
        finalize_section(sec)
        secs.append(sec)
    return secs

def finalize_section(sec: CrossSection) -> CrossSection:
    stas=np.array(sec.stations,dtype=float); elev=np.array(sec.elevations,dtype=float)
    if stas.size and np.isfinite(elev).any():
        order=np.argsort(stas); stas=stas[order]; elev=elev[order]
        # remove duplicate stations
        keep=np.concatenate([[True],np.diff(stas)>1e-6])
        stas=stas[keep]; elev=elev[keep]
        sec.stations=[float(x) for x in stas]; sec.elevations=[float(z) if np.isfinite(z) else np.nan for z in elev]
        idx=int(np.nanargmin(elev)) if np.isfinite(elev).any() else 0
        sec.talweg_station=float(stas[idx]); sec.talweg_elevation=float(elev[idx])
        if sec.left_bank_station is None or sec.right_bank_station is None:
            sec.left_bank_station=float(np.nanpercentile(stas,25)); sec.right_bank_station=float(np.nanpercentile(stas,75))
        sec.total_width_m=float(np.nanmax(stas)-np.nanmin(stas)) if stas.size>1 else 0
        sec.main_channel_width_m=float(sec.right_bank_station-sec.left_bank_station) if sec.right_bank_station is not None and sec.left_bank_station is not None else None
        if not sec.original_stations:
            sec.original_stations=list(sec.stations); sec.original_elevations=list(sec.elevations)
    return qa_section(sec)

def qa_section(sec: CrossSection) -> CrossSection:
    warn=[]
    stas=np.array(sec.stations,dtype=float); elev=np.array(sec.elevations,dtype=float)
    if len(stas)<2: warn.append("Menos de dos puntos station-cota")
    if np.any(np.diff(stas)<=0): warn.append("Estaciones no crecientes o duplicadas")
    if np.isnan(elev).any(): warn.append("Cotas nulas/NoData")
    if sec.left_bank_station is not None and sec.right_bank_station is not None and sec.left_bank_station>=sec.right_bank_station:
        warn.append("Bancos invertidos")
    if sec.talweg_station is not None and sec.left_bank_station is not None and sec.right_bank_station is not None:
        if not (sec.left_bank_station <= sec.talweg_station <= sec.right_bank_station): warn.append("Talweg fuera del canal principal")
    if sec.total_width_m is not None and sec.total_width_m < 2: warn.append("Ancho insuficiente")
    if warn:
        sec.status_qa="revisar" if sec.status_qa=="aceptada" else sec.status_qa
        sec.notes=(sec.notes or "") + " | QA: " + "; ".join(warn)
    return sec

def sections_connectivity(sections: List[CrossSection]) -> pd.DataFrame:
    rows=[]
    sections=sorted(sections,key=lambda s:s.km)
    for i in range(len(sections)-1):
        a,b=sections[i],sections[i+1]
        dx=(b.km-a.km)*1000
        rows.append({"section_up_id":a.section_id,"section_down_id":b.section_id,"km_up":a.km,"km_down":b.km,
                     "longitud_channel_m":dx,"longitud_lob_m":dx,"longitud_rob_m":dx,
                     "z_talweg_up":a.talweg_elevation,"z_talweg_down":b.talweg_elevation,
                     "pendiente_talweg_pct":((a.talweg_elevation or 0)-(b.talweg_elevation or 0))/dx*100 if dx else np.nan})
    return pd.DataFrame(rows)

def section_model_df(sections: List[CrossSection]) -> pd.DataFrame:
    rows=[]
    for s in sections:
        rows.append({"section_id":s.section_id,"nombre":s.name,"pk_km":s.km,"origen":s.origin,"tipo_seccion":s.section_type,"estado_QA":s.status_qa,
                     "talweg_station":s.talweg_station,"cota_fondo":s.talweg_elevation,"banco_izquierdo":s.left_bank_station,"banco_derecho":s.right_bank_station,
                     "ancho_total_m":s.total_width_m,"ancho_canal_principal_m":s.main_channel_width_m,"Manning_LOB":s.manning_lob,"Manning_Channel":s.manning_channel,"Manning_ROB":s.manning_rob,
                     "contraccion":s.contraction_coeff,"expansion":s.expansion_coeff,"fuente_topografica":s.source_topography,"plan":s.plan,"observaciones":s.notes})
    return pd.DataFrame(rows)

def section_points_df(sections: List[CrossSection]) -> pd.DataFrame:
    rows=[]
    for s in sections:
        for sta,z in zip(s.stations,s.elevations):
            rows.append({"section_id":s.section_id,"pk_km":s.km,"station_m":sta,"cota_m":z})
    return pd.DataFrame(rows)

# ---------------------------- Plots/UI ---------------------------------------
def plot_hypsometric(df: pd.DataFrame):
    fig=go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["area_relativa_pct"], y=df["elevacion_m"], mode="lines", name="Curva hipsométrica"))
    fig.update_layout(xaxis_title="Área relativa sobre cota [%]", yaxis_title="Elevación [m]", height=350)
    return fig

def plot_profile(profile: pd.DataFrame, sections: Optional[List[CrossSection]]=None):
    fig=go.Figure()
    if not profile.empty:
        fig.add_trace(go.Scatter(x=profile["km"], y=profile["cota_dem_m"], mode="lines", name="Perfil DEM"))
    if sections:
        fig.add_trace(go.Scatter(x=[s.km for s in sections], y=[s.talweg_elevation for s in sections], mode="markers", name="Talweg secciones"))
    fig.update_layout(xaxis_title="km", yaxis_title="Cota [m]", height=350)
    return fig

def plot_section(sec: CrossSection):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=sec.stations,y=sec.elevations,mode="lines+markers",name=sec.name))
    for x,nm in [(sec.left_bank_station,"Banco izq"),(sec.talweg_station,"Talweg"),(sec.right_bank_station,"Banco der")]:
        if x is not None:
            fig.add_vline(x=x, annotation_text=nm)
    fig.update_layout(xaxis_title="Station [m]", yaxis_title="Cota [m]", height=380)
    return fig

def plot_sections_3d(profile: pd.DataFrame, sections: List[CrossSection]):
    fig=go.Figure()
    if not profile.empty:
        fig.add_trace(go.Scatter3d(x=profile["km"], y=np.zeros(len(profile)), z=profile["cota_dem_m"], mode="lines", name="Perfil eje", line=dict(width=5)))
    for s in sections[:300]:
        fig.add_trace(go.Scatter3d(x=[s.km]*len(s.stations), y=s.stations, z=s.elevations, mode="lines", name=f"km {s.km:.3f}", showlegend=False, line=dict(width=2)))
    fig.update_layout(height=650, scene=dict(xaxis_title="km", yaxis_title="Station [m]", zaxis_title="Cota [m]"))
    return fig

def render_satellite(axis: LineString, sections: List[CrossSection], pc1: Point, pc2: Point, to_wgs: Transformer, height=650):
    def ptll(x,y):
        lon,lat=to_wgs.transform(x,y); return [lat,lon]
    axis_ll=[ptll(x,y) for x,y in axis.coords]
    secs=[]
    for s in sections[:600]:
        if len(s.line_xy)>=2:
            secs.append({"km":s.km,"status":s.status_qa,"p0":ptll(*s.line_xy[0]),"p1":ptll(*s.line_xy[-1])})
    pc_ll=[]
    for nm,p in [("PC cuenca soporte",pc1),("PC hidrológico",pc2)]: pc_ll.append({"name":nm,"pt":ptll(p.x,p.y)})
    payload=json.dumps({"axis":axis_ll,"sections":secs,"points":pc_ll})
    html=f"""
    <div id='map' style='height:{height}px;'></div>
    <link rel='stylesheet' href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>
    <script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'></script>
    <script>
    const data={payload};
    const map=L.map('map',{{preferCanvas:true}});
    const img=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',{{maxZoom:19, attribution:'Esri World Imagery'}}).addTo(map);
    const osm=L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19, attribution:'OSM'}});
    L.control.layers({{'Satélite':img,'OSM':osm}},null,{{collapsed:false}}).addTo(map);
    let b=[];
    if(data.axis.length>1){{L.polyline(data.axis,{{color:'#ffd400',weight:5}}).addTo(map).bindPopup('Eje útil'); b=b.concat(data.axis);}}
    (data.sections||[]).forEach(s=>{{
      const col=s.status==='rechazada'?'#ff3333':(s.status==='revisar'?'#ffaa00':'#00e5ff');
      L.polyline([s.p0,s.p1],{{color:col,weight:2,opacity:0.85}}).addTo(map).bindPopup('Sección km '+s.km.toFixed(3)+'<br>'+s.status);
      const c=[(s.p0[0]+s.p1[0])/2,(s.p0[1]+s.p1[1])/2];
      L.marker(c,{{icon:L.divIcon({{className:'',html:'<div style="background:rgba(0,0,0,.65);color:white;padding:1px 4px;border-radius:4px;font:11px Arial;white-space:nowrap">km '+s.km.toFixed(3)+'</div>'}})}}).addTo(map);
      b.push(s.p0); b.push(s.p1);
    }});
    (data.points||[]).forEach(p=>{{L.marker(p.pt).addTo(map).bindPopup(p.name); b.push(p.pt);}});
    if(b.length) map.fitBounds(b,{{padding:[20,20]}}); else map.setView([-30,-71],10);
    </script>
    """
    components.html(html, height=height+20)

# ------------------------------ Exports --------------------------------------
def _kml_coords_line(line: LineString, to_wgs: Transformer) -> str:
    parts=[]
    for x,y in line.coords:
        lon,lat=to_wgs.transform(x,y); parts.append(f"{lon:.8f},{lat:.8f},0")
    return " ".join(parts)

def _kml_coords_poly(poly: Polygon, to_wgs: Transformer) -> str:
    parts=[]
    for x,y in poly.exterior.coords:
        lon,lat=to_wgs.transform(x,y); parts.append(f"{lon:.8f},{lat:.8f},0")
    return " ".join(parts)

def build_kmz(data: Dict[str,Any], crs_utm: CRS) -> bytes:
    to_wgs=tx(crs_utm,"EPSG:4326")
    kml=['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>HIDROSED_V1</name>']
    kml.append('<Style id="poly1"><LineStyle><color>ff00aa00</color><width>2</width></LineStyle><PolyStyle><color>3300aa00</color></PolyStyle></Style>')
    kml.append('<Style id="poly2"><LineStyle><color>ffaa0000</color><width>2</width></LineStyle><PolyStyle><color>33aa0000</color></PolyStyle></Style>')
    kml.append('<Style id="axis"><LineStyle><color>ff00ffff</color><width>4</width></LineStyle></Style>')
    kml.append('<Style id="xs"><LineStyle><color>ffffaa00</color><width>2</width></LineStyle></Style>')
    for nm,key,style in [("Cuenca soporte","basin_support","poly1"),("Subcuenca hidrológica","basin_hydro","poly2")]:
        poly=data.get(key,{}).get("polygon")
        if isinstance(poly,Polygon): kml.append(f'<Placemark><name>{nm}</name><styleUrl>#{style}</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>{_kml_coords_poly(poly,to_wgs)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
    for nm,key in [("Eje completo","axis_full"),("Eje útil","axis_useful")]:
        line=data.get(key)
        if isinstance(line,LineString): kml.append(f'<Placemark><name>{nm}</name><styleUrl>#axis</styleUrl><LineString><coordinates>{_kml_coords_line(line,to_wgs)}</coordinates></LineString></Placemark>')
    for s in data.get("sections",[]):
        line=LineString(s.line_xy)
        kml.append(f'<Placemark><name>{s.name}</name><styleUrl>#xs</styleUrl><LineString><coordinates>{_kml_coords_line(line,to_wgs)}</coordinates></LineString></Placemark>')
    kml.append('</Document></kml>')
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml","".join(kml))
    return buf.getvalue()

def export_excel(result: Dict[str,Any]) -> bytes:
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as writer:
        morphs=[]
        for key in ["morph_support","morph_hydro"]:
            m=result.get(key)
            if m: morphs.append(asdict(m))
        pd.DataFrame(morphs).to_excel(writer,"morfometria_cuencas",index=False)
        result.get("profile",pd.DataFrame()).to_excel(writer,"perfil_longitudinal",index=False)
        section_model_df(result.get("sections",[])).to_excel(writer,"modelo_geometrico",index=False)
        section_points_df(result.get("sections",[])).to_excel(writer,"hecras_station_elevation",index=False)
        sections_connectivity(result.get("sections",[])).to_excel(writer,"conectividad_hecras",index=False)
        result.get("hyps_support",pd.DataFrame()).to_excel(writer,"hipsometrica_soporte",index=False)
        result.get("hyps_hydro",pd.DataFrame()).to_excel(writer,"hipsometrica_hidrologica",index=False)
    return buf.getvalue()

def technical_database_json(result: Dict[str,Any]) -> bytes:
    def clean(o):
        if isinstance(o,(Polygon,LineString,Point)): return o.wkt
        if isinstance(o,pd.DataFrame): return o.to_dict(orient="records")
        if isinstance(o,np.generic): return o.item()
        if isinstance(o,float) and (math.isnan(o) or math.isinf(o)): return None
        if hasattr(o,"__dataclass_fields__"): return asdict(o)
        if isinstance(o,CRS): return o.to_string()
        raise TypeError(f"No serializable: {type(o)}")
    payload={
        "version":"HIDROSED_V1",
        "cuenca_morfometria_soporte": result.get("morph_support"),
        "cuenca_morfometria_hidrologica": result.get("morph_hydro"),
        "axis_info": result.get("axis_info"),
        "perfil_longitudinal": result.get("profile",pd.DataFrame()),
        "secciones_modelo": section_model_df(result.get("sections",[])),
        "secciones_station_elevation": section_points_df(result.get("sections",[])),
        "conectividad_hecras": sections_connectivity(result.get("sections",[])),
        "advertencias": result.get("warnings",[]),
    }
    return json.dumps(payload,ensure_ascii=False,indent=2,default=clean).encode("utf-8")

# --------------------------- App compute -------------------------------------
def compute_all(inputs: Dict[str,Any]) -> Dict[str,Any]:
    pc_support_name, pc_support_wgs = first_point(inputs["pc_support_file"], "PC cuenca soporte")
    pc_hydro_name, pc_hydro_wgs = first_point(inputs["pc_hydro_file"], "PC hidrológico")
    axis_name, axis_wgs = longest_line(inputs["axis_file"], "Eje del cauce")
    crs_utm = utm_crs_from_lonlat(pc_support_wgs.x, pc_support_wgs.y)
    pc_support = project_geom(pc_support_wgs,"EPSG:4326",crs_utm)
    pc_hydro = project_geom(pc_hydro_wgs,"EPSG:4326",crs_utm)
    axis_full = project_geom(axis_wgs,"EPSG:4326",crs_utm)
    tmpdir=tempfile.mkdtemp(prefix="hidrosed_v1_")
    if inputs["dem_source"].startswith("Open"):
        bounds=bbox_from_inputs(pc_support_wgs,pc_hydro_wgs,axis_wgs,inputs["margin_km"])
        dem_path=download_opentopo_dem(inputs["api_key"],inputs["demtype"],bounds,tmpdir,inputs["n_tiles"])
    else:
        dem_path=save_uploaded_dem(inputs["dem_file"],tmpdir)
    dem, transform, crs_utm, valid, dem_utm_path = reproject_dem_to_utm(dem_path,crs_utm,inputs["resolution_m"],tmpdir)
    axis_oriented, orientation, z_up, z_down=orient_axis(axis_full,dem,transform,valid,inputs["manual_invert"])
    axis_useful, km_s, km_h=useful_axis_between_pcs(axis_oriented,pc_support,pc_hydro)
    filled=priority_flood_fill(dem,valid)
    rec, valid_flat=compute_d8(filled,valid,transform)
    acc=compute_accumulation(rec,valid_flat)
    snap_radii=[float(x.strip()) for x in inputs["snap_radii"].split(",") if x.strip()]
    basin_support=delineate_basin(pc_support,"Cuenca soporte",dem,transform,valid,rec,acc,crs_utm,snap_radii,axis_oriented,inputs["axis_snap_buffer_m"])
    basin_hydro=delineate_basin(pc_hydro,"Subcuenca hidrológica",dem,transform,valid,rec,acc,crs_utm,snap_radii,axis_oriented,inputs["axis_snap_buffer_m"])
    morph_support=morphometry("Cuenca soporte",basin_support,dem,transform,valid)
    morph_hydro=morphometry("Subcuenca hidrológica",basin_hydro,dem,transform,valid)
    hyps_support=hypsometric_curve(dem,transform,basin_support["polygon"],valid)
    hyps_hydro=hypsometric_curve(dem,transform,basin_hydro["polygon"],valid)
    profile=generate_profile(axis_useful,dem,transform,valid,inputs["profile_step_m"])
    if inputs["section_mode"].startswith("Natural"):
        sections=make_natural_sections(axis_useful,dem,transform,valid,inputs["section_spacing_m"],inputs["half_width_m"],inputs["station_step_m"])
    else:
        sections=make_prismatic_sections(axis_useful,profile,inputs["section_spacing_m"],inputs["prism_kind"],inputs["bottom_width_m"],inputs["prism_height_m"],inputs["talud_l"],inputs["talud_r"],inputs["manning_n"])
    # length downstream metadata
    sections=sorted(sections,key=lambda s:s.km)
    for i,s in enumerate(sections[:-1]):
        dx=(sections[i+1].km-s.km)*1000
        s.length_downstream_m=dx; s.length_lob_m=dx; s.length_channel_m=dx; s.length_rob_m=dx
    warnings=[]
    warnings += morph_support.warnings + morph_hydro.warnings
    if axis_useful.length < 50: warnings.append("Tramo útil del eje menor a 50 m; revisar PC y eje.")
    axis_info=AxisInfo(axis_oriented.length,axis_useful.length,km_s,km_h,orientation,z_up,z_down,warnings=[])
    return dict(crs_utm=crs_utm, dem=dem, transform=transform, valid=valid, dem_path=dem_utm_path,
                pc_support=pc_support, pc_hydro=pc_hydro, axis_full=axis_oriented, axis_useful=axis_useful,
                basin_support=basin_support, basin_hydro=basin_hydro, morph_support=morph_support, morph_hydro=morph_hydro,
                hyps_support=hyps_support, hyps_hydro=hyps_hydro, profile=profile, sections=sections, axis_info=axis_info, warnings=warnings)

# ----------------------------- Streamlit UI ----------------------------------
def sidebar_inputs() -> Dict[str,Any]:
    st.sidebar.header("HIDROSED_V1 · Entradas")
    pc_support_file=st.sidebar.file_uploader("PC cuenca soporte/general · KMZ/KML", type=["kmz","kml"], key="pc_support")
    pc_hydro_file=st.sidebar.file_uploader("PC hidrológico/subcuenca · KMZ/KML", type=["kmz","kml"], key="pc_hydro")
    axis_file=st.sidebar.file_uploader("Eje del cauce · KMZ/KML obligatorio", type=["kmz","kml"], key="axis")
    st.sidebar.divider()
    dem_source=st.sidebar.radio("Fuente DEM", ["OpenTopography COP30/API", "GeoTIFF manual"], index=0)
    api_key=""; dem_file=None; demtype="COP30"; n_tiles=4; margin_km=20.0
    if dem_source.startswith("Open"):
        api_key=st.sidebar.text_input("API Key OpenTopography", type="password")
        demtype=st.sidebar.selectbox("DEM", ["COP30","NASADEM","SRTMGL1","SRTMGL3","AW3D30"], index=0)
        margin_km=st.sidebar.number_input("Margen descarga DEM [km]", min_value=1.0, max_value=120.0, value=25.0, step=5.0)
        n_tiles=st.sidebar.slider("DEM parciales", 1, 12, 4)
    else:
        dem_file=st.sidebar.file_uploader("DEM GeoTIFF", type=["tif","tiff"], key="dem")
    st.sidebar.divider()
    resolution_m=st.sidebar.selectbox("Resolución interna [m]", [30,60,90,120], index=1)
    snap_radii=st.sidebar.text_input("Radios ajuste drenaje [m]", value="150,300,600,1000")
    axis_snap_buffer_m=st.sidebar.number_input("Corredor ajuste al eje [m]", min_value=50.0, max_value=5000.0, value=700.0, step=50.0)
    manual_invert=st.sidebar.checkbox("Invertir sentido del eje manualmente", value=False)
    st.sidebar.divider()
    profile_step_m=st.sidebar.number_input("Paso perfil longitudinal [m]", min_value=5.0, max_value=1000.0, value=50.0, step=5.0)
    section_mode=st.sidebar.selectbox("Modo secciones", ["Natural desde DEM", "Prismática rectangular/trapecial"], index=0)
    section_spacing_m=st.sidebar.number_input("Separación secciones [m]", min_value=5.0, max_value=2000.0, value=100.0, step=5.0)
    half_width_m=st.sidebar.number_input("Semi-ancho sección natural [m]", min_value=5.0, max_value=5000.0, value=100.0, step=5.0)
    station_step_m=st.sidebar.number_input("Paso station [m]", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
    prism_kind=st.sidebar.selectbox("Tipo prismático", ["trapecial","rectangular"], index=0)
    bottom_width_m=st.sidebar.number_input("Ancho fondo [m]", min_value=0.1, max_value=1000.0, value=5.0, step=0.5)
    prism_height_m=st.sidebar.number_input("Altura canal [m]", min_value=0.1, max_value=100.0, value=2.0, step=0.1)
    talud_l=st.sidebar.number_input("Talud izq H:V", min_value=0.0, max_value=50.0, value=1.5, step=0.1)
    talud_r=st.sidebar.number_input("Talud der H:V", min_value=0.0, max_value=50.0, value=1.5, step=0.1)
    manning_n=st.sidebar.number_input("Manning canal", min_value=0.010, max_value=0.200, value=0.035, step=0.001, format="%.3f")
    return locals()

def section_editor(result: Dict[str,Any]):
    sections=result["sections"]
    if not sections:
        st.info("No hay secciones generadas."); return
    labels=[f"km {s.km:.3f} · {s.status_qa} · {s.section_id}" for s in sections]
    idx=st.selectbox("Seleccionar sección", range(len(sections)), format_func=lambda i: labels[i])
    sec=sections[idx]
    col1,col2=st.columns([1.2,1])
    with col1:
        st.plotly_chart(plot_section(sec), use_container_width=True)
    with col2:
        st.markdown(f"**{sec.name}**")
        sec.status_qa=st.selectbox("Estado QA", ["aceptada","revisar","corregida","rechazada"], index=["aceptada","revisar","corregida","rechazada"].index(sec.status_qa) if sec.status_qa in ["aceptada","revisar","corregida","rechazada"] else 1)
        sec.left_bank_station=st.number_input("Banco izquierdo station", value=float(sec.left_bank_station or min(sec.stations)), step=0.5)
        sec.right_bank_station=st.number_input("Banco derecho station", value=float(sec.right_bank_station or max(sec.stations)), step=0.5)
        sec.talweg_station=st.number_input("Talweg station", value=float(sec.talweg_station or 0), step=0.5)
        sec.manning_lob=st.number_input("Manning LOB", value=float(sec.manning_lob), step=0.001, format="%.3f")
        sec.manning_channel=st.number_input("Manning Channel", value=float(sec.manning_channel), step=0.001, format="%.3f")
        sec.manning_rob=st.number_input("Manning ROB", value=float(sec.manning_rob), step=0.001, format="%.3f")
    df=pd.DataFrame({"station_m":sec.stations,"cota_m":sec.elevations})
    edited=st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"edit_{sec.section_id}")
    b1,b2,b3,b4=st.columns(4)
    if b1.button("Aplicar cambios"):
        sec.stations=edited["station_m"].astype(float).tolist(); sec.elevations=edited["cota_m"].astype(float).tolist()
        sec.origin="modificada"; sec.status_qa="corregida"; sec.change_log.append({"metodo":"edición directa","section_id":sec.section_id})
        sections[idx]=finalize_section(sec); result["sections"]=sections; st.session_state["hidrosed_result"]=result; st.rerun()
    if b2.button("Eliminar sección"):
        del sections[idx]; result["sections"]=sections; st.session_state["hidrosed_result"]=result; st.rerun()
    if b3.button("Duplicar sección"):
        new=CrossSection(**asdict(sec)); new.section_id=f"{sec.section_id}_copy"; new.km+=0.001; new.name+= " copia"; new.origin="modificada"
        sections.append(new); sections.sort(key=lambda s:s.km); result["sections"]=sections; st.session_state["hidrosed_result"]=result; st.rerun()
    if b4.button("Restaurar original"):
        if sec.original_stations:
            sec.stations=list(sec.original_stations); sec.elevations=list(sec.original_elevations); sec.status_qa="aceptada"; sections[idx]=finalize_section(sec); st.session_state["hidrosed_result"]=result; st.rerun()
    st.divider()
    st.subheader("Insertar sección por puntos directamente")
    st.caption("Ingrese km + station + cota. Coordenadas x/y son opcionales; sirven como trazabilidad topográfica.")
    direct=st.data_editor(pd.DataFrame({"km":[float(sec.km)]*3,"station_m":[-10.0,0.0,10.0],"x":[np.nan]*3,"y":[np.nan]*3,"cota_m":[np.nan]*3}), num_rows="dynamic", use_container_width=True, key="direct_points")
    if st.button("Insertar sección por puntos directos"):
        ddf=direct.dropna(subset=["station_m","cota_m"])
        for km,g in ddf.groupby("km"):
            d=float(km)*1000; sl=section_line(result["axis_useful"], min(max(d,0), result["axis_useful"].length), max(abs(g["station_m"]).max(),1))
            new=CrossSection(section_id=f"INS_{len(sections):04d}", name=f"Insertada km {float(km):.3f}", km=float(km), line_xy=list(sl.coords), stations=g["station_m"].astype(float).tolist(), elevations=g["cota_m"].astype(float).tolist(), origin="modificada", source_topography="puntos directos", section_type="insertada", status_qa="corregida", notes="Ingresada directamente en app")
            sections.append(finalize_section(new))
        sections.sort(key=lambda s:s.km); result["sections"]=sections; st.session_state["hidrosed_result"]=result; st.success("Secciones insertadas."); st.rerun()

def main():
    st.title("HIDROSED_V1 · Base modular")
    st.caption("Morfometría de cuenca + eje del cauce + secciones. Preparado para módulos futuros de hidrología, hidráulica, sedimentos y socavación.")
    inputs=sidebar_inputs()
    required=[inputs["pc_support_file"],inputs["pc_hydro_file"],inputs["axis_file"]]
    dem_ok=(inputs["dem_source"].startswith("Open") and bool(inputs["api_key"])) or (inputs["dem_source"].startswith("Geo") and inputs["dem_file"] is not None)
    if st.sidebar.button("Procesar HIDROSED_V1", type="primary"):
        if not all(required):
            st.error("Debe cargar PC cuenca soporte, PC hidrológico y eje del cauce."); st.stop()
        if not dem_ok:
            st.error("Debe ingresar API Key OpenTopography o cargar DEM GeoTIFF."); st.stop()
        with st.spinner("Procesando DEM, delimitación, morfometría, eje y secciones..."):
            result=compute_all(inputs)
        st.session_state["hidrosed_result"]=result
        st.success("Procesamiento finalizado.")
    if "hidrosed_result" not in st.session_state:
        st.info("Carga los tres insumos obligatorios y el DEM/API Key, luego presiona **Procesar HIDROSED_V1**.")
        st.stop()
    result=st.session_state["hidrosed_result"]
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["1 · Cuenca y morfometría","2 · Eje y perfil","3 · Secciones","4 · Vista satelital","5 · Base técnica","6 · Descargas"])
    with tab1:
        m1=result["morph_support"]; m2=result["morph_hydro"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Área soporte [km²]", f"{m1.area_km2:,.2f}")
        c2.metric("Área hidrológica [km²]", f"{m2.area_km2:,.2f}")
        c3.metric("Pendiente media soporte [%]", f"{m1.mean_slope_pct:,.2f}")
        c4.metric("Confianza soporte [%]", f"{m1.confidence_pct:,.0f}")
        st.dataframe(pd.DataFrame([asdict(m1),asdict(m2)]), use_container_width=True)
        h1,h2=st.columns(2)
        h1.plotly_chart(plot_hypsometric(result["hyps_support"]), use_container_width=True)
        h2.plotly_chart(plot_hypsometric(result["hyps_hydro"]), use_container_width=True)
        if result["warnings"]:
            st.warning(" | ".join(result["warnings"]))
    with tab2:
        ax=result["axis_info"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Longitud eje útil [km]", f"{ax.length_useful_m/1000:,.3f}")
        c2.metric("km PC soporte", f"{ax.km_pc_support:,.3f}")
        c3.metric("km PC hidrológico", f"{ax.km_pc_hydro:,.3f}")
        c4.metric("Orientación", ax.orientation)
        st.plotly_chart(plot_profile(result["profile"], result["sections"]), use_container_width=True)
        st.dataframe(result["profile"], use_container_width=True, height=350)
    with tab3:
        st.subheader("Secciones transversales y edición")
        st.dataframe(section_model_df(result["sections"]), use_container_width=True, height=250)
        section_editor(result)
        st.subheader("Perfil 3D")
        st.plotly_chart(plot_sections_3d(result["profile"], result["sections"]), use_container_width=True)
    with tab4:
        render_satellite(result["axis_useful"], result["sections"], result["pc_support"], result["pc_hydro"], tx(result["crs_utm"],"EPSG:4326"))
    with tab5:
        st.subheader("Base técnica extraíble para módulos futuros")
        st.markdown("Estas tablas alimentan directamente módulos de hidrología, hidráulica 1D, sedimentos, socavación e informe técnico.")
        st.dataframe(section_model_df(result["sections"]), use_container_width=True)
        st.dataframe(sections_connectivity(result["sections"]), use_container_width=True)
        st.json(json.loads(technical_database_json(result).decode("utf-8")))
    with tab6:
        st.download_button("Descargar base técnica JSON", technical_database_json(result), file_name="HIDROSED_V1_base_tecnica.json", mime="application/json")
        st.download_button("Descargar Excel técnico", export_excel(result), file_name="HIDROSED_V1_base_tecnica.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Descargar KMZ", build_kmz(result,result["crs_utm"]), file_name="HIDROSED_V1_cuenca_eje_secciones.kmz", mime="application/vnd.google-earth.kmz")

if __name__ == "__main__":
    main()
