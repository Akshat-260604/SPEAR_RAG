"""
geo_parser.py
─────────────
Parses natural language spatial queries (English) to extract:
  - location name + bounding box (via geopy Nominatim)
  - manual coordinates (lat,lon or lat/lon)
  - year / year range  (defaults to full available range: 2020–2024)
  - month (if specified)
  - query type: landcover | crop | flood | drought | fire | general

Returns a SpatialQuery dataclass.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# ─── Data class ───────────────────────────────────────────────────────────────

# Data availability range
DATA_YEAR_MIN = 2020
DATA_YEAR_MAX = 2024

@dataclass
class SpatialQuery:
    raw_query:   str
    location:    str                              # human-readable name
    bbox:        Tuple[float,float,float,float]   # (lat_min, lat_max, lon_min, lon_max)
    year_start:  Optional[int]   = None
    year_end:    Optional[int]   = None
    month:       Optional[int]   = None
    query_type:  str             = "general"      # landcover | crop | flood | drought | fire | general
    keywords:    List[str]       = field(default_factory=list)

    def __str__(self):
        return (
            f"Location: {self.location}\n"
            f"BBox: lat[{self.bbox[0]:.3f}–{self.bbox[1]:.3f}] "
            f"lon[{self.bbox[2]:.3f}–{self.bbox[3]:.3f}]\n"
            f"Year: {self.year_start}–{self.year_end}  Month: {self.month}\n"
            f"QueryType: {self.query_type}"
        )


# ─── Constants ────────────────────────────────────────────────────────────────

MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12,
}

QUERY_TYPE_KEYWORDS = {
    "landcover": ["landcover","land cover","land use","land class","classification","what is there","what land"],
    "crop":      ["crop","agriculture","farming","farm","vegetation","ndvi","crop condition","crop health","yield","kharif","rabi","paddy","wheat","rice"],
    "flood":     ["flood","flooding","inundation","deluge","waterlogged","submerged"],
    "drought":   ["drought","dry","rainfall deficit","water scarcity","arid"],
    "fire":      ["fire","wildfire","burn","forest fire"],
    "heatwave":  ["heat","temperature","heatwave","hot","lst"],
    "water":     ["water body","river","lake","reservoir","waterbody","ndwi"],
}

# Known India region → rough bbox expansions (fallback if geocoder is slow)
KNOWN_REGIONS = {
    "delhi":       (28.40, 28.90, 76.80, 77.40),
    "mumbai":      (18.85, 19.30, 72.75, 73.00),
    "bangalore":   (12.80, 13.10, 77.40, 77.80),
    "chennai":     (12.90, 13.25, 80.10, 80.35),
    "kolkata":     (22.40, 22.70, 88.20, 88.55),
    "hyderabad":   (17.20, 17.65, 78.20, 78.65),
    "assam":       (24.00, 28.00, 89.70, 96.00),
    "kerala":      ( 8.30, 12.80, 74.80, 77.50),
    "odisha":      (17.80, 22.60, 81.30, 87.50),
    "bihar":       (24.30, 27.60, 83.30, 88.30),
    "rajasthan":   (23.00, 30.20, 69.50, 78.30),
    "uttarakhand": (28.70, 31.50, 77.60, 81.10),
    "gujarat":     (20.10, 24.70, 68.20, 74.50),
    "punjab":      (29.50, 32.50, 73.90, 76.70),
    "haryana":     (27.60, 30.90, 74.50, 77.60),
    "india":       ( 6.50, 37.10, 68.00, 97.50),
}


# ─── Geocoder ─────────────────────────────────────────────────────────────────

_geocoder = Nominatim(user_agent="spear_rag_v1", timeout=10)

def _geocode(location: str) -> Optional[Tuple[float,float,float,float]]:
    """Return (lat_min, lat_max, lon_min, lon_max) for a location string."""
    # Check known regions first (fast, no network)
    key = location.lower().strip()
    for region, bbox in KNOWN_REGIONS.items():
        if region in key or key in region:
            return bbox

    # Fallback: Nominatim geocoding
    try:
        result = _geocoder.geocode(location + ", India", exactly_one=True)
        if result is None:
            result = _geocoder.geocode(location, exactly_one=True)
        if result is None:
            return None

        bboxes = result.raw.get("boundingbox")
        if bboxes:
            lat_min, lat_max, lon_min, lon_max = map(float, bboxes)
            # Add a small buffer (0.3°) for cities
            buf = 0.3
            return (lat_min - buf, lat_max + buf, lon_min - buf, lon_max + buf)
        else:
            lat, lon = result.latitude, result.longitude
            buf = 0.5
            return (lat - buf, lat + buf, lon - buf, lon + buf)
    except GeocoderTimedOut:
        return None


# ─── Helper extractors ────────────────────────────────────────────────────────

def _extract_years(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract year range, clamped to available data (2020–2024)."""
    years = [int(y) for y in re.findall(r'\b((?:19|20)\d{2})\b', text)]
    # Clamp to data availability window
    years = [y for y in years if DATA_YEAR_MIN <= y <= DATA_YEAR_MAX]
    if not years:
        return DATA_YEAR_MIN, DATA_YEAR_MAX   # default: all available years
    return min(years), max(years)


def _extract_coords(text: str) -> Optional[Tuple[float,float,float,float]]:
    """Detect manual coordinates like '28.6,77.2' or '28.6N 77.2E' and return bbox."""
    # Pattern: two floats close together (lat, lon)
    m = re.search(r'(\d{1,2}\.\d+)\s*[,/\s]\s*(\d{2,3}\.\d+)', text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if 6.5 <= lat <= 37.1 and 68.0 <= lon <= 97.5:   # India bounds check
            buf = 0.5
            return (lat - buf, lat + buf, lon - buf, lon + buf)
    return None


def _extract_month(text: str) -> Optional[int]:
    text_lower = text.lower()
    for word, num in MONTH_MAP.items():
        if word in text_lower:
            return num
    return None


def _extract_location(text: str) -> Optional[str]:
    """
    Simple heuristic: find capitalised words or known region names.
    Also handles Hindi-to-English common place names.
    """
    # Try known regions first
    text_lower = text.lower()
    for region in KNOWN_REGIONS:
        if region in text_lower:
            return region.capitalize()

    # Look for capitalized words (English place names)
    caps = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    _stop = {
        "what","where","when","which","how","the","was","did",
        "india","flood","drought","crop","fire","heat","year",
        # common query verbs / starters
        "give","show","analyze","analyse","find","tell","get",
        "check","display","calculate","compute","fetch","list",
        "explain","describe","report","compare","use","using",
        "land","cover","crop","flood","near","for","and","with",
        "season","rabi","kharif","status","extent","health",
        "sentinel","planet","climate","area","region","district",
        "latitude","longitude","coordinates","ndvi","ndwi","lst",
    }
    for cap in caps:
        if cap.lower() not in _stop:
            return cap

    return None


def _extract_query_type(text: str) -> str:
    text_lower = text.lower()
    for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return qtype
    return "general"


# ─── Main parser ──────────────────────────────────────────────────────────────

def parse_query(query: str) -> SpatialQuery:
    """
    Parse a natural language spatial query (English) into a SpatialQuery object.

    Supports:
      - Place names: "Delhi", "Punjab", "Assam"
      - Coordinates: "28.6, 77.2" or "28.6N 77.2E"
      - Year range (clamped to 2020–2024, defaults to full range)
      - Query types: landcover, crop, flood, drought, fire, heatwave, water
    """
    # Try manual coordinates first
    coord_bbox = _extract_coords(query)
    location_name = _extract_location(query)
    year_start, year_end = _extract_years(query)
    month       = _extract_month(query)
    query_type  = _extract_query_type(query)

    # Determine bbox: coordinates > named location > all-India
    if coord_bbox is not None:
        bbox = coord_bbox
        location_name = location_name or f"{coord_bbox[0]:.2f}N,{coord_bbox[2]:.2f}E"
    else:
        bbox = None
        if location_name:
            bbox = _geocode(location_name)
        if bbox is None:
            print(f"[GeoParser] Could not geocode '{location_name}' — defaulting to all-India")
            bbox = KNOWN_REGIONS["india"]
            location_name = location_name or "India"

    return SpatialQuery(
        raw_query  = query,
        location   = location_name or "Unknown",
        bbox       = bbox,
        year_start = year_start,
        year_end   = year_end,
        month      = month,
        query_type = query_type,
        keywords   = list(QUERY_TYPE_KEYWORDS.get(query_type, [])),
    )


if __name__ == "__main__":
    test_queries = [
        "Delhi region mein 2020 mein flood ka kya situation tha",
        "What was the drought situation in Rajasthan in 2018-2019?",
        "Kerala mein 2018 mein flooding kaise thi",
        "Show me crop stress in Punjab during August 2022",
        "Assam floods 2022",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = parse_query(q)
        print(result)
