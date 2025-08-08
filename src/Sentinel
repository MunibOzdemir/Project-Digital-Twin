# %%
import requests  # HTTP-verzoeken sturen naar Sentinel Hub API
import json  # GeoJSON-bestand inlezen als Python-dict
import numpy as np  # Numerieke berekeningen en array-manipulatie
from io import BytesIO  # BytesIO om binair beeldmateriaal in geheugen te laden
from PIL import Image  # Image openen en converteren naar NumPy-array
import matplotlib.pyplot as plt  # Visualisaties maken
from datetime import datetime, timedelta  # Datums manipuleren

# --- Configuratie & credentials ---
CLIENT_ID = "sh-93c6fbd0-8c4a-4e40-8c59-d06889413797"
CLIENT_SECRET = "LKVq6MTE0S3kohQjRI1Yuj03aU5frOTm"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Pad naar je GeoJSON-bestand met gebied van interesse
GEOJSON_PATH = "../data/alkmaar.geojson"

# Maximale wolkendekking in procenten (stel gerust hoger in bij weinig beelden)
MAX_CLOUD = 1

# --- GeoJSON inlezen en juiste geometrie extracten ---
with open(GEOJSON_PATH) as f:
    gj = json.load(f)
# Indien 'features' key bestaat, pak de eerste feature; anders direct geometry of hele object
geom = gj["features"][0]["geometry"] if "features" in gj else gj.get("geometry", gj)


# --- Functie om toegangstoken te verkrijgen bij Sentinel Hub ---
def get_token():
    """
    Authenticeert met client_credentials grant.
    Returned een bearer token voor verdere API-aanroepen.
    """
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
    )
    r.raise_for_status()  # Stop bij foutstatus
    return r.json()["access_token"]


# Haal token eenmaal op en bouw headers met Authorization
token = get_token()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# --- Generieke functie om Sentinel-2 data te fetchen voor willekeurige banden ---
def fetch_s2(
    bands,
    start_iso,
    end_iso,
    width=512,
    height=512,
    max_cloud=MAX_CLOUD,
    mosaicking_order="leastCC",
):
    """
    Haalt pixelreflecties op voor opgegeven 'bands' in de opgegeven tijdsperiode.
    bands         : lijst van band-namen (bijv. ["B04","B03","B02"])
    start_iso     : ISO-tijdstring beginperiode
    end_iso       : ISO-tijdstring eindperiode
    width, height : resolutie van het beeld in pixels
    max_cloud     : maximum toegestane wolkendekking (%)
    mosaicking_order: 'leastCC' of 'mostRecent'

    Returned een NumPy-array shape=(h, w, len(bands)), dtype=float32, waarden [0,1].
    """
    # Bouw het evalscript dynamisch om precies de gevraagde bands terug te krijgen
    evalscript = f"""
    //VERSION=3
    function setup() {{
      return {{
        input: {bands},
        output: {{ bands: {len(bands)} }}
      }};
    }}
    function evaluatePixel(sample) {{
      return [{', '.join(f"sample.{b}" for b in bands)}];
    }}
    """

    # Payload met bounds, tijdsfilter en processing-opties
    payload = {
        "input": {
            "bounds": {
                "geometry": geom,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "S2L2A",  # Sentinel-2 Level-2A producten
                    "dataFilter": {
                        "timeRange": {"from": start_iso, "to": end_iso},
                        "maxCloudCoverage": max_cloud,
                    },
                    "processing": {"mosaickingOrder": mosaicking_order},
                }
            ],
        },
        "evalscript": evalscript,
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
    }

    # Verstuur request, krijg PNG terug als bytes
    r = requests.post(PROCESS_URL, headers=headers, json=payload)
    r.raise_for_status()

    # Open de PNG in geheugen en zet om naar NumPy-array
    img = Image.open(BytesIO(r.content))
    arr = np.array(img, dtype=np.uint8)

    # Sommige PNG's bevatten een alpha-kanaal, hier strips we dat weg
    if arr.ndim == 3 and arr.shape[2] > len(bands):
        arr = arr[..., : len(bands)]

    # Schaal de 0–255 waarden naar 0.0–1.0 floats voor verdere berekening
    return arr.astype(np.float32) / 255.0


# --- Hulpfuncties voor specifieke doelen ---
def fetch_true_color(start_iso, end_iso):
    """
    Haalt true-color (RGB) beeld op: B04 (rood), B03 (groen), B02 (blauw).
    """
    return fetch_s2(["B04", "B03", "B02"], start_iso, end_iso)


def fetch_ndci(start_iso, end_iso):
    """
    Bereken de Normalized Difference Chlorophyll Index:
      NDCI = (B05 - B04) / (B05 + B04)
    B05 = red-edge band (kanaal 0), B04 = rood (kanaal 1).
    """
    arr = fetch_s2(["B05", "B04"], start_iso, end_iso)
    b5, b4 = arr[..., 0], arr[..., 1]
    # Voeg klein getal toe om delen door nul te voorkomen
    return (b5 - b4) / (b5 + b4 + 1e-6)


# --- 1) True-color visualisatie voor exact twee periodes ---
dates_tc = {
    "Mei 2024": ("2024-05-01T00:00:00Z", "2024-05-31T23:59:59Z"),
    "April 2025": ("2025-04-01T00:00:00Z", "2025-04-30T23:59:59Z"),
}

# Haal beide RGB-beelden op en bewaar in dict
tc_maps = {
    label: fetch_true_color(start, end) for label, (start, end) in dates_tc.items()
}

# Plotten in één rij van 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, (label, img) in zip(axes, tc_maps.items()):
    ax.imshow(img)
    ax.set_title(f"True-color {label}")
    ax.axis("off")  # geen assen tonen voor overzichtelijkheid

plt.tight_layout()
plt.show()

# --- 2) Berekening en visualisatie van NDCI en ΔNDCI ---
# Zelfde periodes als true-color voor consistente vergelijking
dates_ndci = dates_tc

# Haal NDCI-kaarten op
ndci_maps = {
    label: fetch_ndci(start, end) for label, (start, end) in dates_ndci.items()
}

# Bereken verschil tussen April 2025 en Mei 2024
delta_ndci = ndci_maps["April 2025"] - ndci_maps["Mei 2024"]

# Maak 3 subplots: NDCI Mei 2024, NDCI April 2025, én Δ NDCI
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Definieer plotinstellingen voor elke as
settings = [
    (ndci_maps["Mei 2024"], "NDCI Mei 2024", "Spectral", (-1, 1)),
    (ndci_maps["April 2025"], "NDCI April 2025", "Spectral", (-1, 1)),
    (delta_ndci, "Δ NDCI", "RdBu", (-1, 1)),
]

# Plot alle drie, en bewaar de imshow-handles voor colorbar
im_handles = []
for ax, (data, title, cmap, vlim) in zip(axes, settings):
    im = ax.imshow(data, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    ax.set_title(title)
    ax.axis("off")
    im_handles.append(im)

# Voeg één colorbar toe bij ΔNDCI-plot (rechter subplot)
fig.colorbar(im_handles[-1], ax=axes[-1], shrink=0.7)

plt.tight_layout()
plt.show()
