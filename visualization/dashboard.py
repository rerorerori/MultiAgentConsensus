"""
Dashboard — görev kontrol arayüzü (tkinter)
===========================================
Düzen:
  üst şerit : görev süresi, kapsama, harita uzlaşması, paket teslimi, seyrüsefer hatası, akıntı
  sol panel : araç kartları, harita katmanları
  orta      : harita (Doğu sağda, Kuzey yukarıda)
  sağ panel : seyrüsefer tablosu, akıntı pusulası, akustik linkler, eğilim grafikleri, olay kaydı

Koordinatlar: simülasyon NWU çerçevesindedir (x = kuzey, y = batı). Haritada doğu = -y.

Ana döngüden kullanım:
    dashboard = Dashboard(consensus, comm_manager=comm_mgr)
    dashboard.sensor_data[name] = {...}
    dashboard.trigger_beacon_pulse(sender, receiver, snr)
    dashboard.comm_log.append("mesaj")
    dashboard.update(states, targets, comm_quality, "proposed", sim_time=sim_time)
"""

import hashlib
import io
import math
import os
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import urllib.request
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageTk

from config import AGENT_NAMES, MAP_SIZE_M, SENSOR_RANGE

# ── Tema ─────────────────────────────────────────────────────────────
BG       = "#0b1220"
PANEL    = "#101a2c"
CARD     = "#16233a"
CARD_SEL = "#1e3252"
BORDER   = "#26385a"
AREA     = "#0f1d33"    # çalışma alanı zemini
TEXT     = "#e6edf7"
MUTED    = "#93a3bd"
FAINT    = "#5b6c88"
ACCENT   = "#38bdf8"
GOOD     = "#34d399"
WARN     = "#fbbf24"
BAD      = "#f87171"
COVERED  = (45, 212, 191)   # taranmış hücre rengi (RGB)
COVERED_HEX = "#%02x%02x%02x" % COVERED

_PALETTE = ["#38bdf8", "#fb923c", "#c084fc", "#f472b6", "#a3e635", "#facc15"]
AGENT_COLORS = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(AGENT_NAMES)}

STATE_COLORS = {"CONNECTED": GOOD, "DEGRADED": WARN, "RECONNECTING": ACCENT, "OFFLINE": BAD}
MISSION_COLORS = {"MAPPING": ACCENT, "RETURNING": WARN, "HOME": GOOD}

LEFT_W, RIGHT_W = 340, 400
TRAIL_LEN = 6000            # iz uzunluğu (arayüz güncellemesi başına bir nokta)
LINK_FLASH_S = 2.5          # gönderilen paketin haritada vurgulanma süresi (sim s)
COVERAGE_MILESTONES = (25, 50, 75, 90, 99)

# ── Uydu görüntüsü ───────────────────────────────────────────────────
# Simülasyonun (0, 0) noktasının haritadaki karşılığı: Sığacık Körfezi
MAP_DATUM_LAT, MAP_DATUM_LON = 38.1800, 26.7676
EARTH_R = 6378137.0
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_CACHE_DIR = os.path.expanduser("~/.cache/swarm_dashboard/tiles")
TILE_ZOOM_RANGE = (11, 18)
SATELLITE_BRIGHTNESS = 0.62  # üstteki çizimlerin okunması için görüntü karartılır

F = {}                      # yazı tipleri, pencere açılınca doldurulur


def _local_to_latlon(east_m, north_m):
    return (MAP_DATUM_LAT + math.degrees(north_m / EARTH_R),
            MAP_DATUM_LON + math.degrees(east_m / (EARTH_R * math.cos(math.radians(MAP_DATUM_LAT)))))


def _latlon_to_local(lat, lon):
    return (math.radians(lon - MAP_DATUM_LON) * EARTH_R * math.cos(math.radians(MAP_DATUM_LAT)),
            math.radians(lat - MAP_DATUM_LAT) * EARTH_R)


def _deg2tile(lat, lon, zoom):
    n = 2 ** zoom
    return (int((lon + 180.0) / 360.0 * n),
            int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))


def _tile2deg(tx, ty, zoom):
    n = 2 ** zoom
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))), tx / n * 360.0 - 180.0


class SatelliteTiles:
    """Uydu karolarını diskten okur; yoksa arka planda indirip diske yazar."""

    def __init__(self):
        self.version = 0                 # her yeni karoda artar (önbellek anahtarı)
        self._mem, self._pending = {}, set()
        self._lock = threading.Lock()
        os.makedirs(TILE_CACHE_DIR, exist_ok=True)

    def _path(self, key):
        name = "esri_satellite_%d_%d_%d" % key
        return os.path.join(TILE_CACHE_DIR, hashlib.md5(name.encode()).hexdigest() + ".png")

    def get(self, z, x, y):
        """Karo hazırsa PIL görüntüsü, değilse None döndürür ve indirmeyi başlatır."""
        key = (z, x, y)
        with self._lock:
            if key in self._mem:
                return self._mem[key]
            if key in self._pending:
                return None
            self._pending.add(key)
        threading.Thread(target=self._load, args=(key,), daemon=True).start()
        return None

    def _load(self, key):
        path, img = self._path(key), None
        try:
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
            else:
                z, x, y = key
                req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y),
                                             headers={"User-Agent": "swarm-auv-dashboard/1.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = resp.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                with open(path, "wb") as f:
                    f.write(data)
        except Exception:
            img = None                   # ağ yoksa karo boş kalır; sonraki çizimde yeniden denenir
        with self._lock:
            self._pending.discard(key)
            if img is not None:
                self._mem[key] = img
                self.version += 1


def _init_fonts(root):
    families = set(tkfont.families(root))
    ui = next((f for f in ("Segoe UI", "Helvetica Neue", "DejaVu Sans") if f in families), "TkDefaultFont")
    mono = next((f for f in ("Cascadia Mono", "Consolas", "DejaVu Sans Mono") if f in families), "TkFixedFont")
    F.update(
        ui=(ui, 9), small=(ui, 8), small_b=(ui, 8, "bold"),
        head=(ui, 11, "bold"), title=(ui, 13, "bold"), kpi=(ui, 17, "bold"),
        mono=(mono, 9), mono_s=(mono, 8), mono_b=(mono, 9, "bold"))


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _blend(hex_a, hex_b, t):
    """hex_a'dan hex_b'ye doğrusal karışım (t=0 → a, t=1 → b)."""
    a, b = _rgb(hex_a), _rgb(hex_b)
    return "#%02x%02x%02x" % tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _fmt_clock(seconds):
    m, s = divmod(int(max(seconds, 0)), 60)
    return f"{m:02d}:{s:02d}"


def _compass_deg(yaw_nwu_deg):
    """NWU yaw (saat yönünün tersine pozitif) → pusula baş açısı (0 = kuzey, 90 = doğu)."""
    return (-yaw_nwu_deg) % 360.0


def _bind_tree(widget, sequence, callback):
    widget.bind(sequence, callback)
    for child in widget.winfo_children():
        _bind_tree(child, sequence, callback)


def _set_bg_tree(widget, color, skip=()):
    if widget in skip:
        return
    widget.configure(bg=color)
    for child in widget.winfo_children():
        _set_bg_tree(child, color, skip)


class EventLog:
    """Ana döngünün `append` ile yazdığı, arayüzün okuduğu zaman damgalı olay listesi."""

    def __init__(self):
        self.sim_time = 0.0
        self._pending = deque(maxlen=500)

    def append(self, message):
        self._pending.append((self.sim_time, str(message)))

    def pop_new(self):
        new = list(self._pending)
        self._pending.clear()
        return new


class Bar(tk.Canvas):
    """İnce yatay doluluk çubuğu."""

    def __init__(self, parent, width=120, height=6, bg=CARD):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0)
        self._bar_w, self._bar_h = width, height
        self.set(0.0, FAINT)

    def set(self, fraction, color):
        fraction = min(max(fraction, 0.0), 1.0)
        self.delete("all")
        self.create_rectangle(0, 0, self._bar_w, self._bar_h, fill=BORDER, width=0)
        if fraction > 0:
            self.create_rectangle(0, 0, max(2, self._bar_w * fraction), self._bar_h, fill=color, width=0)


class Sparkline(tk.Canvas):
    """Başlık, son değerler ve zaman ekseni olan küçük çizgi grafik."""

    def __init__(self, parent, title, unit, height=92, y_max=None, fmt="{:.1f}"):
        super().__init__(parent, height=height, bg=CARD, highlightthickness=0)
        self.title, self.unit, self.y_max, self.fmt = title, unit, y_max, fmt
        self._t, self._series = [], []
        self.bind("<Configure>", lambda e: self._draw())

    def plot(self, t, series):
        """series: [(değerler, renk), ...]; değerler t ile aynı uzunlukta."""
        self._t, self._series = t, series
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 40:
            return
        self.create_text(10, 12, text=self.title, anchor="w", fill=MUTED, font=F["small_b"])
        x_text = w - 10
        for ys, color in reversed(self._series):
            if len(ys):
                item = self.create_text(x_text, 12, text=f"{self.fmt.format(ys[-1])} {self.unit}",
                                        anchor="e", fill=color, font=F["mono_s"])
                x_text = self.bbox(item)[0] - 10
        x0, x1, y0, y1 = 36, w - 10, 26, h - 16
        if len(self._t) < 2:
            self.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="waiting for data", fill=FAINT, font=F["small"])
            return
        peak = max((float(np.max(ys)) for ys, _ in self._series if len(ys)), default=1.0)
        top = self.y_max if self.y_max else max(peak * 1.15, 1e-3)
        t_end = max(self._t[-1], 1e-6)
        for frac in (0.0, 0.5, 1.0):
            y = y1 - frac * (y1 - y0)
            self.create_line(x0, y, x1, y, fill=BORDER)
            self.create_text(x0 - 5, y, text=f"{top * frac:.2g}", anchor="e", fill=FAINT, font=F["mono_s"])
        self.create_text(x0, h - 7, text="0", anchor="w", fill=FAINT, font=F["mono_s"])
        self.create_text(x1, h - 7, text=_fmt_clock(t_end), anchor="e", fill=FAINT, font=F["mono_s"])
        stride = max(1, len(self._t) // 300)
        ts = np.asarray(self._t[::stride])
        for ys, color in self._series:
            vals = np.asarray(ys[::stride])
            if len(vals) < 2:
                continue
            xs = x0 + ts[:len(vals)] / t_end * (x1 - x0)
            yy = y1 - np.clip(vals / top, 0, 1) * (y1 - y0)
            self.create_line(*np.column_stack([xs, yy]).ravel().tolist(), fill=color, width=1.5)


class VehicleCard(tk.Frame):
    """Sol paneldeki araç kartı: bağlantı, görev, kapsama, seyrüsefer ve hareket özeti."""

    def __init__(self, parent, name, on_click):
        super().__init__(parent, bg=CARD)
        self.name, self.color = name, AGENT_COLORS[name]
        self._strip = tk.Frame(self, bg=self.color, width=5)
        self._strip.pack(side="left", fill="y")
        body = tk.Frame(self, bg=CARD, padx=10, pady=8)
        body.pack(side="left", fill="both", expand=True)

        top = tk.Frame(body, bg=CARD)
        top.pack(fill="x")
        tk.Label(top, text=name.upper(), bg=CARD, fg=self.color, font=F["head"]).pack(side="left")
        self._mission = tk.Label(top, text="", bg=CARD, fg=MUTED, font=F["small_b"])
        self._mission.pack(side="left", padx=(8, 0))
        self._link = tk.Label(top, text="", bg=CARD, fg=MUTED, font=F["small_b"])
        self._link.pack(side="right")

        rows = tk.Frame(body, bg=CARD)
        rows.pack(fill="x", pady=(6, 0))
        rows.columnconfigure(1, weight=1)

        def key(row, text):
            tk.Label(rows, text=text, bg=CARD, fg=MUTED, font=F["small"], anchor="w").grid(
                row=row, column=0, sticky="w", padx=(0, 10), pady=1)

        key(0, "Map coverage")
        cov = tk.Frame(rows, bg=CARD)
        cov.grid(row=0, column=1, sticky="ew")
        self._cov_bar = Bar(cov, width=84)
        self._cov_bar.pack(side="left", pady=5)
        self._cov = tk.Label(cov, text="", bg=CARD, fg=TEXT, font=F["mono"])
        self._cov.pack(side="right")

        key(1, "Nav error")
        nav = tk.Frame(rows, bg=CARD)
        nav.grid(row=1, column=1, sticky="ew")
        self._err = tk.Label(nav, text="—", bg=CARD, fg=TEXT, font=F["mono_b"])
        self._err.pack(side="left")
        self._std = tk.Label(nav, text="", bg=CARD, fg=MUTED, font=F["mono_s"])
        self._std.pack(side="right")

        key(2, "DVL")
        self._dvl = tk.Label(rows, text="—", bg=CARD, fg=MUTED, font=F["small_b"], anchor="w")
        self._dvl.grid(row=2, column=1, sticky="w")

        key(3, "Depth · speed · hdg")
        self._motion = tk.Label(rows, text="—", bg=CARD, fg=TEXT, font=F["mono_s"], anchor="w")
        self._motion.grid(row=3, column=1, sticky="w")

        _bind_tree(self, "<Button-1>", lambda e: on_click(name))

    def set_selected(self, selected):
        _set_bg_tree(self, CARD_SEL if selected else CARD, skip=(self._strip,))

    def refresh(self, sd, link_state, coverage_pct):
        self._link.config(text="● " + link_state, fg=STATE_COLORS.get(link_state, MUTED))
        mission = sd.get("mission", "MAPPING")
        status = sd.get("status", "OK")
        nominal = status in ("OK", "", None)
        self._mission.config(text=mission if nominal else f"{mission} · {status}",
                             fg=MISSION_COLORS.get(mission, MUTED) if nominal else WARN)
        self._cov_bar.set(coverage_pct / 100.0, self.color)
        self._cov.config(text=f"{coverage_pct:5.1f} %")

        if "nav_error" in sd:
            err = sd["nav_error"]
            self._err.config(text=f"{err:.2f} m", fg=TEXT if err < 1.0 else WARN if err < 5.0 else BAD)
            self._std.config(text=f"1σ {sd.get('ekf_std', 0.0):.2f} m")
            lost = sd.get("dvl_lost", 0.0) > 0.5
            self._dvl.config(text="NO BOTTOM LOCK" if lost else "BOTTOM LOCK", fg=BAD if lost else GOOD)
        if "x" in sd:
            self._motion.config(text=f"{abs(sd['z']):.0f} m · {sd.get('vel', 0.0):.2f} m/s · "
                                     f"{_compass_deg(sd.get('heading', 0.0)):03.0f}°")


class MapView(tk.Canvas):
    """Çalışma alanı haritası: kapsama, izler, hedefler, akustik linkler ve araçlar."""

    def __init__(self, parent, fusion, layers, coverage_source, follow, on_select):
        super().__init__(parent, bg=BG, highlightthickness=0, cursor="crosshair")
        self.fusion, self.layers = fusion, layers
        self.coverage_source, self.follow, self.on_select = coverage_source, follow, on_select
        self.half = MAP_SIZE_M / 2.0
        self.ppm = None                 # piksel / metre (ilk boyutlandırmada sığdırılır)
        self._auto_fit = True           # kullanıcı yakınlaştırıp kaydırana kadar pencereye sığdır
        self.center = [0.0, 0.0]          # ekran merkezinin (doğu, kuzey) konumu
        self.selected = None
        self.sim_time = 0.0

        self.pos, self.yaw = {}, {}       # gerçek konum (NWU) ve yaw (rad)
        self.ekf, self.ekf_std = {}, {}
        self.home, self.targets = {}, {}
        self.trail = {n: deque(maxlen=TRAIL_LEN) for n in AGENT_NAMES}
        self.ekf_trail = {n: deque(maxlen=TRAIL_LEN) for n in AGENT_NAMES}
        self.link_state = {}
        self.sd = {}                      # araç başına ana döngüden gelen özet veriler
        self.link_flash = {}              # (gönderen, alıcı) → sim zamanı
        self.snr = {}                     # (a, b) → dB
        self.cursor_world = None
        self._drag = None
        self.tiles = SatelliteTiles()
        self._sat_key, self._sat_img = None, None
        self._base_photo = None
        self._retry = None

        self.bind("<Configure>", self._on_resize)
        self.bind("<MouseWheel>", lambda e: self._zoom(e.x, e.y, 1.2 if e.delta > 0 else 1 / 1.2))
        self.bind("<Button-4>", lambda e: self._zoom(e.x, e.y, 1.2))
        self.bind("<Button-5>", lambda e: self._zoom(e.x, e.y, 1 / 1.2))
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Double-Button-1>", lambda e: self.fit())
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)

    # ── Koordinat dönüşümleri ────────────────────────────────────────
    def to_canvas(self, x_north, y_west):
        w, h = self.winfo_width(), self.winfo_height()
        return (w / 2 + (-y_west - self.center[0]) * self.ppm,
                h / 2 - (x_north - self.center[1]) * self.ppm)

    def to_world(self, cx, cy):
        """Ekran pikseli → (doğu, kuzey) metre."""
        w, h = self.winfo_width(), self.winfo_height()
        return (self.center[0] + (cx - w / 2) / self.ppm,
                self.center[1] - (cy - h / 2) / self.ppm)

    def fit(self):
        w, h = self.winfo_width(), self.winfo_height()
        self.ppm = max(0.2, min(w - 130, h - 130) / MAP_SIZE_M)
        self.center = [0.0, 0.0]
        self.follow.set(False)
        self._auto_fit = True
        self.redraw()

    # ── Etkileşim ────────────────────────────────────────────────────
    def _on_resize(self, event):
        if self.ppm is None or self._auto_fit:
            self.fit()
        else:
            self.redraw()

    def _zoom(self, cx, cy, factor):
        east, north = self.to_world(cx, cy)
        self._auto_fit = False
        self.ppm = min(max(self.ppm * factor, 0.3), 40.0)
        w, h = self.winfo_width(), self.winfo_height()
        self.center = [east - (cx - w / 2) / self.ppm, north + (cy - h / 2) / self.ppm]
        self.redraw()

    def _on_press(self, event):
        self._drag = {"x": event.x, "y": event.y, "center": list(self.center), "moved": False}

    def _on_drag(self, event):
        d = self._drag
        if d is None:
            return
        dx, dy = event.x - d["x"], event.y - d["y"]
        if not d["moved"] and math.hypot(dx, dy) < 4:
            return
        d["moved"] = True
        self._auto_fit = False
        self.follow.set(False)
        self.center = [d["center"][0] - dx / self.ppm, d["center"][1] + dy / self.ppm]
        self.redraw()

    def _on_release(self, event):
        d, self._drag = self._drag, None
        if d is None or d["moved"]:
            return
        hit, best = None, 18.0
        for name, p in self.pos.items():
            cx, cy = self.to_canvas(p[0], p[1])
            dist = math.hypot(cx - event.x, cy - event.y)
            if dist < best:
                hit, best = name, dist
        self.on_select(hit)

    def _on_motion(self, event):
        if self.ppm:
            self.cursor_world = self.to_world(event.x, event.y)
            self._draw_cursor_readout()

    def _on_leave(self, event):
        self.cursor_world = None
        self._draw_cursor_readout()

    # ── Çizim ────────────────────────────────────────────────────────
    def redraw(self):
        if self.ppm is None:
            return
        if self.follow.get() and self.selected in self.pos:
            p = self.pos[self.selected]
            self.center = [-p[1], p[0]]
        self.delete("all")
        self._draw_base()
        if self.layers["grid"].get():
            self._draw_grid()
        self.create_rectangle(*self._area_box(), outline=_blend(ACCENT, BG, 0.45), width=1.5)
        if self.layers["links"].get():
            self._draw_links()
        if self.layers["trails"].get():
            self._draw_trails()
        if self.layers["swath"].get():
            self._draw_swaths()
        if self.layers["targets"].get():
            self._draw_targets()
        self._draw_vehicles()
        self._draw_hud()
        self._draw_cursor_readout()

    def _area_box(self):
        x0, y0 = self.to_canvas(self.half, self.half)       # kuzeybatı köşe
        x1, y1 = self.to_canvas(-self.half, -self.half)     # güneydoğu köşe
        return x0, y0, x1, y1

    def _coverage_grid(self):
        maps = self.fusion.maps
        if self.coverage_source.get() == "selected" and self.selected in maps:
            return maps[self.selected].grid
        return np.maximum.reduce([maps[n].grid for n in AGENT_NAMES if n in maps])

    def _draw_base(self):
        """Zemin (uydu görüntüsü veya düz renk) ile kapsama katmanını tek görüntüde birleştirir."""
        w, h = self.winfo_width(), self.winfo_height()
        if self.layers["satellite"].get():
            key = (w, h, round(self.ppm, 5), round(self.center[0], 2), round(self.center[1], 2), self.tiles.version)
            if key != self._sat_key:
                self._sat_img, complete = self._compose_satellite(w, h)
                self._sat_key = key if complete else None
                if not complete and self._retry is None:
                    self._retry = self.after(600, self._retry_redraw)
            base = self._sat_img.copy()
        else:
            base = Image.new("RGB", (w, h), _rgb(BG))
            ImageDraw.Draw(base).rectangle([int(v) for v in self._area_box()], fill=_rgb(AREA))
        if self.layers["coverage"].get():
            overlay = self._coverage_overlay(w, h)
            if overlay is not None:
                base.paste(overlay[0], overlay[1], overlay[0])
        self._base_photo = ImageTk.PhotoImage(base)
        self.create_image(0, 0, image=self._base_photo, anchor="nw")

    def _retry_redraw(self):
        self._retry = None
        self.redraw()

    def _compose_satellite(self, w, h):
        """Görünen bölgenin uydu karolarını ekran boyutunda tek görüntüye yapıştırır."""
        img = Image.new("RGB", (w, h), _rgb(AREA))
        metres_per_tile_px = 256.0 / self.ppm
        circumference = 2 * math.pi * EARTH_R * math.cos(math.radians(MAP_DATUM_LAT))
        z = int(round(math.log2(circumference / metres_per_tile_px)))
        z = min(max(z, TILE_ZOOM_RANGE[0]), TILE_ZOOM_RANGE[1])
        east_min, north_max = self.to_world(0, 0)
        east_max, north_min = self.to_world(w, h)
        tx0, ty0 = _deg2tile(*_local_to_latlon(east_min, north_max), z)
        tx1, ty1 = _deg2tile(*_local_to_latlon(east_max, north_min), z)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 80:
            return img, True
        complete = True
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile = self.tiles.get(z, tx, ty)
                if tile is None:
                    complete = False
                    continue
                east_a, north_a = _latlon_to_local(*_tile2deg(tx, ty, z))
                east_b, north_b = _latlon_to_local(*_tile2deg(tx + 1, ty + 1, z))
                x0 = int(math.floor(w / 2 + (east_a - self.center[0]) * self.ppm))
                y0 = int(math.floor(h / 2 - (north_a - self.center[1]) * self.ppm))
                x1 = int(math.ceil(w / 2 + (east_b - self.center[0]) * self.ppm))
                y1 = int(math.ceil(h / 2 - (north_b - self.center[1]) * self.ppm))
                img.paste(tile.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.BILINEAR), (x0, y0))
        return ImageEnhance.Brightness(img).enhance(SATELLITE_BRIGHTNESS), complete

    def _coverage_overlay(self, w, h):
        """Kapsama ızgarasının görünen kısmı: (yarı saydam görüntü, (sol, üst)) veya None."""
        grid = self._coverage_grid()
        n = grid.shape[0]
        res = MAP_SIZE_M / n
        east_min, north_max = self.to_world(0, 0)
        east_max, north_min = self.to_world(w, h)
        # grid[i, j]: i kuzey ekseni, j batı ekseni (batı = -doğu)
        i0 = max(0, int(math.floor((north_min + self.half) / res)))
        i1 = min(n, int(math.ceil((north_max + self.half) / res)))
        j0 = max(0, int(math.floor((-east_max + self.half) / res)))
        j1 = min(n, int(math.ceil((-east_min + self.half) / res)))
        if i1 <= i0 or j1 <= j0:
            return None
        sub = grid[i0:i1, j0:j1][::-1, ::-1]     # satır 0 = kuzey, sütun 0 = batı
        rgba = np.empty(sub.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = COVERED
        rgba[..., 3] = np.where(sub > 0.7, 0.5, np.clip((sub - 0.5) / 0.2, 0.0, 1.0) * 0.22) * 255
        left, top = self.to_canvas(i1 * res - self.half, j1 * res - self.half)
        size = (max(1, int(round((j1 - j0) * res * self.ppm))), max(1, int(round((i1 - i0) * res * self.ppm))))
        return Image.fromarray(rgba).resize(size, Image.NEAREST), (int(round(left)), int(round(top)))

    def _draw_grid(self):
        step = next((s for s in (10, 25, 50, 100, 200) if s * self.ppm >= 55), 200)
        x0, y0, x1, y1 = self._area_box()
        minor, major = _blend(AREA, TEXT, 0.18), _blend(AREA, TEXT, 0.36)
        k = -int(self.half // step)
        while k * step <= self.half:
            v = k * step
            cx, _ = self.to_canvas(0, -v)        # doğu = v
            _, cy = self.to_canvas(v, 0)         # kuzey = v
            self.create_line(cx, y0, cx, y1, fill=major if v == 0 else minor)
            self.create_line(x0, cy, x1, cy, fill=major if v == 0 else minor)
            label = f"{v:+.0f}" if v else "0"
            self.create_text(cx, y1 + 11, text=label, fill=MUTED, font=F["mono_s"])
            self.create_text(x0 - 6, cy, text=label, fill=MUTED, font=F["mono_s"], anchor="e")
            k += 1
        self.create_text((x0 + x1) / 2, y1 + 28, text="East (m)", fill=MUTED, font=F["small"])
        self.create_text(x0 - 46, (y0 + y1) / 2, text="North (m)", fill=MUTED, font=F["small"], angle=90)

    def _draw_links(self):
        names = [n for n in AGENT_NAMES if n in self.pos]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ax, ay = self.to_canvas(*self.pos[a][:2])
                bx, by = self.to_canvas(*self.pos[b][:2])
                offline = "OFFLINE" in (self.link_state.get(a), self.link_state.get(b))
                self.create_line(ax, ay, bx, by, fill=_blend(BAD, BG, 0.5) if offline else FAINT, dash=(2, 6))
                snr = self.snr.get((a, b))
                if self.layers["labels"].get() and snr is not None and not math.isnan(snr):
                    self.create_text((ax + bx) / 2, (ay + by) / 2 - 9, text=f"{snr:.0f} dB",
                                     fill=MUTED, font=F["mono_s"])
        # Yeni gönderilen paketler: gönderenden alıcıya renkli ok
        for (s, r), t in list(self.link_flash.items()):
            age = self.sim_time - t
            if not 0 <= age <= LINK_FLASH_S:
                del self.link_flash[(s, r)]
            elif s in self.pos and r in self.pos:
                sx, sy = self.to_canvas(*self.pos[s][:2])
                rx, ry = self.to_canvas(*self.pos[r][:2])
                self.create_line(sx, sy, rx, ry, width=2, arrow="last", arrowshape=(10, 12, 4),
                                 fill=_blend(AGENT_COLORS[s], BG, 0.2 + 0.5 * age / LINK_FLASH_S))

    def _draw_trails(self):
        show_ekf = self.layers["ekf"].get()
        for name in AGENT_NAMES:
            color = AGENT_COLORS[name]
            pts = self.trail[name]
            if len(pts) >= 2:
                coords = [c for p in pts for c in self.to_canvas(p[0], p[1])]
                self.create_line(*coords, fill=_blend(color, AREA, 0.3), width=1.5)
            est = self.ekf_trail[name]
            if show_ekf and len(est) >= 2:
                coords = [c for p in est for c in self.to_canvas(p[0], p[1])]
                self.create_line(*coords, fill=_blend(color, "#ffffff", 0.55), width=1, dash=(3, 3))

    def _draw_swaths(self):
        for name, p in self.pos.items():
            yaw = self.yaw[name]
            side = np.array([-math.sin(yaw), math.cos(yaw)]) * SENSOR_RANGE
            ax, ay = self.to_canvas(p[0] + side[0], p[1] + side[1])
            bx, by = self.to_canvas(p[0] - side[0], p[1] - side[1])
            self.create_line(ax, ay, bx, by, fill=_blend(AGENT_COLORS[name], AREA, 0.35), width=2, capstyle="round")

    def _draw_targets(self):
        for name, p in self.home.items():
            cx, cy = self.to_canvas(p[0], p[1])
            self.create_rectangle(cx - 4, cy - 4, cx + 4, cy + 4, outline=_blend(AGENT_COLORS[name], BG, 0.3))
        for name, tgt in self.targets.items():
            if tgt is None or name not in self.pos:
                continue
            color = AGENT_COLORS[name]
            px, py = self.to_canvas(*self.pos[name][:2])
            tx, ty = self.to_canvas(tgt[0], tgt[1])
            self.create_line(px, py, tx, ty, fill=_blend(color, BG, 0.45), dash=(5, 4))
            self.create_oval(tx - 6, ty - 6, tx + 6, ty + 6, outline=color, width=1.5)
            self.create_line(tx - 9, ty, tx + 9, ty, fill=color)
            self.create_line(tx, ty - 9, tx, ty + 9, fill=color)

    def _draw_vehicles(self):
        body = ((14, 0), (4, 4), (-7, 3), (-4, 0), (-7, -3), (4, -4))
        wings = (((-3, 0), (-7, 7), (-7, 3)), ((-3, 0), (-7, -7), (-7, -3)))
        for name in AGENT_NAMES:
            if name not in self.pos:
                continue
            p, yaw = self.pos[name], self.yaw[name]
            state = self.link_state.get(name, "CONNECTED")
            color = _blend(AGENT_COLORS[name], BG, 0.55) if state == "OFFLINE" else AGENT_COLORS[name]
            cx, cy = self.to_canvas(p[0], p[1])
            s, c = math.sin(yaw), math.cos(yaw)

            def px(fwd, left):
                """Gövde eksenindeki piksel ofseti → ekran koordinatı (NWU: x kuzey, y batı)."""
                return cx - (fwd * s + left * c), cy - (fwd * c - left * s)

            if self.layers["ekf"].get() and name in self.ekf:
                pale = _blend(color, "#ffffff", 0.55)
                ex, ey = self.to_canvas(*self.ekf[name])
                r = 3.0 * self.ekf_std.get(name, 0.0) * self.ppm       # 3σ konum belirsizliği
                if r > 4:
                    self.create_oval(ex - r, ey - r, ex + r, ey + r, outline=pale, dash=(2, 3))
                self.create_polygon(ex, ey - 5, ex + 5, ey, ex, ey + 5, ex - 5, ey, fill="", outline=pale)
            if name == self.selected:
                self.create_oval(cx - 22, cy - 22, cx + 22, cy + 22, outline="#ffffff", width=2, dash=(4, 2))
            self.create_line(cx, cy, *px(30, 0), fill=color, width=1.5, arrow="last", arrowshape=(8, 10, 3))
            self.create_oval(cx - 14, cy - 14, cx + 14, cy + 14, outline=STATE_COLORS.get(state, color), width=1.5)
            for wing in wings:
                self.create_polygon(*[v for pt in wing for v in px(*pt)], fill=_blend(color, BG, 0.45), outline="")
            self.create_polygon(*[v for pt in body for v in px(*pt)], fill=color, outline="#000a14", width=1)
            self.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill="#ffffff", outline="")
            if self.layers["labels"].get():
                sd = self.sd.get(name, {})
                label = name.upper()
                if "z" in sd:
                    label += f"\n{abs(sd['z']):.0f} m | {sd.get('vel', 0.0):.1f} m/s"
                self.create_text(cx + 18, cy - 16, text=label, fill=color, font=F["small_b"], anchor="sw")

    def _draw_hud(self):
        w, h = self.winfo_width(), self.winfo_height()
        # Kuzey oku
        nx, ny = w - 34, 44
        self.create_polygon(nx, ny - 20, nx + 8, ny + 6, nx, ny, nx - 8, ny + 6, fill=TEXT, outline="")
        self.create_text(nx, ny + 17, text="N", fill=TEXT, font=F["small_b"])
        # Ölçek çubuğu
        length = next((m for m in (200, 100, 50, 20, 10, 5, 2, 1) if m * self.ppm <= 150), 1)
        x0, y0 = 20, h - 24
        x1 = x0 + length * self.ppm
        self.create_line(x0, y0, x1, y0, fill=TEXT, width=2)
        self.create_line(x0, y0 - 5, x0, y0 + 1, fill=TEXT, width=2)
        self.create_line(x1, y0 - 5, x1, y0 + 1, fill=TEXT, width=2)
        self.create_text((x0 + x1) / 2, y0 - 11, text=f"{length} m", fill=TEXT, font=F["mono_s"])
        # Lejant (üstte tek satır)
        x, y = 20, 20
        marker = self.create_text(0, 0, text="")

        def entry(text, draw):
            nonlocal x
            draw(x)
            item = self.create_text(x + 24, y, text=text, fill=MUTED, font=F["small"], anchor="w")
            x = self.bbox(item)[2] + 22

        if self.layers["coverage"].get():
            own = self.coverage_source.get() == "selected" and self.selected in self.fusion.maps
            entry(f"Scanned ({self.selected.upper()}'s own map)" if own else "Scanned (fleet union)",
                  lambda x0: self.create_rectangle(x0 + 2, y - 6, x0 + 14, y + 6, outline="",
                                                   fill=_blend(AREA, COVERED_HEX, 0.55)))
        if self.layers["trails"].get():
            entry("True track", lambda x0: self.create_line(x0, y, x0 + 16, y, fill=MUTED, width=1.5))
        if self.layers["ekf"].get():
            entry("EKF estimate, 3σ circle", lambda x0: self.create_line(x0, y, x0 + 16, y, fill=TEXT, dash=(3, 3)))
        if self.layers["targets"].get():
            entry("Target", lambda x0: self.create_oval(x0 + 3, y - 5, x0 + 13, y + 5, outline=MUTED, width=1.5))
            entry("Home", lambda x0: self.create_rectangle(x0 + 4, y - 4, x0 + 12, y + 4, outline=MUTED))
        if x > 20:
            self.tag_lower(self.create_rectangle(8, 6, x - 8, 34, fill=PANEL, outline=BORDER), marker)
        if self.layers["satellite"].get():
            self.create_text(w - 16, h - 36, text="Imagery © Esri", fill=FAINT, font=F["small"], anchor="e")

    def _draw_cursor_readout(self):
        self.delete("cursor")
        if self.cursor_world is None:
            return
        east, north = self.cursor_world
        self.create_text(self.winfo_width() - 16, self.winfo_height() - 18, tags="cursor", anchor="e",
                         text=f"E {east:+7.1f} m    N {north:+7.1f} m", fill=MUTED, font=F["mono_s"])


class CurrentCompass(tk.Canvas):
    """Gerçek akıntı ile araçların kestirdiği akıntıyı aynı pusulada gösterir."""

    def __init__(self, parent, size=132):
        super().__init__(parent, width=size, height=size, bg=CARD, highlightthickness=0)
        self._size = size

    def plot(self, true_ne, estimates):
        """true_ne: (kuzey, doğu) m/s veya None; estimates: {araç: (kuzey, doğu)}."""
        self.delete("all")
        c, r = self._size / 2, self._size / 2 - 16
        mags = [math.hypot(*v) for v in ([true_ne] if true_ne else []) + list(estimates.values())]
        full = max(0.25, math.ceil(max(mags, default=0.0) * 4 - 1e-9) / 4)   # tam ölçek: 0.25 m/s katları
        for frac in (0.5, 1.0):
            self.create_oval(c - r * frac, c - r * frac, c + r * frac, c + r * frac, outline=BORDER)
        self.create_line(c - r, c, c + r, c, fill=BORDER)
        self.create_line(c, c - r, c, c + r, fill=BORDER)
        for text, dx, dy in (("N", 0, -1), ("E", 1, 0), ("S", 0, 1), ("W", -1, 0)):
            self.create_text(c + dx * (r + 9), c + dy * (r + 9), text=text, fill=FAINT, font=F["small"])
        self.create_text(c + r * 0.72, c + r * 0.86, text=f"{full:.2f} m/s", fill=FAINT, font=F["mono_s"])

        def arrow(vec, color, width):
            north, east = vec
            if math.hypot(north, east) > 1e-3:
                self.create_line(c, c, c + east / full * r, c - north / full * r, fill=color, width=width,
                                 arrow="last", arrowshape=(8, 10, 3))

        if true_ne:
            arrow(true_ne, TEXT, 3)
        for name, vec in estimates.items():
            arrow(vec, AGENT_COLORS[name], 1.5)


class Dashboard:
    def __init__(self, consensus_fusion, comm_manager=None):
        self.fusion, self.comm_mgr = consensus_fusion, comm_manager

        # Ana döngünün yazdığı tamponlar
        self.sensor_data = {n: {} for n in AGENT_NAMES}
        self.comm_log = EventLog()
        self._active_links = {}          # eski ana döngülerle uyumluluk için tutulur

        self._paused = False
        self._closed = False
        self._wall_start = time.time()
        self._last_tick = None           # (duvar saati, sim zamanı)
        self._rt_factor = None
        self._hist_t, self._hist_cov = [], []
        self._hist_err = {n: [] for n in AGENT_NAMES}
        self._prev = {n: {} for n in AGENT_NAMES}      # olay kaydı için önceki durumlar
        self._milestones = list(COVERAGE_MILESTONES)
        self._sonar = {"fls": {}, "mbes": {}, "sss": {n: deque(maxlen=220) for n in AGENT_NAMES}}
        self._sonar_win = None

        self.root = tk.Tk()
        self.root.title("Cooperative Seafloor Mapping — Mission Control")
        self.root.configure(bg=BG)
        self.root.geometry("1560x920")
        self.root.minsize(1240, 760)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<space>", lambda e: self._toggle_pause())
        _init_fonts(self.root)

        self.layers = {k: tk.BooleanVar(value=v) for k, v in (
            ("satellite", True), ("coverage", True), ("trails", True), ("ekf", True), ("targets", True),
            ("links", True), ("swath", True), ("grid", True), ("labels", True))}
        self.coverage_source = tk.StringVar(value="union")
        self.follow = tk.BooleanVar(value=False)

        self._build_header()
        self._build_footer()
        self._build_left()
        self._build_right()
        self.map = MapView(self.root, self.fusion, self.layers, self.coverage_source, self.follow, self._select)
        self.map.pack(side="left", fill="both", expand=True)
        self.root.update()

    # ── Yerleşim ─────────────────────────────────────────────────────
    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=PANEL, fg=MUTED, font=F["small_b"], anchor="w").pack(
            fill="x", padx=14, pady=(12, 5))

    def _kpi(self, parent, title):
        box = tk.Frame(parent, bg=PANEL, padx=16)
        box.pack(side="left", fill="y")
        tk.Label(box, text=title, bg=PANEL, fg=MUTED, font=F["small_b"], anchor="w").pack(fill="x", pady=(9, 0))
        value = tk.Label(box, text="—", bg=PANEL, fg=TEXT, font=F["kpi"], anchor="w")
        value.pack(fill="x")
        sub = tk.Label(box, text="", bg=PANEL, fg=FAINT, font=F["small"], anchor="w")
        sub.pack(fill="x", pady=(0, 8))
        tk.Frame(parent, bg=BORDER, width=1).pack(side="left", fill="y", pady=12)
        return value, sub

    def _button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg=CARD, fg=TEXT, activebackground=CARD_SEL,
                         activeforeground=TEXT, relief="flat", bd=0, padx=12, pady=3, font=F["ui"], cursor="hand2")

    def _check(self, parent, text, variable):
        return tk.Checkbutton(parent, text=text, variable=variable, command=self._redraw, bg=PANEL, fg=TEXT,
                              selectcolor=CARD, activebackground=PANEL, activeforeground=TEXT, font=F["small"],
                              anchor="w", bd=0, highlightthickness=0)

    def _build_header(self):
        bar = tk.Frame(self.root, bg=PANEL)
        bar.pack(side="top", fill="x")
        tk.Frame(self.root, bg=BORDER, height=1).pack(side="top", fill="x")

        brand = tk.Frame(bar, bg=PANEL, padx=14, width=LEFT_W)
        brand.pack(side="left", fill="y")
        brand.pack_propagate(False)
        tk.Label(brand, text="COOPERATIVE SEAFLOOR MAPPING", bg=PANEL, fg=TEXT, font=F["head"], anchor="w").pack(
            fill="x", pady=(16, 0))
        tk.Label(brand, text=f"{len(AGENT_NAMES)} × REMUS-100  ·  HoloOcean", bg=PANEL, fg=MUTED,
                 font=F["small"], anchor="w").pack(fill="x")
        tk.Frame(bar, bg=BORDER, width=1).pack(side="left", fill="y", pady=12)

        self._k_time = self._kpi(bar, "MISSION TIME")
        self._k_cov = self._kpi(bar, "AREA SCANNED")
        self._k_rms = self._kpi(bar, "MAP DISAGREEMENT")
        self._k_pdr = self._kpi(bar, "PACKET DELIVERY")
        self._k_nav = self._kpi(bar, "NAV ERROR")
        self._k_cur = self._kpi(bar, "OCEAN CURRENT")

        right = tk.Frame(bar, bg=PANEL, padx=14)
        right.pack(side="right", fill="y")
        self._status = tk.Label(right, text="STARTING", bg=PANEL, fg=ACCENT, font=F["head"])
        self._status.pack(anchor="e", pady=(12, 4))
        buttons = tk.Frame(right, bg=PANEL)
        buttons.pack(anchor="e")
        self._pause_btn = self._button(buttons, "Pause", self._toggle_pause)
        self._pause_btn.pack(side="left", padx=(0, 6))
        self._button(buttons, "Sonar", self._open_sonar).pack(side="left", padx=(0, 6))
        self._button(buttons, "Fit map", lambda: self.map.fit()).pack(side="left")

    def _build_footer(self):
        bar = tk.Frame(self.root, bg=PANEL)
        bar.pack(side="bottom", fill="x")
        tk.Frame(self.root, bg=BORDER, height=1).pack(side="bottom", fill="x")
        tk.Label(bar, bg=PANEL, fg=FAINT, font=F["small"], anchor="w",
                 text="Scroll: zoom   ·   Drag: pan   ·   Click a vehicle: select   ·   "
                      "Double-click: fit map   ·   Space: pause").pack(side="left", padx=14, pady=4)
        self._foot = tk.Label(bar, text="", bg=PANEL, fg=FAINT, font=F["small"], anchor="e")
        self._foot.pack(side="right", padx=14)

    def _build_left(self):
        panel = tk.Frame(self.root, bg=PANEL, width=LEFT_W)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)
        tk.Frame(self.root, bg=BORDER, width=1).pack(side="left", fill="y")

        self._section(panel, "VEHICLES")
        self.cards = {}
        for name in AGENT_NAMES:
            card = VehicleCard(panel, name, self._toggle_select)
            card.pack(fill="x", padx=12, pady=(0, 8))
            self.cards[name] = card

        self._section(panel, "MAP LAYERS")
        grid = tk.Frame(panel, bg=PANEL)
        grid.pack(fill="x", padx=10)
        labels = (("satellite", "Satellite map"), ("coverage", "Scanned area"), ("trails", "True tracks"), ("ekf", "EKF estimate"),
                  ("targets", "Targets / home"), ("links", "Acoustic links"), ("swath", "Sonar swath"),
                  ("grid", "Grid"), ("labels", "Labels"))
        for i, (key, text) in enumerate(labels):
            self._check(grid, text, self.layers[key]).grid(row=i // 2, column=i % 2, sticky="w")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self._section(panel, "SCANNED AREA SHOWS")
        src = tk.Frame(panel, bg=PANEL)
        src.pack(fill="x", padx=10)
        for value, text in (("union", "Fleet union (what was really scanned)"),
                            ("selected", "Selected vehicle's own map (its belief)")):
            tk.Radiobutton(src, text=text, value=value, variable=self.coverage_source, command=self._redraw,
                           bg=PANEL, fg=TEXT, selectcolor=CARD, activebackground=PANEL, activeforeground=TEXT,
                           font=F["small"], anchor="w", bd=0, highlightthickness=0).pack(fill="x")
        self._check(src, "Follow selected vehicle", self.follow).pack(fill="x", pady=(6, 0))

    def _build_right(self):
        panel = tk.Frame(self.root, bg=PANEL, width=RIGHT_W)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        tk.Frame(self.root, bg=BORDER, width=1).pack(side="right", fill="y")

        # Seyrüsefer tablosu
        self._section(panel, "NAVIGATION  ·  EKF vs ground truth")
        table = tk.Frame(panel, bg=CARD, padx=10, pady=6)
        table.pack(fill="x", padx=12)
        for col, text in enumerate(("", "Error", "1σ", "NIS", "DVL", "Range fixes")):
            tk.Label(table, text=text, bg=CARD, fg=MUTED, font=F["small"]).grid(
                row=0, column=col, sticky="w" if col == 0 else "e", padx=0 if col == 0 else (10, 0))
            table.columnconfigure(col, weight=1)
        self._nav_cells = {}
        for row, name in enumerate(AGENT_NAMES, start=1):
            tk.Label(table, text=name.upper(), bg=CARD, fg=AGENT_COLORS[name], font=F["small_b"]).grid(
                row=row, column=0, sticky="w", pady=1)
            cells = []
            for col in range(1, 6):
                cell = tk.Label(table, text="—", bg=CARD, fg=TEXT, font=F["mono"])
                cell.grid(row=row, column=col, sticky="e", padx=(10, 0))
                cells.append(cell)
            self._nav_cells[name] = cells

        # Akıntı
        self._section(panel, "OCEAN CURRENT  ·  white: true, colour: estimated")
        cur = tk.Frame(panel, bg=CARD, padx=8, pady=6)
        cur.pack(fill="x", padx=12)
        self._compass = CurrentCompass(cur)
        self._compass.pack(side="left")
        text = tk.Frame(cur, bg=CARD)
        text.pack(side="left", fill="both", expand=True, padx=(10, 4))
        tk.Label(text, text="speed · direction (flows toward)", bg=CARD, fg=FAINT, font=F["small"],
                 anchor="e").pack(fill="x", pady=(4, 2))
        self._cur_rows = {}
        for key, color in [("true", TEXT)] + [(n, AGENT_COLORS[n]) for n in AGENT_NAMES]:
            row = tk.Frame(text, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text="TRUE" if key == "true" else key.upper(), bg=CARD, fg=color, font=F["small_b"],
                     anchor="w").pack(side="left")
            value = tk.Label(row, text="—", bg=CARD, fg=TEXT, font=F["mono"], anchor="e")
            value.pack(side="right")
            self._cur_rows[key] = value

        # Akustik linkler
        self._section(panel, "ACOUSTIC LINKS  ·  range, SNR")
        links = tk.Frame(panel, bg=CARD, padx=10, pady=6)
        links.pack(fill="x", padx=12)
        self._link_rows = {}
        pairs = [(a, b) for i, a in enumerate(AGENT_NAMES) for b in AGENT_NAMES[i + 1:]]
        for row, (a, b) in enumerate(pairs):
            tk.Label(links, text=f"{a.upper()} – {b.upper()}", bg=CARD, fg=TEXT, font=F["small_b"]).grid(
                row=row, column=0, sticky="w", pady=2)
            dist = tk.Label(links, text="—", bg=CARD, fg=MUTED, font=F["mono"])
            dist.grid(row=row, column=1, sticky="e", padx=(10, 0))
            snr = tk.Label(links, text="—", bg=CARD, fg=TEXT, font=F["mono"])
            snr.grid(row=row, column=2, sticky="e", padx=(10, 8))
            bar = Bar(links, width=70)
            bar.grid(row=row, column=3, sticky="e")
            self._link_rows[(a, b)] = (dist, snr, bar)
        links.columnconfigure(1, weight=1)

        # Eğilimler
        self._section(panel, "TRENDS")
        self._spark_cov = Sparkline(panel, "Area scanned", "%", y_max=100.0)
        self._spark_cov.pack(fill="x", padx=12)
        self._spark_err = Sparkline(panel, "Nav error", "m", fmt="{:.2f}")
        self._spark_err.pack(fill="x", padx=12, pady=(6, 0))

        # Olay kaydı
        self._section(panel, "EVENT LOG")
        self._log = tk.Text(panel, bg=CARD, fg=TEXT, font=F["mono_s"], relief="flat", bd=0, wrap="word",
                            padx=8, pady=6, state="disabled", height=4, cursor="arrow")
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._log.tag_config("time", foreground=FAINT)
        self._log.tag_config("good", foreground=GOOD)
        self._log.tag_config("warn", foreground=WARN)
        self._log.tag_config("bad", foreground=BAD)
        self._log.tag_config("info", foreground=TEXT)

    # ── Kullanıcı eylemleri ──────────────────────────────────────────
    def _select(self, name):
        self.map.selected = name
        for n, card in self.cards.items():
            card.set_selected(n == name)
        if name is None:
            self.follow.set(False)
        self._redraw()
        if self._sonar_win is not None:
            self._render_sonar()

    def _toggle_select(self, name):
        self._select(None if self.map.selected == name else name)

    def _redraw(self):
        if not self._closed:
            self.map.redraw()

    def _toggle_pause(self):
        self._paused = not self._paused
        self._pause_btn.config(text="Resume" if self._paused else "Pause")
        if self._paused:
            self._status.config(text="PAUSED", fg=WARN)

    def _on_close(self):
        """Pencere kapatılınca simülasyon arayüzsüz devam eder."""
        self._closed = True
        self._paused = False
        self.root.destroy()

    # ── Ana döngü arayüzü ────────────────────────────────────────────
    def trigger_beacon_pulse(self, sender, receiver, snr=15.0):
        if not self._closed:
            self.map.link_flash[(sender, receiver)] = self.comm_log.sim_time

    def console_log(self, message):
        self.comm_log.append(message)

    def update(self, states, targets, comm_quality, mode_name="proposed", sim_time=None):
        if self._closed:
            return
        now = time.time()
        if sim_time is None:
            sim_time = now - self._wall_start
        if self._last_tick is not None and now > self._last_tick[0]:
            rate = (sim_time - self._last_tick[1]) / (now - self._last_tick[0])
            self._rt_factor = rate if self._rt_factor is None else 0.8 * self._rt_factor + 0.2 * rate
        self.comm_log.sim_time = sim_time

        link_states = self.fusion.get_comm_states()
        self._update_map_data(states, targets, link_states, sim_time)
        self._store_sonar(states)

        coverage_union = self.fusion.get_union_coverage_pct()
        errors = {n: sd["nav_error"] for n, sd in self.sensor_data.items() if "nav_error" in sd}
        self._hist_t.append(sim_time)
        self._hist_cov.append(coverage_union)
        for name in AGENT_NAMES:
            self._hist_err[name].append(errors.get(name, 0.0))

        self._log_state_changes(link_states, coverage_union)
        self._refresh_header(sim_time, coverage_union, comm_quality, errors, mode_name)
        for name, card in self.cards.items():
            card.refresh(self.sensor_data.get(name, {}), link_states.get(name, "CONNECTED"),
                         self.fusion.get_coverage_pct(name))
        self._refresh_navigation()
        self._refresh_current()
        self._refresh_links()
        self._spark_cov.plot(self._hist_t, [(self._hist_cov, COVERED_HEX)])
        self._spark_err.plot(self._hist_t, [(self._hist_err[n], AGENT_COLORS[n]) for n in AGENT_NAMES if n in errors])
        self._refresh_log()
        self.map.redraw()
        if self._sonar_win is not None:
            self._render_sonar()

        self._pump()
        while self._paused and not self._closed:      # duraklatma simülasyonu da bekletir
            time.sleep(0.03)
            self._pump()
        self._last_tick = (time.time(), sim_time)

    def _pump(self):
        try:
            self.root.update()
        except tk.TclError:
            self._closed = True

    def _update_map_data(self, states, targets, link_states, sim_time):
        m = self.map
        m.sim_time, m.link_state, m.sd = sim_time, link_states, self.sensor_data
        for name in AGENT_NAMES:
            pose = states.get(name, {}).get("PoseSensor")
            if pose is not None:
                pos = np.array(pose[:3, 3], dtype=float)
                m.pos[name] = pos
                m.yaw[name] = math.atan2(pose[1, 0], pose[0, 0])
                m.trail[name].append((pos[0], pos[1]))
                m.home.setdefault(name, (pos[0], pos[1]))
            sd = self.sensor_data.get(name, {})
            if "ekf_x" in sd:
                m.ekf[name] = (sd["ekf_x"], sd["ekf_y"])
                m.ekf_std[name] = sd.get("ekf_std", 0.0)
                m.ekf_trail[name].append(m.ekf[name])
            m.targets[name] = targets.get(name) if targets else None
        if self.comm_mgr is not None:
            for a, b in self._link_rows:
                m.snr[(a, b)] = self.comm_mgr.get_snr(a, b)

    def _log_state_changes(self, link_states, coverage_union):
        for name in AGENT_NAMES:
            sd, prev = self.sensor_data.get(name, {}), self._prev[name]
            current = {"link": link_states.get(name), "mission": sd.get("mission"),
                       "dvl": (sd.get("dvl_lost", 0.0) > 0.5) if "dvl_lost" in sd else None}
            if prev:
                tag = name.upper()
                if current["link"] != prev["link"]:
                    self.comm_log.append(f"[LINK] {tag}: {prev['link']} → {current['link']}")
                if current["mission"] != prev["mission"] and current["mission"]:
                    self.comm_log.append(f"[MISSION] {tag}: {current['mission']}")
                if None not in (current["dvl"], prev["dvl"]) and current["dvl"] != prev["dvl"]:
                    self.comm_log.append(f"[DVL] {tag}: bottom lock " + ("LOST" if current["dvl"] else "regained"))
            self._prev[name] = current
        while self._milestones and coverage_union >= self._milestones[0]:
            self.comm_log.append(f"[COVERAGE] {self._milestones.pop(0)} % of the area scanned")

    def _refresh_header(self, sim_time, coverage_union, comm_quality, errors, mode_name):
        value, sub = self._k_time
        value.config(text="T+" + _fmt_clock(sim_time))
        sub.config(text=f"{self._rt_factor:.2f}× real time" if self._rt_factor else "simulation time")

        value, sub = self._k_cov
        value.config(text=f"{coverage_union:.1f} %", fg=GOOD if coverage_union >= 99 else TEXT)
        sub.config(text=f"vehicles believe {self.fusion.get_avg_coverage_pct():.1f} %")

        rms = self.fusion.get_consensus_rms()
        value, sub = self._k_rms
        value.config(text=f"{rms:.3f}")
        sub.config(text="RMS between vehicle maps")

        summary = comm_quality.get_summary() if comm_quality is not None else {}
        pdr = summary.get("global_pdr", 0.0)
        value, sub = self._k_pdr
        value.config(text=f"{100 * pdr:.0f} %", fg=TEXT if pdr >= 0.7 else WARN if pdr >= 0.4 else BAD)
        sub.config(text=f"{summary.get('total_received', 0)} of {summary.get('total_sent', 0)} packets")

        value, sub = self._k_nav
        if errors:
            mean = float(np.mean(list(errors.values())))
            worst = max(errors, key=errors.get)
            value.config(text=f"{mean:.2f} m", fg=TEXT if mean < 1.0 else WARN if mean < 5.0 else BAD)
            sub.config(text=f"worst {errors[worst]:.2f} m ({worst.upper()})")
        else:
            sub.config(text="no estimator running")

        value, sub = self._k_cur
        estimates = [sd["curr_est_mag"] for sd in self.sensor_data.values() if "curr_est_mag" in sd]
        truth = next((sd["curr_true_mag"] for sd in self.sensor_data.values() if "curr_true_mag" in sd), None)
        if truth is not None:
            value.config(text=f"{truth:.2f} m/s")
            sub.config(text=f"estimated {np.mean(estimates):.2f} m/s" if estimates else "true value")
        else:
            sub.config(text="not available")

        if not self._paused:
            missions = [sd.get("mission", "MAPPING") for sd in self.sensor_data.values()]
            if all(m == "HOME" for m in missions):
                self._status.config(text="MISSION COMPLETE", fg=GOOD)
            elif any(m in ("RETURNING", "HOME") for m in missions):
                self._status.config(text="RETURNING HOME", fg=WARN)
            else:
                self._status.config(text="MAPPING", fg=ACCENT)
        self._foot.config(text=f"mode: {mode_name}   ·   wall time {_fmt_clock(time.time() - self._wall_start)}")

    def _refresh_navigation(self):
        for name, cells in self._nav_cells.items():
            sd = self.sensor_data.get(name, {})
            if "nav_error" not in sd:
                continue
            err, std, nis = sd["nav_error"], sd.get("ekf_std", 0.0), sd.get("nis_dvl")
            lost = sd.get("dvl_lost", 0.0) > 0.5
            cells[0].config(text=f"{err:.2f} m", fg=TEXT if err < 1.0 else WARN if err < 5.0 else BAD)
            cells[1].config(text=f"{std:.2f} m")
            cells[2].config(text="—" if nis is None or math.isnan(nis) else f"{nis:.1f}")
            cells[3].config(text="LOST" if lost else "LOCK", fg=BAD if lost else GOOD)
            cells[4].config(text=str(sd["range_updates"]) if "range_updates" in sd else "—")

    def _refresh_current(self):
        def describe(north, east):
            return f"{math.hypot(north, east):.2f} m/s · {math.degrees(math.atan2(east, north)) % 360:03.0f}°"

        truth, estimates = None, {}
        for name in AGENT_NAMES:
            sd = self.sensor_data.get(name, {})
            if "curr_true_n" in sd:
                truth = (sd["curr_true_n"], sd["curr_true_e"])
            if "curr_est_n" in sd:
                estimates[name] = (sd["curr_est_n"], sd["curr_est_e"])
                self._cur_rows[name].config(text=describe(*estimates[name]))
        if truth:
            self._cur_rows["true"].config(text=describe(*truth))
        self._compass.plot(truth, estimates)

    def _refresh_links(self):
        for (a, b), (dist_lbl, snr_lbl, bar) in self._link_rows.items():
            if a in self.map.pos and b in self.map.pos:
                dist_lbl.config(text=f"{np.linalg.norm(self.map.pos[a] - self.map.pos[b]):.0f} m")
            snr = self.map.snr.get((a, b))
            if snr is None or math.isnan(snr):
                continue
            color = GOOD if snr > 20 else WARN if snr > 10 else BAD
            snr_lbl.config(text=f"{snr:.0f} dB", fg=color)
            bar.set(snr / 40.0, color)

    def _refresh_log(self):
        new = self.comm_log.pop_new()
        if not new:
            return
        self._log.config(state="normal")
        for t, message in new:
            upper = message.upper()
            tag = ("bad" if "LOST" in upper or "OFFLINE" in upper else
                   "warn" if "DEGRADED" in upper or "RETURNING" in upper or "STUCK" in upper else
                   "good" if "[MISSION]" in upper or "[COVERAGE]" in upper or "REGAINED" in upper else "info")
            self._log.insert("end", _fmt_clock(t) + "  ", "time", message + "\n", tag)
        if int(self._log.index("end-1c").split(".")[0]) > 400:
            self._log.delete("1.0", "100.0")
        self._log.see("end")
        self._log.config(state="disabled")

    # ── Sonar penceresi ──────────────────────────────────────────────
    def _store_sonar(self, states):
        for name in AGENT_NAMES:
            st = states.get(name, {})
            fls = st.get("ImagingSonar")
            if fls is not None and getattr(fls, "ndim", 0) == 2:
                self._sonar["fls"][name] = np.array(fls, dtype=np.float32)
            sss = st.get("SSS", st.get("SidescanSonar"))
            if sss is not None and getattr(sss, "ndim", 0) == 1 and len(sss) > 0:
                self._sonar["sss"][name].append(np.array(sss, dtype=np.float32))
            mbes = st.get("MBES", st.get("ProfilingSonar"))
            if mbes is not None and getattr(mbes, "ndim", 0) == 2:
                self._sonar["mbes"][name] = np.array(mbes, dtype=np.float32)

    def _open_sonar(self):
        if self._sonar_win is not None:
            self._sonar_win.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("Sonar")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.protocol("WM_DELETE_WINDOW", self._close_sonar)
        self._sonar_win, self._sonar_views = win, {}
        self._sonar_title = tk.Label(win, text="", bg=BG, fg=TEXT, font=F["head"], anchor="w")
        self._sonar_title.grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 0))
        for col, (key, title) in enumerate((("fls", "Forward-looking (imaging) sonar"),
                                            ("sss", "Sidescan waterfall (newest on top)"),
                                            ("mbes", "Profiling sonar"))):
            tk.Label(win, text=title, bg=BG, fg=MUTED, font=F["small_b"]).grid(row=1, column=col, padx=14, pady=(8, 4))
            view = tk.Label(win, bg=CARD, bd=0)
            view.grid(row=2, column=col, padx=14, pady=(0, 14))
            self._sonar_views[key] = view
        self._render_sonar()

    def _close_sonar(self):
        self._sonar_win.destroy()
        self._sonar_win = None

    def _render_sonar(self):
        name = self.map.selected or AGENT_NAMES[0]
        self._sonar_title.config(text=f"Sonar — {name.upper()}   (select a vehicle on the map to switch)",
                                 fg=AGENT_COLORS[name])
        rows = self._sonar["sss"][name]
        data = {"fls": self._sonar["fls"].get(name),
                "sss": np.array(rows)[::-1] if rows else None,
                "mbes": self._sonar["mbes"].get(name)}
        for key, view in self._sonar_views.items():
            arr = data[key]
            if arr is None:
                arr = np.zeros((2, 2), dtype=np.float32)
            elif key != "sss":
                arr = arr[::-1]                      # menzil yukarı doğru artsın
            peak = max(float(np.percentile(arr, 99.5)), 1e-6)
            idx = (np.clip(arr / peak, 0, 1) * 255).astype(np.uint8)
            photo = ImageTk.PhotoImage(Image.fromarray(_SONAR_LUT[idx]).resize((340, 340), Image.BILINEAR))
            view.config(image=photo)
            view.image = photo


def _build_sonar_lut():
    stops = np.array([0.0, 0.35, 0.7, 1.0])
    colors = np.array([_rgb(CARD), (14, 116, 144), (250, 204, 21), (255, 255, 255)], dtype=float)
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, stops, colors[:, c]) for c in range(3)], axis=1).astype(np.uint8)


_SONAR_LUT = _build_sonar_lut()
