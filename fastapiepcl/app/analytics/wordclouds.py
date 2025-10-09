import pandas as pd
from typing import Dict, List, Optional
import random
import math
import colorsys
from io import BytesIO

# Optional dependency: python-wordcloud
try:
    from wordcloud import WordCloud  # type: ignore
    _WC_AVAILABLE = True
except Exception:
    WordCloud = None  # type: ignore
    _WC_AVAILABLE = False


def _standardize_department(series: pd.Series) -> pd.Series:
    """Return a cleaned department series suitable for counting.

    - Casts to string
    - Strips whitespace
    - Title-cases values for nicer display
    - Drops empty strings
    """
    s = series.astype(str).fillna("").str.strip()
    s = s[s.ne("")]
    # Title case for better presentation without changing acronyms too much
    try:
        s = s.str.title()
    except Exception:
        # If any unexpected dtype issues occur, fall back to original
        pass
    return s


def compute_department_counts(df: Optional[pd.DataFrame]) -> pd.Series:
    """Compute value counts for the `department` column if present.

    Returns an empty Series if the dataframe is None or the column is missing.
    """
    if df is None or 'department' not in df.columns:
        return pd.Series(dtype='int64')
    s = _standardize_department(df['department'])
    return s.value_counts()


def build_words_from_counts(counts: pd.Series, *, top_n: int = 50, color: Optional[str] = None,
                            extra_meta: Optional[Dict[str, str]] = None,
                            stopwords: Optional[set] = None) -> List[Dict[str, object]]:
    """Convert a Series of counts into a list of wordcloud dicts.

    Required keys for the component: 'text', 'value'. Optional: 'color' and any metadata.
    """
    words: List[Dict[str, object]] = []
    if counts is None or counts.empty:
        return words
    if stopwords is None:
        stopwords = set()
    for text, value in counts.items():
        if str(text).strip().lower() in stopwords:
            continue
        # trim extremely long labels for readability
        display_text = str(text)
        if len(display_text) > 36:
            display_text = display_text[:33] + "…"
        # append after filtering; we'll cut to top_n post-filter
        item: Dict[str, object] = {"text": display_text, "value": int(value)}
        if color:
            item["color"] = color
        if extra_meta:
            item.update(extra_meta)
        words.append(item)
    # sort and take top_n after filtering
    words.sort(key=lambda w: w.get('value', 0), reverse=True)
    words = words[:top_n]
    return words


def get_incident_hazard_department_words(incident_df: Optional[pd.DataFrame],
                                         hazard_df: Optional[pd.DataFrame],
                                         *,
                                         top_n: int = 50,
                                         min_count: int = 1,
                                         extra_stopwords: Optional[set] = None) -> Dict[str, List[Dict[str, object]]]:
    """Prepare word lists for incident and hazard department counts.

    Returns a dict with keys 'incident' and 'hazard' mapping to word lists compatible with
    streamlit-wordcloud's visualize function.
    """
    inc_counts = compute_department_counts(incident_df)
    haz_counts = compute_department_counts(hazard_df)
    if min_count > 1:
        inc_counts = inc_counts[inc_counts >= min_count]
        haz_counts = haz_counts[haz_counts >= min_count]

    common_stops = {
        "na", "n/a", "none", "other", "others", "misc", "miscellaneous",
        "not assigned", "not applicable", "unknown"
    }
    if extra_stopwords:
        common_stops.update({str(s).strip().lower() for s in extra_stopwords})

    inc_words = build_words_from_counts(
        inc_counts,
        top_n=top_n,
        color="#16A34A",  # green
        extra_meta={"type": "Incident"},
        stopwords=common_stops,
    )
    haz_words = build_words_from_counts(
        haz_counts,
        top_n=top_n,
        color="#F59E0B",  # amber
        extra_meta={"type": "Hazard"},
        stopwords=common_stops,
    )
    return {"incident": inc_words, "hazard": haz_words}


def create_word_cloud_html(words: List[Dict[str, object]],
                          width: int = 800,
                          height: int = 400,
                          title: str = "Word Cloud") -> str:
    """Create an HTML/CSS-based word cloud. Self-contained; no third-party deps.

    The layout uses a randomized spiral for reasonable visual distribution.
    """
    if not words:
        return "<div style='text-align:center;padding:16px'>No data to display</div>"

    max_value = max(word['value'] for word in words)
    min_value = min(word['value'] for word in words)

    max_font_size = 48
    min_font_size = 12

    def calc_font_size(value: int) -> float:
        if max_value == min_value:
            return (max_font_size + min_font_size) / 2
        normalized = (value - min_value) / (max_value - min_value)
        return min_font_size + (max_font_size - min_font_size) * normalized

    html_parts: List[str] = [f"""
    <div style="width:{width}px;height:{height}px;position:relative;border:1px solid #e5e7eb;border-radius:8px;background:#f8fafc;overflow:hidden">
      <div style="position:absolute;inset:0;padding:16px">
        <div style="position:absolute;left:16px;top:8px;font-weight:600;color:#111827">{title}</div>
    """]

    shuffled = words.copy()
    random.shuffle(shuffled)

    center_x = width / 2
    center_y = height / 2
    angle_step = 2 * math.pi / max(1, len(shuffled))
    radius_inc = 3

    for i, w in enumerate(shuffled):
        size = calc_font_size(int(w['value']))
        color = w.get('color', '#374151')
        angle = angle_step * i * 5
        radius = radius_inc * i
        x = center_x + radius * math.cos(angle) + random.randint(-24, 24)
        y = center_y + radius * math.sin(angle) + random.randint(-18, 18)
        x = max(10, min(x, width - 120))
        y = max(24, min(y, height - 48))
        rotation = random.randint(-30, 30)
        opacity = 0.7 + (size / max_font_size) * 0.3
        html_parts.append(f"""
        <div title="{w['text']}: {w['value']}" style="position:absolute;left:{x}px;top:{y}px;font-size:{size}px;color:{color};font-weight:{600 if size>30 else 400};transform:rotate({rotation}deg);opacity:{opacity};white-space:nowrap;cursor:default">{w['text']}</div>
        """)

    html_parts.append("""
      </div>
    </div>
    """)
    return ''.join(html_parts)


def create_modern_word_cloud_html(words: List[Dict[str, object]],
                                 width: int = 1100,
                                 height: int = 500,
                                 title: Optional[str] = None,
                                 dark_color: str = "#0b0b15",
                                 light_color: str = "#d1d5db",
                                 use_gradient: bool = True) -> str:
    """Create a modern, non-overlapping, horizontal word cloud like the example.

    Uses a centered flexbox layout so words wrap to new rows automatically
    with consistent spacing. Top-ranked words are big and bold; the rest are
    light gray. This avoids overlap entirely.
    """
    if not words:
        return "<div style='text-align:center;padding:16px'>No data to display</div>"

    # Sort words by value descending
    sorted_words = sorted(words, key=lambda w: w.get('value', 0), reverse=True)

    max_value = max(w['value'] for w in sorted_words)
    min_value = min(w['value'] for w in sorted_words)

    max_font = 96
    min_font = 16

    def size_for(v: int) -> float:
        if max_value == min_value:
            return (max_font + min_font) / 2
        n = (v - min_value) / (max_value - min_value)
        return min_font + n * (max_font - min_font)

    html_parts: List[str] = [
        f"<div style=\"width:{width}px;height:{height}px;overflow:hidden;background:#fff;border-radius:8px;display:flex;flex-wrap:wrap;justify-content:center;align-content:center;gap:14px 18px;row-gap:18px;padding:12px;font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,sans-serif\">"
    ]
    if title:
        html_parts.append(f"<div style=\"position:absolute;left:12px;top:10px;font-weight:600;color:#111827\">{title}</div>")

    top_dark = max(3, min(6, len(sorted_words)//8 or 3))
    for i, w in enumerate(sorted_words):
        value = int(w['value'])
        font_size = size_for(value)
        rank = i
        if use_gradient:
            # interpolate between dark and light based on normalized value
            n = 0 if max_value == min_value else (value - min_value) / (max_value - min_value)
            # Convert hex to rgb
            def hex_to_rgb(h: str):
                h = h.lstrip('#')
                return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            def rgb_to_hex(rgb):
                return '#%02x%02x%02x' % rgb
            dc = hex_to_rgb(dark_color)
            lc = hex_to_rgb(light_color)
            mix = tuple(int(lc[j] + (dc[j] - lc[j]) * (n**0.6)) for j in range(3))
            color = rgb_to_hex(mix)
        else:
            color = dark_color if rank < top_dark else light_color
        weight = 900 if rank < top_dark else 600
        opacity = 1.0 if rank < top_dark else 0.9
        html_parts.append(
            f"<span title=\"{w['text']}: {value}\" style=\"font-size:{font_size}px;color:{color};font-weight:{weight};opacity:{opacity};white-space:nowrap;line-height:1\">{w['text']}</span>"
        )

    html_parts.append("</div>")
    return ''.join(html_parts)

def create_plotly_word_cloud(words: List[Dict[str, object]], title: str = "Word Cloud"):
    """Create a simple Plotly-based text scatter word cloud.

    Returns a go.Figure or None when no data.
    """
    try:
        import plotly.graph_objects as go  # local import to avoid global dependency at import time
    except Exception:
        return None

    if not words:
        return None

    max_value = max(word['value'] for word in words)
    min_value = min(word['value'] for word in words)

    def scale(value: int) -> float:
        if max_value == min_value:
            return 28
        norm = (value - min_value) / (max_value - min_value)
        return 10 + 46 * norm

    xs: List[float] = []
    ys: List[float] = []
    texts: List[str] = []
    sizes: List[float] = []
    colors: List[str] = []

    for w in words:
        xs.append(random.uniform(0, 10))
        ys.append(random.uniform(0, 10))
        texts.append(str(w['text']))
        sizes.append(scale(int(w['value'])))
        colors.append(w.get('color', '#374151'))

    fig = go.Figure()
    for i in range(len(texts)):
        fig.add_trace(go.Scatter(x=[xs[i]], y=[ys[i]], text=[texts[i]], mode='text',
                                 textfont=dict(size=sizes[i], color=colors[i]),
                                 hovertext=f"{texts[i]}: {words[i]['value']}", hoverinfo='text', showlegend=False))
    fig.update_layout(title=title, xaxis=dict(visible=False), yaxis=dict(visible=False), height=420,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=50,b=0))
    return fig


class WordCloudGenerator:
    """Efficient SVG/Canvas-based word cloud generator with minimal deps."""

    def __init__(self, width: int = 800, height: int = 400):
        self.width = width
        self.height = height
        self._positions_cache: Dict[int, tuple] = {}

    def _calculate_font_size(
        self,
        value: int,
        min_val: int,
        max_val: int,
        min_size: int = 10,
        max_size: int = 60,
    ) -> int:
        if max_val == min_val:
            return (max_size + min_size) // 2
        # logarithmic scaling for better distribution
        log_value = math.log(value + 1)
        log_min = math.log(min_val + 1)
        log_max = math.log(max_val + 1)
        normalized = 0.5 if log_max == log_min else (log_value - log_min) / (log_max - log_min)
        return int(min_size + (max_size - min_size) * normalized)

    def _spiral_position(self, index: int) -> tuple:
        if index in self._positions_cache:
            return self._positions_cache[index]
        angle = index * 0.5  # Archimedean spiral
        radius = 5 + index * 2
        x = self.width / 2 + radius * math.cos(angle)
        y = self.height / 2 + radius * math.sin(angle)
        x = max(50, min(x, self.width - 50))
        y = max(30, min(y, self.height - 30))
        self._positions_cache[index] = (x, y)
        return x, y

    @staticmethod
    def _vary_color(base_color: str, variation: float = 0.2) -> str:
        hex_color = base_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        l = max(0.3, min(0.8, l + random.uniform(-variation, variation)))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    def create_svg_cloud(self, words: List[Dict[str, object]], title: str = "") -> str:
        if not words:
            return "<div style='text-align:center;padding:16px'>No data available</div>"
        words = sorted(words, key=lambda x: x['value'], reverse=True)
        values = [w['value'] for w in words]
        min_val, max_val = min(values), max(values)
        svg: List[str] = [
            f'<svg width="{self.width}" height="{self.height}" style="background:linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%);border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.06)">'
        ]
        if title:
            svg.append(
                f'<text x="{self.width/2}" y="25" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">{title}</text>'
            )
        used_positions: List[tuple] = []
        for i, w in enumerate(words):
            size = self._calculate_font_size(int(w['value']), min_val, max_val)
            color = self._vary_color(w.get('color', '#4a5568'))
            x, y = self._spiral_position(i)
            attempts = 0
            while attempts < 10:
                collision = False
                for px, py, ps in used_positions:
                    dist = math.hypot(x - px, y - py)
                    if dist < (size + ps) * 0.5:
                        collision = True
                        break
                if not collision:
                    break
                x += random.randint(-20, 20)
                y += random.randint(-20, 20)
                x = max(50, min(x, self.width - 50))
                y = max(50, min(y, self.height - 50))
                attempts += 1
            used_positions.append((x, y, size))
            rotation = random.randint(-15, 15)
            opacity = 0.7 + (size / 60) * 0.3
            svg.append(
                f'<g transform="translate({x},{y}) rotate({rotation})">'
                f'<text x="0" y="0" text-anchor="middle" font-size="{size}" font-family="Arial, sans-serif" '
                f'font-weight="{700 if size>30 else 400}" fill="{color}" opacity="{opacity}" '
                f'style="cursor:default">'
                f'<title>{w["text"]}: {w["value"]}</title>{w["text"]}</text></g>'
            )
        svg.append('</svg>')
        return ''.join(svg)

    def create_canvas_cloud(self, words: List[Dict[str, object]], title: str = "") -> str:
        if not words:
            return "<div style='text-align:center;padding:16px'>No data available</div>"
        canvas_id = f"wordcloud_{random.randint(1000, 9999)}"
        import json
        words_json = json.dumps(words)
        return (
            f'<canvas id="{canvas_id}" width="{self.width}" height="{self.height}" '
            f'style="border:1px solid #e5e7eb;border-radius:8px"></canvas>'
            '<script>(function(){'
            f'const c=document.getElementById("{canvas_id}");const ctx=c.getContext("2d");'
            f'const words={words_json};'
            f'const grad=ctx.createLinearGradient(0,0,{self.width},{self.height});'
            'grad.addColorStop(0,"#f8f9fa");grad.addColorStop(1,"#e9ecef");ctx.fillStyle=grad;'
            f'ctx.fillRect(0,0,{self.width},{self.height});'
            f'if("{title}"){{ctx.font="700 20px Arial";ctx.fillStyle="#111827";ctx.textAlign="center";ctx.fillText("{title}",{self.width/2},30);}}'
            'const vals=words.map(w=>w.value);const min=Math.min(...vals),max=Math.max(...vals);'
            f'const cx={self.width/2},cy={self.height/2};'
            'words.sort((a,b)=>b.value-a.value).forEach((w,i)=>{'
            'const size=10+((w.value-min)/(max-min||1))*50;'
            'const ang=i*0.5,rad=5+i*2;let x=cx+rad*Math.cos(ang),y=cy+rad*Math.sin(ang);'
            f'x=Math.max(50,Math.min(x,{self.width}-50));y=Math.max(50,Math.min(y,{self.height}-50));'
            'ctx.save();ctx.translate(x,y);ctx.rotate((Math.random()-0.5)*0.3);'
            'ctx.font=`${size>30?"bold":"normal"} ${size}px Arial`;ctx.fillStyle=w.color||"#4a5568";'
            'ctx.globalAlpha=0.7+(size/60)*0.3;ctx.textAlign="center";ctx.fillText(w.text,0,0);ctx.restore();'
            '});})();</script>'
        )


def create_python_wordcloud_image(
    words: List[Dict[str, object]],
    *,
    width: int = 1200,
    height: int = 450,
    background_color: str = "white",
    max_words: int = 300,
    colormap: str = "coolwarm",
    prefer_horizontal: float = 0.9,
    random_state: int = 42,
) -> Optional[bytes]:
    """Generate a PNG image (as bytes) using the `wordcloud` library from a list of `{text, value}` items.

    Returns None if the optional dependency is not available or if `words` is empty.
    """
    if not words:
        return None
    if not _WC_AVAILABLE or WordCloud is None:
        return None
    # Build frequencies mapping
    freqs: Dict[str, int] = {}
    for w in words:
        try:
            t = str(w.get("text", "")).strip()
            v = int(w.get("value", 0))
        except Exception:
            continue
        if t and v > 0:
            freqs[t] = freqs.get(t, 0) + v
    if not freqs:
        return None

    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        max_words=max_words,
        colormap=colormap,
        prefer_horizontal=prefer_horizontal,
        random_state=random_state,
        normalize_plurals=False,
        collocations=False,
    )
    wc.generate_from_frequencies(freqs)

    image = wc.to_image()  # PIL Image
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
