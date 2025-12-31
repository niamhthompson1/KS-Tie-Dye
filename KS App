import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from pathlib import Path

# -----------------------------
# Helpers & Config
# -----------------------------

def hex_to_rgb(hex_color: str):
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

brand_colors_hex = ["CCB102", "FE7B01", "FF87B1", "4892FD", "C6C8B1", "000000", "FFFFFF"]
brand_colors_names = ["KS Yellow", "KS Orange", "KS Pink", "KS Blue", "KS Grey", "KS Black", "KS White"]


# -----------------------------
# Filename handling
# -----------------------------

def clean_filename(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "script"
    p = Path(name)
    if p.suffix.lower() in [".txt", ".py"]:
        return p.stem
    return name


# -----------------------------
# Export helpers
# -----------------------------

def export_png_bytes(img_rgb: np.ndarray, dpi: int) -> bytes:
    buf = BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG", dpi=(dpi, dpi))
    return buf.getvalue()


def export_pdf_bytes_pillow(img_rgb: np.ndarray, dpi: int) -> bytes:
    """
    No-reportlab fallback.
    Note: Photoshop may not always interpret PDF physical size/DPI as expected.
    PNG is the most reliable for Photoshop DPI workflows.
    """
    buf = BytesIO()
    Image.fromarray(img_rgb).convert("RGB").save(buf, format="PDF", resolution=dpi)
    return buf.getvalue()


# -----------------------------
# Image Generation (UNCHANGED)
# -----------------------------

def add_noise(img_array, noise_strength=0.05):
    h, w = img_array.shape[:2]
    if noise_strength <= 0:
        epsilon = 1e-5
        return np.clip(img_array.astype(np.float32) + epsilon, 0, 255).astype(np.uint8)

    scale = min(h, w) / 800
    effective_strength = noise_strength * scale * 0.5
    noise = np.random.normal(0, 255 * effective_strength, img_array.shape).astype(np.float32)
    noisy_img = img_array.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def make_spiral_with_circular_blots(
    h, w, colors_array, shape_complexity,
    blot_size_factor=1.5, frame_jitter=0, bg_color_idx=None
):
    bg_color = colors_array[bg_color_idx] if bg_color_idx is not None else np.array([255, 255, 255])
    pattern = np.ones((h, w, 3), dtype=np.float32) * bg_color[None, None, :]

    n_rings = int(3 + shape_complexity * 12)
    n_blots_per_ring = 60

    for ring in range(n_rings):
        radius = (ring + 1) / n_rings * 1.2
        for b in range(n_blots_per_ring):
            theta = 2 * np.pi * b / n_blots_per_ring + ring * 0.5
            cx = radius * np.cos(theta) + 0.01 * np.sin(frame_jitter + b)
            cy = radius * np.sin(theta) + 0.01 * np.cos(frame_jitter + ring)
            px = int((cx + 1) / 2 * (w - 1))
            py = int((cy + 1) / 2 * (h - 1))

            min_size, max_size = 40, 120
            blot_radius = int(np.random.randint(min_size, max_size) * blot_size_factor)

            color_idx = np.random.randint(len(colors_array))
            color = colors_array[color_idx]

            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((x_grid - px) ** 2 + (y_grid - py) ** 2)
            mask = np.clip(1 - dist / blot_radius, 0, 1)
            pattern = pattern * (1 - mask[..., None]) + mask[..., None] * color

    return np.clip(pattern, 0, 255).astype(np.uint8)


def psychedelic_tiedye(
    out_size=(800, 800),
    style="spiral",
    distortion_strength=1.0,
    shape_complexity=1.0,
    colors=None,
    blend_colors=True,
    noise_strength=0.05,
    hue_shift=0,
    sat_factor=1.0,
    val_factor=1.0,
    wavy_edges=0.0,
    scatter_color_rgb=None,
    blot_size_factor=1.5,
    frame_jitter=0,
    bg_color_idx=None
):
    h, w = out_size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X)

    if distortion_strength > 0:
        dx = distortion_strength * 0.2 * (np.random.rand(*X.shape) - 0.5)
        dy = distortion_strength * 0.2 * (np.random.rand(*Y.shape) - 0.5)
        X += dx
        Y += dy
        R = np.sqrt(X**2 + Y**2)
        T = np.arctan2(Y, X)

    colors_array = np.array(colors, dtype=np.float32)
    tiedye = None

    if style == "spiral":
        spiral_turns = int(3 + shape_complexity * 10)
        spiral_freq = 5 + shape_complexity * 10
        pattern = np.sin(R * spiral_freq + T * spiral_turns)

    elif style == "spiral_with_stripes":
        spiral_turns = int(3 + shape_complexity * 10)
        spiral_freq = 5 + shape_complexity * 10
        stripe_count = int(6 + shape_complexity * 12)
        spiral_pattern = np.sin(R * spiral_freq + T * spiral_turns)
        stripes_pattern = np.sin(T * stripe_count * np.pi)
        stripes_pattern = np.sign(stripes_pattern) * np.exp(-((T % (2*np.pi/stripe_count)) / 0.08)**2)
        pattern = spiral_pattern + 0.5 * stripes_pattern

    elif style == "spiral_with_wavy_stripes":
        spiral_turns = int(3 + shape_complexity * 10)
        spiral_freq = 5 + shape_complexity * 10
        stripe_count = int(6 + shape_complexity * 12)
        waviness = wavy_edges

        wavy_offset_x = waviness * 0.08 * np.sin(12*Y + np.random.rand()*5) + waviness * 0.05 * np.cos(18*X + np.random.rand()*5)
        wavy_offset_y = waviness * 0.08 * np.cos(14*X + np.random.rand()*5) + waviness * 0.05 * np.sin(16*Y + np.random.rand()*5)

        R_wavy = R + wavy_offset_x
        T_wavy = T + wavy_offset_y

        spiral_pattern = np.sin(R_wavy * spiral_freq + T_wavy * spiral_turns)
        stripes_pattern = np.sin(T_wavy * stripe_count * np.pi)
        stripes_mask = np.sign(stripes_pattern) * np.exp(-((T_wavy % (2*np.pi/stripe_count)) / 0.08)**2)

        base_pattern = spiral_pattern + 0.5 * stripes_mask
        base_pattern = (base_pattern - base_pattern.min()) / (base_pattern.max() - base_pattern.min() + 1e-9)

        n_colors = len(colors_array)
        pattern_scaled = base_pattern * (n_colors - 1)
        idx_lower = np.floor(pattern_scaled).astype(int)
        idx_upper = np.clip(idx_lower + 1, 0, n_colors - 1)
        t = pattern_scaled - idx_lower

        tiedye = (1-t[..., None]) * colors_array[idx_lower] + t[..., None] * colors_array[idx_upper]
        tiedye = tiedye.astype(np.uint8)

        if scatter_color_rgb is not None:
            stripe_mask = (np.abs(np.sin(T_wavy * stripe_count * np.pi)) > 0.9)
            scatter_prob = 0.02 + 0.03 * shape_complexity
            random_mask = np.random.rand(*stripe_mask.shape) < scatter_prob
            final_mask = stripe_mask & random_mask
            tiedye[final_mask] = scatter_color_rgb

    elif style == "spiral_with_circular_blots":
        tiedye = make_spiral_with_circular_blots(
            h, w, colors_array, shape_complexity,
            blot_size_factor=blot_size_factor,
            frame_jitter=frame_jitter,
            bg_color_idx=bg_color_idx
        )

    elif style == "concentric circles":
        if shape_complexity <= 0.01:
            pattern = np.ones_like(R)
        else:
            ring_count = int(3 + shape_complexity * 15)
            pattern = np.sin(R * ring_count * np.pi)

    else:
        pattern = np.sin(R * 12 + T * 6)

    if tiedye is None:
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-9)
        if blend_colors:
            n_colors = len(colors_array)
            pattern_scaled = pattern * (n_colors - 1)
            idx_lower = np.floor(pattern_scaled).astype(int)
            idx_upper = np.clip(idx_lower + 1, 0, n_colors - 1)
            t = pattern_scaled - idx_lower
            tiedye = (1-t[..., None]) * colors_array[idx_lower] + t[..., None] * colors_array[idx_upper]
            tiedye = tiedye.astype(np.uint8)
        else:
            n_colors = len(colors_array)
            band_indices = np.floor(pattern * n_colors).astype(int) % n_colors
            tiedye = colors_array[band_indices].astype(np.uint8)

    tiedye = add_noise(tiedye, noise_strength)

    img_hsv = cv2.cvtColor(tiedye, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[..., 0] = (img_hsv[..., 0] + hue_shift) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * sat_factor, 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * val_factor, 0, 255)
    result_adjusted = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return result_adjusted


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("KS Tie-Dye Generator")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Size")

    max_px = st.slider("Max render size (px)", 4000, 20000, 4000, 500)
    st.caption("Higher max values may be slow / memory heavy.")

    size_mode = st.radio("Units", ["Pixels", "Centimeters"], horizontal=True)
    export_dpi = st.slider("Export DPI", 100, 600, 300, 10)

    if size_mode == "Pixels":
        width_px = int(st.number_input("Width (px)", 64, int(max_px), 800, 50))
        height_px = int(st.number_input("Height (px)", 64, int(max_px), 800, 50))
        width_cm = height_cm = None
    else:
        # 0.1 cm → 400.0 cm (== 1 mm → 4000 mm)
        width_cm = float(st.number_input("Width (cm)", 0.1, 400.0, 10.0, 0.1))
        height_cm = float(st.number_input("Height (cm)", 0.1, 400.0, 10.0, 0.1))

        desired_w_px = int((width_cm / 2.54) * export_dpi)
        desired_h_px = int((height_cm / 2.54) * export_dpi)

        width_px = min(desired_w_px, int(max_px))
        height_px = min(desired_h_px, int(max_px))

        if (desired_w_px != width_px) or (desired_h_px != height_px):
            st.warning(
                f"Requested {desired_w_px:,}×{desired_h_px:,} px at {export_dpi} DPI, "
                f"capped to {width_px:,}×{height_px:,} px by Max render size."
            )
        st.caption(f"Render: {width_px:,} × {height_px:,} px")

    export_format = st.radio("Export format", ["PNG", "PDF"])

    st.subheader("Look")
    distortion_strength = st.slider("Distortion strength / chaos", 0.0, 2.0, 1.0, 0.05)
    shape_complexity = st.slider("Shape Complexity", 0.0, 2.0, 1.0, 0.05)
    noise_strength = st.slider("Noise / Grain intensity", 0.0, 1.0, 0.05, 0.01)
    blend_colors = st.checkbox("Blend colors smoothly", value=True)

    style = st.selectbox(
        "Choose a tie-dye style",
        [
            "spiral",
            "spiral_with_stripes",
            "spiral_with_wavy_stripes",
            "spiral_with_circular_blots",
            "concentric circles",
        ],
    )

    wavy_edges = 0.0
    scatter_color_rgb = None
    bg_color_idx = None
    blot_size_factor = 1.5

    if style == "spiral_with_wavy_stripes":
        wavy_edges = st.slider("Waviness / irregularity", 0.0, 1.0, 0.3, 0.01)
        scatter = st.checkbox("Scatter a single color along straight stripes?")
        if scatter:
            selected_color = st.selectbox("Scatter color", options=brand_colors_names)
            scatter_color_rgb = hex_to_rgb(brand_colors_hex[brand_colors_names.index(selected_color)])

    if style == "spiral_with_circular_blots":
        bg_color_idx = st.selectbox(
            "Background color",
            range(len(brand_colors_names)),
            format_func=lambda i: brand_colors_names[i],
        )
        blot_size_factor = st.slider("Blot Size", 0.5, 3.0, 1.5, 0.1)

selected_colors = st.multiselect(
    "Choose 2–4 brand colors",
    options=brand_colors_names,
    default=brand_colors_names[:3],
)
selected_brand_colors_rgb = [
    hex_to_rgb(brand_colors_hex[brand_colors_names.index(c)]) for c in selected_colors
]

with col1:
    if not selected_brand_colors_rgb:
        selected_brand_colors_rgb = [hex_to_rgb(c) for c in brand_colors_hex[:3]]

    result = psychedelic_tiedye(
        out_size=(height_px, width_px),
        style=style,
        distortion_strength=distortion_strength,
        shape_complexity=shape_complexity,
        colors=selected_brand_colors_rgb,
        blend_colors=blend_colors,
        noise_strength=noise_strength,
        wavy_edges=wavy_edges,
        scatter_color_rgb=scatter_color_rgb,
        blot_size_factor=blot_size_factor,
        bg_color_idx=bg_color_idx,
    )

    st.image(result, caption="Generated Tie-Dye", use_container_width=True)

    if export_format == "PNG":
        out_bytes = export_png_bytes(result, export_dpi)
        st.download_button("Download Tie-Dye (PNG)", out_bytes, "tie_dye.png", "image/png")
    else:
        out_bytes = export_pdf_bytes_pillow(result, export_dpi)
        st.download_button("Download Tie-Dye (PDF)", out_bytes, "tie_dye.pdf", "application/pdf")

    if size_mode == "Centimeters" and width_cm is not None and height_cm is not None:
        st.caption(
            f"Target: {width_cm:.1f} × {height_cm:.1f} cm at {export_dpi} DPI "
            f"(render {width_px:,} × {height_px:,} px)."
        )
    else:
        st.caption(
            f"Target: {width_px:,} × {height_px:,} px at {export_dpi} DPI "
            f"= {width_px/export_dpi:.2f} × {height_px/export_dpi:.2f} inches in Photoshop."
        )

st.divider()
st.header("Python Script")

code = st.text_area("Edit Python", height=250)
raw = st.text_input("Filename", "tie_dye_script")
name = clean_filename(raw)

st.caption(f"Saving as: **{name}.py**")

st.download_button(
    "Save .py",
    data=(code or "").encode("utf-8"),
    file_name=f"{name}.py",
    mime="application/octet-stream",
)

