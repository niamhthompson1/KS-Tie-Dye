import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO

# -----------------------------
# Helpers & Config
# -----------------------------

# --- Helper to convert hex to RGB ---
def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

# Brand colors
brand_colors_hex = ["CCB102", "FE7B01", "FF87B1", "4892FD", "C6C8B1", "000000", "FFFFFF"]
brand_colors_names = ["KS Yellow", "KS Orange", "KS Pink", "KS Blue", "KS Grey", "KS Black", "KS White"]

# -----------------------------
# Image Generation
# -----------------------------

# --- Add noise/grain ---
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

# --- Spiral with circular blots ---
def make_spiral_with_circular_blots(h, w, colors_array, shape_complexity,
                                    blot_size_factor=1.5, frame_jitter=0, bg_color_idx=None):
    bg_color = colors_array[bg_color_idx] if bg_color_idx is not None else np.array([255, 255, 255])
    pattern = np.ones((h, w, 3), dtype=np.float32) * bg_color[None, None, :]

    n_rings = int(3 + shape_complexity * 12)
    n_blots_per_ring = 60

    # Generate blots
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

            # Draw blot (circle) with soft edge
            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((x_grid - px) ** 2 + (y_grid - py) ** 2)
            mask = np.clip(1 - dist / blot_radius, 0, 1)
            pattern = pattern * (1 - mask[..., None]) + mask[..., None] * color

    return np.clip(pattern, 0, 255).astype(np.uint8)

# --- Tie-dye generator ---
def psychedelic_tiedye(out_size=(800, 800),
                       style="spiral", distortion_strength=1.0,
                       shape_complexity=1.0, colors=None,
                       blend_colors=True, noise_strength=0.05,
                       hue_shift=0, sat_factor=1.0, val_factor=1.0,
                       wavy_edges=0.0, scatter_color_rgb=None,
                       blot_size_factor=1.5, frame_jitter=0,
                       bg_color_idx=None):

    h, w = out_size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X)

    # Distortion / chaos
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
        base_pattern = (base_pattern - base_pattern.min()) / (base_pattern.max() - base_pattern.min())
        n_colors = len(colors_array)
        pattern_scaled = base_pattern * (n_colors - 1)
        idx_lower = np.floor(pattern_scaled).astype(int)
        idx_upper = np.clip(idx_lower + 1, 0, n_colors - 1)
        t = pattern_scaled - idx_lower
        tiedye = (1-t[..., None])*colors_array[idx_lower] + t[..., None]*colors_array[idx_upper]
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
            blot_size_factor=blot_size_factor, frame_jitter=frame_jitter, bg_color_idx=bg_color_idx
        )

    elif style == "rings":
        n_centers = int(1 + shape_complexity * 5)
        pattern = np.zeros_like(R)
        for _ in range(n_centers):
            cx, cy = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
            r = np.sqrt((X-cx)**2 + (Y-cy)**2)
            n_rings = int(3 + shape_complexity * 10)
            for i in range(n_rings):
                freq = 2 + i * 0.5
                pattern += np.sin(r * freq * np.pi)
        pattern /= (n_centers * n_rings)

    elif style == "concentric circles":
        if shape_complexity <= 0.01:
            pattern = np.ones_like(R)
        else:
            ring_count = int(3 + shape_complexity * 15)
            pattern = np.sin(R * ring_count * np.pi)

    elif style == "dot clusters (traditional Chinese)":
        n_dots = int(20 + shape_complexity * 50)
        pattern = np.zeros_like(R)
        for _ in range(n_dots):
            cx, cy = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
            r = np.sqrt((X-cx)**2 + (Y-cy)**2)
            pattern += np.exp(-((r * (2 + shape_complexity*3))**2))

    if tiedye is None:
        pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        if blend_colors:
            n_colors = len(colors_array)
            pattern_scaled = pattern * (n_colors - 1)
            idx_lower = np.floor(pattern_scaled).astype(int)
            idx_upper = np.clip(idx_lower + 1, 0, n_colors - 1)
            t = pattern_scaled - idx_lower
            tiedye = (1-t[..., None])*colors_array[idx_lower] + t[..., None]*colors_array[idx_upper]
            tiedye = tiedye.astype(np.uint8)
        else:
            n_colors = len(colors_array)
            band_indices = np.floor(pattern * n_colors).astype(int) % n_colors
            tiedye = colors_array[band_indices]

    tiedye = add_noise(tiedye, noise_strength)

    # HSV adjustments (OpenCV uses H in [0,179])
    img_hsv = cv2.cvtColor(tiedye, cv2.COLOR_RGB2HSV).astype(np.float32)
    img_hsv[..., 0] = (img_hsv[..., 0] + hue_shift) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1]*sat_factor, 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2]*val_factor, 0, 255)
    result_adjusted = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return result_adjusted

# -----------------------------
# GIF Utilities
# -----------------------------

def build_gif(frames_rgb, fps=8, loop=0, optimize=True):
    """
    frames_rgb: list of PIL Images in RGB mode
    Returns: BytesIO buffer of an animated GIF with a consistent palette
    """
    assert len(frames_rgb) > 0, "Need at least one frame to build a GIF"
    # Quantize first frame to create a global palette
    first_q = frames_rgb[0].quantize(method=Image.MEDIANCUT)
    # Use first frame's palette for all others for stable colors
    pal_frames = [first_q]
    for im in frames_rgb[1:]:
        q = im.quantize(palette=first_q)
        pal_frames.append(q)

    duration_ms = int(1000 / max(fps, 1))
    buf = BytesIO()
    pal_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pal_frames[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,     # restore to background between frames (avoid trails)
        optimize=optimize
    )
    buf.seek(0)
    return buf

def generate_tiedye_frames(n_frames, width, height, *,
                           hue_loop=True, scale=1.0,
                           style="spiral", distortion_strength=1.0,
                           shape_complexity=1.0, colors=((255,255,255),(0,0,0)),
                           blend_colors=True, noise_strength=0.05,
                           wavy_edges=0.0, scatter_color_rgb=None,
                           blot_size_factor=1.5, bg_color_idx=None):
    """Generate a list of PIL RGB frames, optionally downscaled for preview."""
    w = max(64, int(width * scale))
    h = max(64, int(height * scale))
    hue_step = (180 // max(n_frames, 1)) if hue_loop else 0

    frames = []
    for i in range(n_frames):
        frame = psychedelic_tiedye(
            out_size=(h, w),
            style=style,
            distortion_strength=distortion_strength + 0.3 * np.sin(i/4),
            shape_complexity=shape_complexity,
            colors=colors,
            blend_colors=blend_colors,
            noise_strength=noise_strength,
            hue_shift=(i * hue_step) % 180,
            wavy_edges=wavy_edges,
            scatter_color_rgb=scatter_color_rgb,
            blot_size_factor=blot_size_factor,
            frame_jitter=i,
            bg_color_idx=bg_color_idx
        )
        frames.append(Image.fromarray(frame).convert("RGB"))
    return frames

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("KS Tie-Dye Generator")

col1, col2 = st.columns([2, 1])

with col2:
    export_width = st.number_input("Export width (px)", 256, 4000, 800, 50)
    export_height = st.number_input("Export height (px)", 256, 4000, 800, 50)
    export_dpi = st.number_input("Export resolution (DPI)", 72, 600, 300, 10)
    export_format = st.radio("Export format", ["PNG", "PDF", "GIF"])

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
            "rings",
            "dot clusters (traditional Chinese)"
        ]
    )

    # Style-specific controls
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
        bg_color_idx = st.selectbox("Background color", range(len(brand_colors_names)), format_func=lambda i: brand_colors_names[i])
        blot_size_factor = st.slider("Blot Size", 0.5, 3.0, 1.5, 0.1)

    # --- GIF controls (only shown if GIF is selected) ---
    if export_format == "GIF":
        st.markdown("**GIF Settings**")
        gif_fps = st.slider("GIF FPS", 4, 24, 8, 1)
        gif_frames = st.slider("GIF frames", 8, 60, 24, 2)
        gif_mode = st.radio("GIF Mode", ["Preview", "Final render"], horizontal=True)

        # Preview controls
        preview_frames = st.slider("Preview frames", 4, 30, min(12, gif_frames), 2)
        preview_scale = st.slider("Preview scale (relative)", 0.1, 1.0, 0.4, 0.05)

        # Action buttons
        colA, colB = st.columns(2)
        do_preview = colA.button("Make Preview")
        do_render = colB.button("Render & Download GIF")

# Color selection
selected_colors = st.multiselect(
    "Choose 2–4 brand colors",
    options=brand_colors_names,
    default=brand_colors_names[:3]
)
selected_brand_colors_rgb = [hex_to_rgb(brand_colors_hex[brand_colors_names.index(c)]) for c in selected_colors]

with col1:
    # Fallback if nothing chosen
    if not selected_brand_colors_rgb:
        selected_brand_colors_rgb = [hex_to_rgb(c) for c in brand_colors_hex[:3]]

    # Live still preview (PNG style image in app)
    result = psychedelic_tiedye(
        out_size=(export_height, export_width),
        style=style,
        distortion_strength=distortion_strength,
        shape_complexity=shape_complexity,
        colors=selected_brand_colors_rgb,
        blend_colors=blend_colors,
        noise_strength=noise_strength,
        wavy_edges=wavy_edges,
        scatter_color_rgb=scatter_color_rgb,
        blot_size_factor=blot_size_factor,
        bg_color_idx=bg_color_idx
    )
    st.image(result, caption="Generated Tie-Dye", use_container_width=True)

    # Exports
    buf = BytesIO()
    result_pil = Image.fromarray(result)

    if export_format == "PNG":
        # Include DPI metadata
        result_pil.save(buf, format="PNG", dpi=(export_dpi, export_dpi))
        st.download_button("Download Tie-Dye Image (PNG)", data=buf.getvalue(),
                           file_name="tie_dye.png", mime="image/png")

    elif export_format == "PDF":
        result_pil.save(buf, format="PDF", resolution=export_dpi)
        st.download_button("Download Tie-Dye PDF", data=buf.getvalue(),
                           file_name="tie_dye.pdf", mime="application/pdf")

    elif export_format == "GIF":
        # Preview flow
        if 'gif_fps' in locals():  # shown only when GIF format selected
            if gif_mode == "Preview" and do_preview:
                frames_preview = generate_tiedye_frames(
                    n_frames=preview_frames,
                    width=export_width,
                    height=export_height,
                    scale=preview_scale,
                    style=style,
                    distortion_strength=distortion_strength,
                    shape_complexity=shape_complexity,
                    colors=selected_brand_colors_rgb,
                    blend_colors=blend_colors,
                    noise_strength=noise_strength,
                    wavy_edges=wavy_edges,
                    scatter_color_rgb=scatter_color_rgb,
                    blot_size_factor=blot_size_factor,
                    bg_color_idx=bg_color_idx
                )
                buf_preview = build_gif(frames_preview, fps=gif_fps, loop=0, optimize=True)
                st.image(buf_preview.getvalue(), caption="GIF Preview", use_container_width=True)
                st.info(f"Preview: {preview_frames} frames at {gif_fps} FPS, "
                        f"scaled to {int(export_width*preview_scale)}×{int(export_height*preview_scale)}")

            # Final render
            if do_render or (gif_mode == "Final render" and do_preview):
                frames_full = generate_tiedye_frames(
                    n_frames=gif_frames,
                    width=export_width,
                    height=export_height,
                    scale=1.0,
                    style=style,
                    distortion_strength=distortion_strength,
                    shape_complexity=shape_complexity,
                    colors=selected_brand_colors_rgb,
                    blend_colors=blend_colors,
                    noise_strength=noise_strength,
                    wavy_edges=wavy_edges,
                    scatter_color_rgb=scatter_color_rgb,
                    blot_size_factor=blot_size_factor,
                    bg_color_idx=bg_color_idx
                )
                buf_full = build_gif(frames_full, fps=gif_fps, loop=0, optimize=True)

                st.image(buf_full.getvalue(), caption="Final GIF", use_container_width=True)
                st.download_button(
                    "Download Tie-Dye GIF",
                    data=buf_full.getvalue(),
                    file_name="tie_dye.gif",
                    mime="image/gif"
                )
                st.success(f"Rendered: {gif_frames} frames at {gif_fps} FPS, size {export_width}×{export_height}")
