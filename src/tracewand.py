#!/usr/bin/env python3
"""
TraceWand: an interactive magic-wand raster-to-SVG tracer.

Usage:
    python tracewand.py input.png

Controls:
    Left click : choose the seed point and preview/vectorize the region
    Sliders    : retrace the current seed with updated parameters
    S          : save the current preview as SVG
    R          : reset the preview
    Q / Esc    : quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np
import svgwrite

try:
    import potrace
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: potrace / pypotrace.\n"
        "Install it first, for example: pip install pypotrace"
    ) from exc


COLOR_TOLERANCE = 24

RDP_FACTOR = 0.005

OPT_TOLERANCE = 0.4

CURVE_SMOOTHING_RADIUS = 2

POTRACE_TURDSIZE = 10

POTRACE_ALPHAMAX = 1.3

MORPH_KERNEL_SIZE = 3

DISPLAY_WIDTH_FRACTION = 2 / 3
DISPLAY_HEIGHT_FRACTION = 2 / 3
FALLBACK_SCREEN_SIZE = (1400, 900)
RETRACE_DEBOUNCE_SECONDS = 0.12

WINDOW_NAME = "TraceWand - click a region"
TOLERANCE_TRACKBAR = "Tolerance"
RDP_TRACKBAR = "RDP x10000"
OPT_TRACKBAR = "Opt x100"
SMOOTH_TRACKBAR = "Smooth"
RDP_TRACKBAR_SCALE = 10000
OPT_TRACKBAR_SCALE = 100


@dataclass
class TraceResult:
    click_x: int
    click_y: int
    selected_mask: np.ndarray
    optimized_mask: np.ndarray
    svg_path_data: str
    svg_node_count: int
    output_path: str


class TraceWandApp:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.image_bgr = self._load_image(image_path)
        self.preview_bgr = self.image_bgr.copy()
        self.initial_window_size = fit_image_to_screen(
            self.image_bgr.shape[1],
            self.image_bgr.shape[0],
        )
        self.display_canvas_size = self.initial_window_size
        self.image_view_rect = (0, 0, self.initial_window_size[0], self.initial_window_size[1])
        self.last_result: Optional[TraceResult] = None
        self.color_tolerance = COLOR_TOLERANCE
        self.rdp_factor = RDP_FACTOR
        self.opt_tolerance = OPT_TOLERANCE
        self.curve_smoothing_radius = CURVE_SMOOTHING_RADIUS
        self.current_seed: Optional[tuple[int, int]] = None
        self.retrace_pending = False
        self.last_parameter_change_time = 0.0

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return image

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar(
            TOLERANCE_TRACKBAR,
            WINDOW_NAME,
            self.color_tolerance,
            255,
            self._on_tolerance_change,
        )
        cv2.createTrackbar(
            RDP_TRACKBAR,
            WINDOW_NAME,
            int(round(self.rdp_factor * RDP_TRACKBAR_SCALE)),
            500,
            self._on_rdp_change,
        )
        cv2.createTrackbar(
            OPT_TRACKBAR,
            WINDOW_NAME,
            int(round(self.opt_tolerance * OPT_TRACKBAR_SCALE)),
            200,
            self._on_opt_change,
        )
        cv2.createTrackbar(
            SMOOTH_TRACKBAR,
            WINDOW_NAME,
            self.curve_smoothing_radius,
            12,
            self._on_smooth_change,
        )
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        self._render_preview()

        print("TraceWand is running.")
        print(
            f"Original image: {self.image_bgr.shape[1]}x{self.image_bgr.shape[0]} | "
            f"initial window: {self.initial_window_size[0]}x{self.initial_window_size[1]}"
        )
        print(
            "Left click: choose seed | sliders update current seed | "
            "S: save SVG | R: reset | Q/Esc: quit"
        )

        while True:
            key = cv2.waitKey(20) & 0xFF
            self._retrace_if_pending()
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                self.preview_bgr = self.image_bgr.copy()
                self.last_result = None
                self.current_seed = None
                self.retrace_pending = False
                self._render_preview()
            if key in (ord("s"), ord("S")) and self.last_result is not None:
                self._save_svg_result(self.last_result)
                print(f"[TraceWand] Saved {self.last_result.output_path}")
            elif key in (ord("s"), ord("S")):
                print("[TraceWand] Nothing to save yet. Left click a region first.")

        cv2.destroyAllWindows()

    def _on_tolerance_change(self, value: int) -> None:
        self.color_tolerance = int(value)
        self._mark_retrace_pending()

    def _on_rdp_change(self, value: int) -> None:
        self.rdp_factor = max(0.0, value / RDP_TRACKBAR_SCALE)
        self._mark_retrace_pending()

    def _on_opt_change(self, value: int) -> None:
        self.opt_tolerance = max(0.0, value / OPT_TRACKBAR_SCALE)
        self._mark_retrace_pending()

    def _on_smooth_change(self, value: int) -> None:
        self.curve_smoothing_radius = max(0, int(value))
        self._mark_retrace_pending()

    def _mark_retrace_pending(self) -> None:
        if self.current_seed is None:
            return
        self.retrace_pending = True
        self.last_parameter_change_time = time.monotonic()

    def _retrace_if_pending(self) -> None:
        if not self.retrace_pending or self.current_seed is None:
            return
        if time.monotonic() - self.last_parameter_change_time < RETRACE_DEBOUNCE_SECONDS:
            return

        self.retrace_pending = False
        self._trace_and_update_preview(
            self.current_seed[0],
            self.current_seed[1],
            label="Updated preview",
        )

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        image_x, image_y = self._display_to_image_point(x, y)
        self.current_seed = (image_x, image_y)
        self.retrace_pending = False
        self._trace_and_update_preview(image_x, image_y, label="Preview ready")

    def _trace_and_update_preview(self, image_x: int, image_y: int, label: str) -> None:
        try:
            result = self.trace_click(image_x, image_y)
        except Exception as exc:
            self.last_result = None
            print(
                f"[TraceWand] Failed to trace click ({image_x}, {image_y}): {exc}",
                file=sys.stderr,
            )
            return

        self.last_result = result
        self.preview_bgr = self._build_preview(result.optimized_mask)
        self._render_preview()
        print(
            f"[TraceWand] {label} | "
            f"click: ({image_x}, {image_y}) | "
            f"tolerance: {self.color_tolerance} | "
            f"rdp: {self.rdp_factor:.4f} | "
            f"opt: {self.opt_tolerance:.2f} | "
            f"smooth: {self.curve_smoothing_radius} | "
            f"SVG path total node count: {result.svg_node_count} | "
            f"press S to save"
        )

    def trace_click(self, x: int, y: int) -> TraceResult:
        selected_mask = create_magic_wand_mask(
            self.image_bgr,
            seed_point=(x, y),
            tolerance=self.color_tolerance,
        )
        optimized_mask = optimize_mask_for_low_node_svg(
            selected_mask,
            rdp_factor=self.rdp_factor,
            smoothing_radius=self.curve_smoothing_radius,
        )
        svg_path_data, svg_node_count = potrace_mask_to_svg_path(
            optimized_mask,
            opt_tolerance=self.opt_tolerance,
        )

        if not svg_path_data.strip():
            raise ValueError("Potrace returned an empty path. Try a larger or cleaner region.")

        output_path = make_output_path(self.image_path, x, y)
        result = TraceResult(
            click_x=x,
            click_y=y,
            selected_mask=selected_mask,
            optimized_mask=optimized_mask,
            svg_path_data=svg_path_data,
            svg_node_count=svg_node_count,
            output_path=output_path,
        )
        return result

    def _save_svg_result(self, result: TraceResult) -> None:
        sampled_bgr = self.image_bgr[result.click_y, result.click_x]
        fill_color = bgr_to_hex(sampled_bgr)
        height, width = self.image_bgr.shape[:2]

        dwg = svgwrite.Drawing(
            filename=result.output_path,
            size=(width, height),
            viewBox=f"0 0 {width} {height}",
            profile="full",
        )
        dwg.add(
            dwg.path(
                d=result.svg_path_data,
                fill=fill_color,
                stroke="none",
                fill_rule="evenodd",
            )
        )
        dwg.save()

    def _build_preview(self, optimized_mask: np.ndarray) -> np.ndarray:
        preview = self.image_bgr.copy()

        overlay = preview.copy()
        overlay[optimized_mask > 0] = (0, 220, 255)
        preview = cv2.addWeighted(overlay, 0.28, preview, 0.72, 0)

        contours, _hierarchy = cv2.findContours(
            optimized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        return preview

    def _render_preview(self) -> None:
        canvas_width, canvas_height = self.display_canvas_size
        self.display_canvas_size = (canvas_width, canvas_height)
        canvas = self._build_display_canvas(self.preview_bgr, canvas_width, canvas_height)
        cv2.imshow(WINDOW_NAME, canvas)

    def _build_display_canvas(
        self,
        image_bgr: np.ndarray,
        canvas_width: int,
        canvas_height: int,
    ) -> np.ndarray:
        image_height, image_width = image_bgr.shape[:2]
        scale = min(canvas_width / image_width, canvas_height / image_height)
        view_width = max(1, int(round(image_width * scale)))
        view_height = max(1, int(round(image_height * scale)))
        view_x = max(0, (canvas_width - view_width) // 2)
        view_y = max(0, (canvas_height - view_height) // 2)
        self.image_view_rect = (view_x, view_y, view_width, view_height)

        canvas = np.full((canvas_height, canvas_width, 3), 32, dtype=np.uint8)
        resized = cv2.resize(
            image_bgr,
            (view_width, view_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas[view_y : view_y + view_height, view_x : view_x + view_width] = resized
        return canvas

    def _display_to_image_point(self, display_x: int, display_y: int) -> tuple[int, int]:
        image_height, image_width = self.image_bgr.shape[:2]
        view_x, view_y, view_width, view_height = self.image_view_rect
        local_x = display_x - view_x
        local_y = display_y - view_y
        image_x = int(round(local_x * image_width / view_width))
        image_y = int(round(local_y * image_height / view_height))
        return (
            int(np.clip(image_x, 0, image_width - 1)),
            int(np.clip(image_y, 0, image_height - 1)),
        )


def create_magic_wand_mask(
    image_bgr: np.ndarray,
    seed_point: tuple[int, int],
    tolerance: int,
) -> np.ndarray:
    """Use cv2.floodFill to select a fixed-range color region as a binary mask."""
    height, width = image_bgr.shape[:2]
    x, y = seed_point
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Seed point out of bounds: ({x}, {y})")

    flood_image = image_bgr.copy()
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    flags = (
        4
        | cv2.FLOODFILL_FIXED_RANGE
        | cv2.FLOODFILL_MASK_ONLY
        | (255 << 8)
    )

    cv2.floodFill(
        flood_image,
        flood_mask,
        seedPoint=(x, y),
        newVal=(0, 0, 0),
        loDiff=lo_diff,
        upDiff=up_diff,
        flags=flags,
    )

    return flood_mask[1 : height + 1, 1 : width + 1].copy()


def fit_image_to_screen(image_width: int, image_height: int) -> tuple[int, int]:
    """Return a preview size that fits on screen while preserving aspect ratio."""
    screen_width, screen_height = get_screen_size()
    max_width = int(screen_width * DISPLAY_WIDTH_FRACTION)
    max_height = int(screen_height * DISPLAY_HEIGHT_FRACTION)
    scale = min(1.0, max_width / image_width, max_height / image_height)
    display_width = max(1, int(round(image_width * scale)))
    display_height = max(1, int(round(image_height * scale)))
    return display_width, display_height


def get_screen_size() -> tuple[int, int]:
    """Best-effort screen-size detection with a conservative fallback."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return screen_size
    except Exception:
        return FALLBACK_SCREEN_SIZE


def optimize_mask_for_low_node_svg(
    selected_mask: np.ndarray,
    rdp_factor: float = RDP_FACTOR,
    smoothing_radius: int = CURVE_SMOOTHING_RADIUS,
) -> np.ndarray:
    """
    Apply the required three-stage pre-Potrace optimization.

    Step 3.1: morphology opening removes tiny protrusions.
    Step 3.2: RDP simplifies contour geometry, then redraws clean filled shapes.
    Step 3.3 happens in potrace_mask_to_svg_path().
    """
    if selected_mask.dtype != np.uint8:
        selected_mask = selected_mask.astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    denoised_mask = cv2.morphologyEx(selected_mask, cv2.MORPH_OPEN, kernel)

    contours, _hierarchy = cv2.findContours(
        denoised_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    optimized_mask = np.zeros_like(denoised_mask)
    simplified_contours: list[np.ndarray] = []

    for contour in contours:
        if len(contour) < 3:
            continue

        contour_for_rdp = smooth_closed_contour(contour, smoothing_radius)

        perimeter = cv2.arcLength(contour_for_rdp, closed=True)
        if perimeter <= 0:
            continue

        epsilon = rdp_factor * perimeter
        simplified = cv2.approxPolyDP(contour_for_rdp, epsilon=epsilon, closed=True)
        if len(simplified) >= 3:
            simplified_contours.append(simplified)

    if simplified_contours:
        cv2.drawContours(
            optimized_mask,
            simplified_contours,
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
            lineType=cv2.LINE_8,
        )

    return optimized_mask


def smooth_closed_contour(contour: np.ndarray, radius: int) -> np.ndarray:
    """Smooth a closed contour with circular moving average before RDP simplification."""
    if radius <= 0 or len(contour) < 3:
        return contour

    points = contour.reshape(-1, 2).astype(np.float32)
    window_size = radius * 2 + 1
    if len(points) <= window_size:
        return contour

    smoothed = np.zeros_like(points)
    for offset in range(-radius, radius + 1):
        smoothed += np.roll(points, shift=offset, axis=0)
    smoothed /= window_size

    return np.rint(smoothed).astype(np.int32).reshape(-1, 1, 2)


def potrace_mask_to_svg_path(
    mask: np.ndarray,
    opt_tolerance: float = OPT_TOLERANCE,
) -> tuple[str, int]:
    """
    Trace a binary mask with Potrace and convert the result to SVG path data.

    Returns:
        (path_d, node_count)
    """
    binary = (mask > 0).astype(np.uint8)

    bitmap = potrace.Bitmap(binary)
    path_collection = bitmap.trace(
        turdsize=POTRACE_TURDSIZE,
        alphamax=POTRACE_ALPHAMAX,
        opttolerance=opt_tolerance,
    )

    commands: list[str] = []
    node_count = 0

    for curve in path_collection:
        start_x, start_y = point_to_xy(curve.start_point)
        commands.append(f"M {fmt(start_x)} {fmt(start_y)}")
        node_count += 1

        for segment in curve:
            end_x, end_y = point_to_xy(segment.end_point)
            if segment.is_corner:
                corner_x, corner_y = point_to_xy(segment.c)
                commands.append(
                    f"L {fmt(corner_x)} {fmt(corner_y)} "
                    f"L {fmt(end_x)} {fmt(end_y)}"
                )
                node_count += 2
            else:
                c1_x, c1_y = point_to_xy(segment.c1)
                c2_x, c2_y = point_to_xy(segment.c2)
                commands.append(
                    f"C {fmt(c1_x)} {fmt(c1_y)} "
                    f"{fmt(c2_x)} {fmt(c2_y)} "
                    f"{fmt(end_x)} {fmt(end_y)}"
                )
                node_count += 3

        commands.append("Z")

    return " ".join(commands), node_count


def point_to_xy(point: Iterable[float]) -> tuple[float, float]:
    x, y = point
    return float(x), float(y)


def fmt(value: float) -> str:
    """Format path coordinates compactly without losing useful subpixel precision."""
    rounded = round(float(value), 3)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def bgr_to_hex(color_bgr: np.ndarray) -> str:
    b, g, r = [int(v) for v in color_bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def make_output_path(image_path: str, x: int, y: int) -> str:
    image_dir = os.path.dirname(os.path.abspath(image_path))
    return os.path.join(image_dir, f"output_{x}_{y}.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive magic-wand tracing tool with aggressive SVG node reduction."
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=COLOR_TOLERANCE,
        help=f"Color tolerance for floodFill. Default: {COLOR_TOLERANCE}",
    )
    parser.add_argument(
        "--rdp-factor",
        type=float,
        default=RDP_FACTOR,
        help=f"RDP epsilon factor. Default: {RDP_FACTOR}",
    )
    parser.add_argument(
        "--opt-tolerance",
        type=float,
        default=OPT_TOLERANCE,
        help=f"Potrace opttolerance. Default: {OPT_TOLERANCE}",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=CURVE_SMOOTHING_RADIUS,
        help=f"Closed-contour smoothing radius before RDP. Default: {CURVE_SMOOTHING_RADIUS}",
    )
    return parser.parse_args()


def main() -> None:
    global COLOR_TOLERANCE, RDP_FACTOR, OPT_TOLERANCE, CURVE_SMOOTHING_RADIUS

    args = parse_args()
    COLOR_TOLERANCE = args.tolerance
    RDP_FACTOR = args.rdp_factor
    OPT_TOLERANCE = args.opt_tolerance
    CURVE_SMOOTHING_RADIUS = args.smooth

    app = TraceWandApp(args.image)
    app.run()


if __name__ == "__main__":
    main()
