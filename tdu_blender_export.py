"""
tdu_blender_export.py — Blender render script for TDU training data generation
-------------------------------------------------------------------------------
Run this from Blender's scripting workspace, or from the command line:

    blender --background my_scene.blend --python tdu_blender_export.py

What this script does:
  1. Applies a Halton sub-pixel jitter to the camera each frame
     (same pattern used by FSR/DLSS — gives the model overlapping
     sub-pixel samples to reconstruct from)
  2. Renders two passes per frame:
       - Input:  540p, 4 samples, no denoiser  (noisy low-res)
       - Target: 1080p, 256 samples, OIDN on   (clean high-res)
  3. Exports render passes as 32-bit EXR files:
       input/  → colour (noisy), motion vectors, depth
       target/ → colour (clean, denoised)
  4. Saves a metadata.json per sequence with jitter offsets and frame count

Output folder structure:
    OUTPUT_DIR/
      sequence_000/
        input/
          colour/   frame_0000.exr  frame_0001.exr  ...
          motion/   frame_0000.exr  ...
          depth/    frame_0000.exr  ...
        target/
          colour/   frame_0000.exr  ...
        metadata.json

Configurable constants are at the top of the file — adjust to taste.

Requirements:
    Blender 3.6 or 4.x with Cycles renderer.
    The scene must already be animated (camera + objects moving).
"""

import bpy
import json
import math
import os

# ---------------------------------------------------------------------------
# Configuration — edit these for your scene
# ---------------------------------------------------------------------------

OUTPUT_DIR = "/home/samuel/Documents/FSRURenderTest/sequence_000"  # output folder
START_FRAME = 1  # first frame to render
END_FRAME = 60  # last frame to render (60 frames per sequence)

# Input render settings (low-quality, noisy)
INPUT_RES_X = 960  # 540p landscape
INPUT_RES_Y = 540
INPUT_SAMPLES = 4  # intentionally noisy

# Target render settings (high-quality, clean)
TARGET_RES_X = 1920  # 1080p
TARGET_RES_Y = 1080
TARGET_SAMPLES = 256  # high enough that OIDN cleans it well

# Jitter pattern — Halton sequence base 2 (x) and base 3 (y)
# This is the same sub-pixel offset pattern used by FSR 4 and DLSS.
# We cycle through JITTER_COUNT offsets across frames.
JITTER_COUNT = 8  # how many distinct jitter positions to cycle through

# ---------------------------------------------------------------------------
# Halton sequence generator
# ---------------------------------------------------------------------------


def halton(index: int, base: int) -> float:
    """
    Returns the index-th element of the Halton low-discrepancy sequence
    for the given base. Output is in [0, 1).

    Halton sequences fill space more evenly than random sampling —
    each new sample fills the largest gap in the sequence so far.
    This is why FSR/DLSS use them for jitter: after N frames, the
    accumulated sub-pixel coverage is as uniform as possible.
    """
    result, f = 0.0, 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


def get_jitter_offsets(count: int) -> list[tuple[float, float]]:
    """
    Returns `count` (dx, dy) jitter offsets in [-0.5, 0.5] pixel units.
    Centred so the average offset over a full cycle is ~zero.
    """
    return [(halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5) for i in range(count)]


# ---------------------------------------------------------------------------
# Render pass setup helpers
# ---------------------------------------------------------------------------


def setup_render_passes(scene: bpy.types.Scene) -> None:
    """
    Enable the render passes we need on the active view layer.
    Called once before rendering starts.
    """
    vl = scene.view_layers[0]
    vl.use_pass_combined = True  # colour
    vl.use_pass_vector = True  # motion vectors (forward + backward)
    vl.use_pass_z = True  # depth


def setup_compositor_input(scene: bpy.types.Scene, out_dir: str) -> None:
    """
    Wire up the compositor to save input render passes (noisy, low-res)
    to EXR files. Called once before the input render loop.

    Node layout:
      Render Layers → File Output (colour)
      Render Layers → File Output (motion)
      Render Layers → File Output (depth)
    """
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    def make_output(name: str, path: str, slot: str) -> None:
        fo = tree.nodes.new("CompositorNodeOutputFile")
        fo.label = name
        fo.base_path = path
        fo.format.file_format = "OPEN_EXR"
        fo.format.color_depth = "32"
        fo.format.exr_codec = "ZIP"
        fo.file_slots[0].path = "frame_"  # Blender appends #### + .exr
        tree.links.new(rl.outputs[slot], fo.inputs[0])
        fo.location = (400, {"colour": 200, "motion": 0, "depth": -200}[name])

    make_output("colour", os.path.join(out_dir, "input", "colour"), "Image")
    make_output("motion", os.path.join(out_dir, "input", "motion"), "Vector")
    make_output("depth", os.path.join(out_dir, "input", "depth"), "Depth")


def setup_compositor_target(scene: bpy.types.Scene, out_dir: str) -> None:
    """
    Wire up the compositor to save the target render (clean, high-res).
    Applies the OIDN denoiser node before the file output.
    """
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    # OIDN denoiser — uses the albedo and normal passes for better results
    # if available, but works with colour-only too
    denoise = tree.nodes.new("CompositorNodeDenoise")
    denoise.location = (300, 0)
    denoise.use_hdr = True
    tree.links.new(rl.outputs["Image"], denoise.inputs["Image"])

    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = os.path.join(out_dir, "target", "colour")
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.format.exr_codec = "ZIP"
    fo.file_slots[0].path = "frame_"
    fo.location = (600, 0)
    tree.links.new(denoise.outputs["Image"], fo.inputs[0])


# ---------------------------------------------------------------------------
# Camera jitter application
# ---------------------------------------------------------------------------


def apply_jitter(
    scene: bpy.types.Scene, dx: float, dy: float, res_x: int, res_y: int
) -> None:
    """
    Applies a sub-pixel shift to the camera's sensor by adjusting its
    shift_x / shift_y properties.

    Blender's camera shift is in units of the sensor width (1.0 = one
    full sensor width of shift), so we convert from pixels to that unit.

    dx, dy are in pixels in [-0.5, 0.5].
    """
    cam = scene.camera
    if cam is None:
        raise RuntimeError("No active camera in scene.")

    # Convert pixel offset to sensor-unit offset
    cam.data.shift_x = dx / res_x
    cam.data.shift_y = dy / res_y


def clear_jitter(scene: bpy.types.Scene) -> None:
    """Reset camera jitter to zero (no shift)."""
    cam = scene.camera
    if cam:
        cam.data.shift_x = 0.0
        cam.data.shift_y = 0.0


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------


def make_dirs(base: str) -> None:
    for subdir in [
        "input/colour",
        "input/motion",
        "input/depth",
        "target/colour",
    ]:
        os.makedirs(os.path.join(base, subdir), exist_ok=True)


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------


def render_sequence() -> None:
    scene = bpy.context.scene
    jitter = get_jitter_offsets(JITTER_COUNT)

    make_dirs(OUTPUT_DIR)

    # Metadata we'll save at the end
    metadata = {
        "frames": [],
        "input_res": [INPUT_RES_X, INPUT_RES_Y],
        "target_res": [TARGET_RES_X, TARGET_RES_Y],
        "input_samples": INPUT_SAMPLES,
        "target_samples": TARGET_SAMPLES,
        "jitter_count": JITTER_COUNT,
    }

    setup_render_passes(scene)

    for frame_idx, frame_num in enumerate(range(START_FRAME, END_FRAME + 1)):
        scene.frame_set(frame_num)

        # Sub-pixel jitter for this frame (cycles through JITTER_COUNT offsets)
        dx, dy = jitter[frame_idx % JITTER_COUNT]

        print(f"\n[TDU] Frame {frame_num}/{END_FRAME}  jitter=({dx:.4f}, {dy:.4f})")

        # ── INPUT RENDER ──────────────────────────────────────────────────
        scene.render.engine = "CYCLES"
        scene.render.resolution_x = INPUT_RES_X
        scene.render.resolution_y = INPUT_RES_Y
        scene.cycles.samples = INPUT_SAMPLES
        scene.cycles.use_denoising = False

        apply_jitter(scene, dx, dy, INPUT_RES_X, INPUT_RES_Y)
        setup_compositor_input(scene, OUTPUT_DIR)
        bpy.ops.render.render(write_still=False)

        # ── TARGET RENDER ─────────────────────────────────────────────────
        # Target is rendered WITHOUT jitter — it's our clean reference.
        # The model learns to produce a jitter-free output; the loss
        # compares against this clean, centred ground truth.
        clear_jitter(scene)
        scene.render.resolution_x = TARGET_RES_X
        scene.render.resolution_y = TARGET_RES_Y
        scene.cycles.samples = TARGET_SAMPLES
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"

        setup_compositor_target(scene, OUTPUT_DIR)
        bpy.ops.render.render(write_still=False)

        # Record jitter offset for this frame (needed by the dataset loader
        # to pass as jitter_offset to the model)
        metadata["frames"].append(
            {
                "frame": frame_num,
                "jitter": [dx, dy],
            }
        )

    # Reset camera to no jitter when done
    clear_jitter(scene)

    # Save metadata
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[TDU] Done. Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_sequence()
