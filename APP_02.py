import gradio as gr
import os
import tempfile
import subprocess
import shutil
from PIL import Image
import time
import io
import zipfile

def get_blender_path():
    blender_path = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
    if os.path.exists(blender_path):
        return blender_path
    raise FileNotFoundError("Blender not found. Please install Blender 4.4 at the default location.")

def convert_png_to_svg_using_potrace(png_path, svg_path, threshold=60, resize_factor=2.0):
    """
    Convert a PNG image to SVG using Potrace with strong thresholding and optional upscaling.
    """
    from PIL import Image

    pbm_path = png_path.replace('.png', '.pbm')

    try:
        # Step 1: Open and optionally upscale
        img = Image.open(png_path).convert("L")  # Grayscale
        if resize_factor != 1.0:
            new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
            img = img.resize(new_size, Image.LANCZOS)

        # Step 2: Strong binary threshold
        img = img.point(lambda x: 0 if x < threshold else 255, mode='1')

        # Step 3: Save PBM for Potrace
        img.save(pbm_path)
        print(f"[✓] PBM saved at: {pbm_path}")
    except Exception as e:
        raise RuntimeError(f"[✗] Failed to create PBM: {e}")

    try:
        potrace_path = r"C:\Tools\Potrace\potrace.exe"  # ← Update this to your Potrace path
        if not os.path.exists(potrace_path):
            raise FileNotFoundError(f"Potrace not found at {potrace_path}")

        cmd = [potrace_path, pbm_path, "--svg", "-o", svg_path]
        print(f"Running Potrace: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("Potrace stdout:", result.stdout)
        print("Potrace stderr:", result.stderr)

        if result.returncode != 0 or not os.path.exists(svg_path):
            raise RuntimeError("Potrace failed or SVG not created.")

        print(f"[✓] SVG created: {svg_path}")
    except Exception as e:
        raise RuntimeError(f"[✗] Potrace error: {e}")
    
def clean_svg_by_path_length(svg_path, min_length=5.0):
    """
    Removes paths from an SVG file that are shorter than a threshold.
    """
    import xml.etree.ElementTree as ET
    from svgpathtools import parse_path

    if not os.path.exists(svg_path):
        raise FileNotFoundError(f"SVG not found: {svg_path}")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Inkscape uses this namespace
    ns = {"svg": "http://www.w3.org/2000/svg"}

    # Find and filter <path> elements
    removed = 0
    for elem in list(root.findall(".//svg:path", ns)):
        d_attr = elem.get("d")
        if d_attr:
            try:
                path = parse_path(d_attr)
                if path.length() < min_length:
                    root.remove(elem)
                    removed += 1
            except Exception as e:
                print(f"Skipping path due to error: {e}")

    # Save cleaned SVG
    tree.write(svg_path)
    print(f"[✓] Removed {removed} short paths from: {svg_path}")



def convert_svg_to_dxf(svg_path, dxf_path):
    inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"
    if not os.path.exists(inkscape_path):
        raise FileNotFoundError("Inkscape not found at expected path.")

    cmd = [
        inkscape_path,
        svg_path,
        "--export-type=dxf",
        f"--export-filename={dxf_path}"
    ]
    print(f"Converting SVG to DXF: {svg_path} → {dxf_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Inkscape stdout:", result.stdout)
    print("Inkscape stderr:", result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Inkscape conversion failed.")
    print(f"DXF exists: {os.path.exists(dxf_path)}")

def process_model(model_file_path):
    try:
        blender_path = get_blender_path()
        script_path = os.path.join(os.path.dirname(__file__), "generate_drawings.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Blender script not found at {script_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "input_model" + os.path.splitext(model_file_path)[1])
            shutil.copy(model_file_path, model_path)

            output_dir = os.path.join(temp_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                blender_path,
                "--background",
                "--python", script_path,
                "--",
                model_path,
                output_dir
            ]

            print("Running Blender:", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            print("Blender stdout:", result.stdout)
            print("Blender stderr:", result.stderr)
            if result.returncode != 0:
                raise RuntimeError(result.stderr or "Blender failed.")

            # Load PNGs
            images = []
            for f in os.listdir(output_dir):
                if f.lower().endswith('.png'):
                    path = os.path.join(output_dir, f)
                    with Image.open(path) as img:
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        buf.seek(0)
                        images.append(buf.getvalue())

            # Collect only DXFs from Blender
            dxf_paths = [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.lower().endswith('.dxf')
            ]

            # Zip DXFs
            zip_path = os.path.join(output_dir, "sections_only.dxf.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for dxf in dxf_paths:
                    zipf.write(dxf, arcname=os.path.basename(dxf))

            final_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            shutil.copy(zip_path, final_zip.name)

            return [Image.open(io.BytesIO(img)) for img in images], final_zip.name

    except subprocess.TimeoutExpired:
        raise gr.Error("Blender timed out (15+ minutes). Try a smaller model.")
    except Exception as e:
        raise gr.Error(f"Failed to generate drawings: {e}")


# Gradio UI
with gr.Blocks(title="Architectural Drawing Generator") as demo:
    gr.Markdown("""
    # Architectural Drawing Generator
    Upload a 3D model (GLB/OBJ/FBX) to generate plan/section views and DXF files.
    """)

    with gr.Row():
        with gr.Column():
            model_input = gr.Model3D(label="Upload 3D Model")
            process_btn = gr.Button("Generate Drawings", variant="primary")

        with gr.Column():
            gallery = gr.Gallery(label="Generated Drawings", columns=3, preview=True)
            dxf_download = gr.File(label="Download DXF (Traced from Images)", interactive=False)

    process_btn.click(fn=process_model, inputs=model_input, outputs=[gallery, dxf_download])

if __name__ == "__main__":
    demo.launch(debug=True)
