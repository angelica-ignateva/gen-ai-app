# sections.py (updated with DXF export)
import gradio as gr
import torch
import numpy as np
from rembg import remove
import tempfile
import os
from PIL import Image
import shutil
import io

from diffusers import DiffusionPipeline
from diffusers import FluxPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

import trimesh
import pyrender
import ezdxf

# Initialize diffusion pipelines
pipe_sdxl_turbo = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo").to("cuda")
pipe_sdxl = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to("cuda")
pipe_flux = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell").to("cuda")
hy3d_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')

state = {"model": None, "drawings_dir": None}


def generate_pavilion(description, num_steps=1, seed=None, model_choice="SDXL-Turbo"):
    try:
        prompt = f"{description}, aerial view, photorealistic, high detail, architectural visualization, futuristic design, intricate details, elegant structure"
        negative_prompt = "blurry, distorted, deformed, text"
        generator = torch.manual_seed(seed) if seed is not None else None
        chosen_pipe = {"SDXL": pipe_sdxl, "Flux": pipe_flux}.get(model_choice, pipe_sdxl_turbo)
        result = chosen_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_steps,
                             guidance_scale=0.0, generator=generator).images[0]
        return result, f"✅ Generated with {model_choice}! Seed: {seed if seed is not None else 'random'}"
    except Exception as e:
        return None, f"❌ Error generating image: {str(e)}"


def remove_background(input_image):
    try:
        if input_image is None:
            return None, "❌ No input image provided"
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        output = remove(input_image)
        return output, "✅ Background removed successfully"
    except Exception as e:
        return None, f"❌ Error removing background: {str(e)}"


def generate_3d_model(input_image, num_steps=1):
    try:
        if input_image is None:
            return None, "❌ No input image provided"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            input_image.save(tmp.name)
            image_path = tmp.name
        mesh = hy3d_pipeline(image=image_path, num_inference_steps=num_steps)[0]
        output_3d_path = os.path.join(os.getcwd(), "generated_model.glb")
        mesh.export(output_3d_path)
        state["model"] = output_3d_path
        html_preview = f"""
        <model-viewer src="file://{output_3d_path}" alt="3D Model" auto-rotate camera-controls style="width: 100%; height: 500px;"></model-viewer>
        """
        return html_preview, f"✅ 3D model generated successfully with {num_steps} steps!"
    except Exception as e:
        return None, f"❌ Error generating 3D model: {str(e)}"


def generate_2d_views(model_path, output_dir):
    mesh = trimesh.load(model_path, force='mesh')
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_pyrender)

    views = {
        "X-Section": np.array([1, 0, 0]),
        "Y-Section": np.array([0, -1, 0]),
        "Z-Section": np.array([0, 0, 1]),
        "Axonometric": np.array([1, -1, 1])
    }

    renderer = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
    for name, direction in views.items():
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cam_pose = trimesh.geometry.look_at(points=[mesh.centroid], eye=mesh.centroid + direction * 2.0)
        node = scene.add(camera, pose=cam_pose)
        color, _ = renderer.render(scene)
        image = Image.fromarray(color)
        image.save(os.path.join(output_dir, f"{name}.png"))
        scene.remove_node(node)
    renderer.delete()


def process_model(model_file):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "input_model.glb")
            shutil.copy(model_file, model_path)
            output_dir = os.path.join(temp_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            generate_2d_views(model_path, output_dir)
            images = []
            for f in os.listdir(output_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(Image.open(os.path.join(output_dir, f)))
            if not images:
                raise RuntimeError("No 2D views generated.")
            state["drawings_dir"] = output_dir
            return images
    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")


def export_drawings_to_dxf():
    try:
        output_dir = state.get("drawings_dir")
        if not output_dir:
            raise RuntimeError("No drawings available to export.")
        doc = ezdxf.new(setup=True)
        msp = doc.modelspace()
        for f in os.listdir(output_dir):
            if f.lower().endswith('.png'):
                img_path = os.path.join(output_dir, f)
                name = os.path.splitext(f)[0]
                # Insert raster image as DXF IMAGE entity
                # Note: this just links the image, not vectorizes it
                msp.add_image_def(filename=img_path, name=name)
        dxf_path = os.path.join(output_dir, "drawings.dxf")
        doc.saveas(dxf_path)
        return dxf_path
    except Exception as e:
        raise gr.Error(f"Export to DXF failed: {str(e)}")


pavilion_examples = [
    ["Glass pavilion with bamboo structure, modern and minimalist design"],
    ["Parametric wooden pavilion in an urban plaza, intricate organic shapes and circular cutouts, warm evening light, summer foliage."],
    ["Futuristic pavilion with carbon fiber frame and integrated LED lighting"]
]

with gr.Blocks(title="Pavilion Design Generator", theme=gr.themes.Glass()) as demo:
    gr.Markdown("""
    <div style='text-align: center; font-size: 2em; font-weight: bold; margin-bottom: 0.5em;'>
    🏛️ Pavilion 3D & 2D Design Generator
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("🎨 Generate Pavilion"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(label="Choose Model", choices=["SDXL-Turbo", "SDXL", "Flux"], value="SDXL-Turbo")
                    input_text = gr.Textbox(label="Description", placeholder="Describe your pavilion...", lines=4)
                    with gr.Accordion("Advanced Settings", open=False):
                        num_steps = gr.Slider(label="Generation Steps", minimum=1, maximum=50, value=1, step=1)
                        seed_input = gr.Number(label="Seed", value=415930750423718, precision=0)
                    gr.Examples(examples=pavilion_examples, inputs=input_text, label="Example Prompts")
                    generate_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=2):
                    output_image = gr.Image(label="Output", type="pil")
                    error_output = gr.Textbox(label="Status", interactive=False)

        with gr.TabItem("✂️ Remove Background"):
            with gr.Row():
                with gr.Column():
                    bg_input_image = gr.Image(label="Upload Pavilion Image", type="pil")
                    bg_remove_btn = gr.Button("Remove Background", variant="primary")
                with gr.Column():
                    bg_output_image = gr.Image(label="Background Removed", type="pil")
                    bg_status = gr.Textbox(label="Status", interactive=False)

        with gr.TabItem("🛖 3D Model Generator"):
            with gr.Row():
                with gr.Column():
                    model_input_image = gr.Image(label="Upload Pavilion Image", type="pil")
                    model_num_steps = gr.Slider(label="3D Generation Steps", minimum=1, maximum=10, value=1, step=1)
                    model_generate_btn = gr.Button("Generate 3D Model", variant="primary")
                with gr.Column():
                    model_output_html = gr.HTML(label="3D Model Preview")
                    model_status = gr.Textbox(label="Status", interactive=False)

        with gr.TabItem("📝 2D Drawings Generator"):
            with gr.Row():
                with gr.Column():
                    drawing_input_model = gr.Model3D(label="Upload 3D Model (.glb, .obj, .fbx)")
                    drawing_btn = gr.Button("Generate Drawings", variant="primary")
                    export_dwg_btn = gr.Button("Export as DWG (DXF)", variant="secondary")
                with gr.Column():
                    drawing_output_gallery = gr.Gallery(label="Generated Drawings", columns=3, preview=True)
                    export_dwg_output = gr.File(label="Download DXF File")

    generate_btn.click(fn=generate_pavilion, inputs=[input_text, num_steps, seed_input, model_choice], outputs=[output_image, error_output])
    bg_remove_btn.click(fn=remove_background, inputs=bg_input_image, outputs=[bg_output_image, bg_status])
    model_generate_btn.click(fn=generate_3d_model, inputs=[model_input_image, model_num_steps], outputs=[model_output_html, model_status])
    drawing_btn.click(fn=process_model, inputs=drawing_input_model, outputs=drawing_output_gallery)
    export_dwg_btn.click(fn=export_drawings_to_dxf, inputs=[], outputs=export_dwg_output)

if __name__ == "__main__":
    demo.launch(debug=True)
