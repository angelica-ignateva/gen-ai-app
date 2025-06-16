import gradio as gr
import torch
import numpy as np
import os
import shutil
import subprocess
import tempfile
import time
import io
from PIL import Image
from rembg import remove
from diffusers import DiffusionPipeline
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download
from InstantMesh.zero123plus.pipeline import Zero123PlusPipeline
from InstantMesh.src.utils.train_util import instantiate_from_config
from InstantMesh.src.utils.camera_util import FOV_to_intrinsics, get_zero123plus_input_cameras, get_circular_camera_poses
from InstantMesh.src.utils.mesh_util import save_glb
from InstantMesh.src.utils.infer_util import remove_background, resize_foreground
from torchvision.transforms import v2
from omegaconf import OmegaConf
from einops import rearrange

import sys
sys.path.append(os.path.abspath("InstantMesh"))

# Initialize SDXL for 2D generation
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Load InstantMesh config and models

config_path = 'InstantMesh/configs/instant-mesh-large.yaml'
config = OmegaConf.load(config_path)
model_config = config.model_config
infer_config = config.infer_config
IS_FLEXICUBES = True if os.path.basename(config_path).startswith('instant-mesh') else False

device = torch.device('cuda')


# Load Zero123Plus pipeline
pipeline = Zero123PlusPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    torch_dtype=torch.float16,
)
pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)
pipeline = pipeline.to(device)

# Load InstantMesh reconstruction model
model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model")
model = instantiate_from_config(model_config)
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
model.load_state_dict(state_dict, strict=True)
model = model.to(device).eval()

# # Global state to store 3D model path
# state = {"model": None}


# Utility functions
def preprocess(input_image, do_remove_bg=True):
    if do_remove_bg:
        input_image = remove(input_image)
        input_image = resize_foreground(input_image, 0.85)
    return input_image


def generate_multiview_images(input_image, sample_steps=75, sample_seed=42):
    seed_everything(sample_seed)
    z123_image = pipeline(input_image, num_inference_steps=sample_steps).images[0]
    return z123_image


def make3d(images):
    global model
    if IS_FLEXICUBES and hasattr(model, "init_flexicubes_geometry"):
        model.init_flexicubes_geometry(device)
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    render_cameras = get_circular_camera_poses(M=120, radius=2.5, elevation=10.0).flatten(-2)
    render_cameras = render_cameras.unsqueeze(0).repeat(1, 1, 1).to(device)
    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)
    mesh_glb_fpath = tempfile.NamedTemporaryFile(suffix=".glb", delete=False).name
    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras)
        mesh_out = model.extract_mesh(planes, use_texture_map=False, **infer_config)
        vertices, faces, vertex_colors = mesh_out
        vertices = vertices[:, [1, 2, 0]]
        save_glb(vertices, faces, vertex_colors, mesh_glb_fpath)
    return mesh_glb_fpath


def generate_3d_model(input_image, num_steps=75, seed=42, do_remove_bg=True):
    try:
        processed_image = preprocess(input_image, do_remove_bg)
        multi_view_img = generate_multiview_images(processed_image, sample_steps=num_steps, sample_seed=seed)
        mesh_glb_fpath = make3d(multi_view_img)
        return mesh_glb_fpath, "✅ 3D model generated successfully!", mesh_glb_fpath
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_pavilion(description, num_steps=1, seed=None):
    prompt = f"{description}, aerial view, photorealistic, high detail, architectural visualization, futuristic design, intricate details, elegant structure"
    negative_prompt = "blurry, distorted, deformed, text"
    generator = torch.manual_seed(seed) if seed is not None else None
    result = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_steps, guidance_scale=0.0, generator=generator, height=64, width=64).images[0]
    return result, f"✅ Image generated! Seed: {seed if seed else 'random'}"


def remove_background_fn(input_image):
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    output = remove(input_image)
    return output, "✅ Background removed!"


def process_model(model_file):
    try:
        blender_path = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
        script_path = os.path.join(os.path.dirname(__file__), "generate_drawings.py")
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "input_model.glb")
            shutil.copy(model_file, model_path)
            output_dir = os.path.join(temp_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            cmd = [
                blender_path, "--background", "--python", script_path, "--", model_path, output_dir
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Blender failed."
                raise RuntimeError(error_msg)
            time.sleep(1)
            images = [Image.open(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                raise RuntimeError("No render output.")
            return images
    except subprocess.TimeoutExpired:
        raise gr.Error("Processing timed out after 5 minutes.")
    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")


# Pavilion Examples
pavilion_examples = [
    ["Glass pavilion with bamboo structure, modern and minimalist design"],
    ["Biophilic pavilion with living green walls and exposed timber beams"],
    ["Futuristic pavilion with carbon fiber frame and integrated LED lighting"]
]

# Gradio app
with gr.Blocks(title="Pavilion Design Generator", theme=gr.themes.Glass()) as demo:
    gr.Markdown("<div style='text-align: center; font-size: 2em; font-weight: bold;'>🏛️ Pavilion 3D & 2D Design Generator</div>")
    with gr.Tabs():
        with gr.TabItem("🎨 Generate Pavilion"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(label="Description", placeholder="Describe your pavilion...", lines=4)
                    with gr.Accordion("Advanced Settings", open=False):
                        num_steps = gr.Slider(label="Generation Steps", minimum=1, maximum=10, value=1, step=1)
                        seed_input = gr.Number(label="Seed", value=1234, precision=0)
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
                    model_num_steps = gr.Slider(label="3D Generation Steps", minimum=5, maximum=75, value=75, step=1)
                    model_generate_btn = gr.Button("Generate 3D Model", variant="primary")
                with gr.Column():
                    model_output_3d = gr.Model3D(label="3D Model Preview")
                    model_status = gr.Textbox(label="Status", interactive=False)
                    model_download = gr.File(label="Download 3D Model (.glb)")
        with gr.TabItem("📝 2D Drawings Generator"):
            with gr.Row():
                with gr.Column():
                    drawing_input_model = gr.Model3D(label="Upload 3D Model (.glb, .obj, .fbx)")
                    drawing_btn = gr.Button("Generate Drawings", variant="primary")
                with gr.Column():
                    drawing_output_gallery = gr.Gallery(label="Generated Drawings", columns=3, preview=True)
    generate_btn.click(fn=generate_pavilion, inputs=[input_text, num_steps, seed_input], outputs=[output_image, error_output])
    bg_remove_btn.click(fn=remove_background_fn, inputs=bg_input_image, outputs=[bg_output_image, bg_status])
    # model_generate_btn.click(fn=generate_3d_model, inputs=[model_input_image, model_num_steps], outputs=[model_output_3d, model_status])
    model_generate_btn.click(
        fn=generate_3d_model,
        inputs=[model_input_image, model_num_steps],
        outputs=[model_output_3d, model_status, model_download]
    )
    drawing_btn.click(fn=process_model, inputs=drawing_input_model, outputs=drawing_output_gallery)
    output_image.change(fn=lambda img: (img, img), inputs=output_image, outputs=[bg_input_image, model_input_image])
    bg_output_image.change(fn=lambda img: img, inputs=bg_output_image, outputs=model_input_image)
    model_output_3d.change(fn=lambda f: f, inputs=model_output_3d, outputs=drawing_input_model)
    model_output_3d.change(
        fn=lambda file: file,
        inputs=model_output_3d,
        outputs=drawing_input_model
    )

if __name__ == "__main__":
    demo.launch(debug=True)
