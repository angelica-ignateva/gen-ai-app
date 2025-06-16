import gradio as gr
import torch
import numpy as np
import os
import tempfile
from PIL import Image
from rembg import remove
from diffusers import DiffusionPipeline
from diffusers import DiffusionPipeline
from diffusers import FluxPipeline
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
import trimesh
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import ezdxf
import io
import zipfile


import sys
sys.path.append(os.path.abspath("InstantMesh"))

# Initialize SDXL for 2D generation
# Initialize diffusion pipelines
pipe_sdxl_turbo = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
pipe_sdxl_turbo = pipe_sdxl_turbo.to("cuda")

pipe_sdxl = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe_sdxl = pipe_sdxl.to("cuda")

# Don't works locally from my side because of missing sentencepiece, 
# tried to install it but it doesn't support python 3.13 so used the SDXL pipeline as a placeholder to test locally
# for Gradio demo, we will use the real Flux pipeline
# pipe_flux = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")
# pipe_flux = pipe_flux.to("cuda")
pipe_flux = pipe_sdxl

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
        return None, f"❌ Error: {str(e)}", None

def generate_pavilion(description, num_steps=1, seed=None, model_choice="SDXL-Turbo"):
    try:
        prompt = f"{description}, aerial view, photorealistic, high detail, architectural visualization, futuristic design, intricate details, elegant structure"
        negative_prompt = "blurry, distorted, deformed, text"

        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        # Choose pipeline
        if model_choice == "SDXL":
            chosen_pipe = pipe_sdxl
        elif model_choice == "Flux":
            chosen_pipe = pipe_flux
        else:
            chosen_pipe = pipe_sdxl_turbo

        result = chosen_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=generator
        ).images[0]

        return result, f"✅ Generated successfully! Seed: {seed if seed is not None else 'random'}"
    except Exception as e:
        return None, f"❌ Error generating image: {str(e)}"
    
def remove_background_fn(input_image):
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    output = remove(input_image)
    return output, "✅ Background removed!"


def look_at_matrix(eye, target, up=[0, 1, 0]):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = (target - eye)
    forward /= np.linalg.norm(forward)

    side = np.cross(forward, up)
    side /= np.linalg.norm(side)

    true_up = np.cross(side, forward)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = side
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[:3, 3] = -eye @ m[:3, :3]
    return m

def generate_2d_drawings(model_file):
    try:
        mesh = trimesh.load(model_file)

        # Get first mesh if input is a scene
        if isinstance(mesh, trimesh.Scene):
            mesh = list(mesh.geometry.values())[0]

        # Create temp folders
        temp_dir = tempfile.mkdtemp()
        img_temp_dir = os.path.join(temp_dir, "images")
        dxf_temp_dir = os.path.join(temp_dir, "dxf")
        os.makedirs(img_temp_dir, exist_ok=True)
        os.makedirs(dxf_temp_dir, exist_ok=True)

        # Define camera transforms
        view_transforms = {
            'top': look_at_matrix([0, 0, 2], [0, 0, 0], up=[0, 1, 0]),
            'front': look_at_matrix([0, -2, 0], [0, 0, 0], up=[0, 0, 1]),
            'left': look_at_matrix([-2, 0, 0], [0, 0, 0], up=[0, 0, 1]),
            'axonometric': look_at_matrix([2, 2, 2], [0, 0, 0], up=[0, 0, 1])
        }


        images = []
        dxf_paths = []

        for view_name, transform in view_transforms.items():
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_aspect('equal')
            ax.axis('off')

            # Copy and transform mesh
            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(transform)

            # Project 3D edges to 2D
            edges = mesh_copy.edges_unique
            segments = mesh_copy.vertices[edges]

            for seg in segments:
                points = seg[:, :2]
                ax.plot(points[:, 0], points[:, 1], 'k-', linewidth=0.3)

            ax.autoscale_view()

            # Save image
            img_path = os.path.join(img_temp_dir, f"{view_name}.png")
            fig.savefig(img_path, bbox_inches='tight', pad_inches=0.1, dpi=300000)
            with open(img_path, 'rb') as f:
                images.append(Image.open(io.BytesIO(f.read())))

            plt.close(fig)

            # Create DXF
            dxf_path = os.path.join(dxf_temp_dir, f"{view_name}.dxf")
            doc = ezdxf.new()
            msp = doc.modelspace()

            for seg in segments:
                pts_2d = seg[:, :2].tolist()
                msp.add_lwpolyline(pts_2d)

            doc.saveas(dxf_path)
            dxf_paths.append(dxf_path)

        # Zip DXF files
        zip_path = os.path.join(temp_dir, "drawings.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for dxf_path in dxf_paths:
                zipf.write(dxf_path, os.path.basename(dxf_path))

        return images, zip_path

    except Exception as e:
        raise gr.Error(f"Failed to generate 2D drawings: {str(e)}")


# Pavilion Examples
pavilion_examples = [
    ["Glass pavilion with bamboo structure, modern and minimalist design, situated over a calm water pond, reflecting natural light and creating a serene environment"],
    ["Biophilic pavilion with living green walls and exposed timber beams, blending nature with architecture in a peaceful garden setting"],
    ["Futuristic pavilion with carbon fiber frame, geometric shapes, and integrated LED lighting that glows at night, located in a contemporary city park"],
    ["Wooden pavilion with an undulating wave-like roof, featuring open seating areas surrounded by natural greenery, inviting visitors to relax and connect with nature"],
    ["Metal pavilion with adjustable sun shades and simple geometric shapes, designed for an urban park setting, providing a shaded and comfortable public space"],
    ["Desert pavilion with lightweight canvas covers and steel frames, using an earthy color palette to harmonize with the sandy surroundings"],
    ["Ice pavilion with crystal-clear glass panels and softly glowing lights, set in a snowy landscape to create a magical, wintry atmosphere"],
    ["High-tech pavilion with transparent OLED walls, modern minimalist furniture, and clean lines, designed as an innovative meeting space in a futuristic city plaza"],
    ["Origami-inspired pavilion with precisely folded metal panels, creating a sculptural form with an intricate interplay of shadows and light"]
]

# Gradio app
with gr.Blocks(title="Pavilion Design Generator", theme=gr.themes.Glass()) as demo:
    gr.Markdown("<div style='text-align: center; font-size: 2em; font-weight: bold;'>🏛️ Pavilion 3D & 2D Design Generator</div>")
    with gr.Tabs():
        with gr.TabItem("🎨 Generate Pavilion"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        label="Choose Model",
                        choices=["SDXL-Turbo", "SDXL", "Flux"],
                        value="SDXL-Turbo"
                    )
                    input_text = gr.Textbox(label="Description", placeholder="Describe your pavilion...", lines=4)
                    with gr.Accordion("Advanced Settings", open=False):
                        num_steps = gr.Slider(label="Generation Steps", minimum=1, maximum=50, value=1, step=1)
                        seed_input = gr.Number(label="Seed", value=0, precision=0)
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
                    model_num_steps = gr.Slider(label="3D Generation Steps", minimum=1, maximum=75, value=5, step=1)
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
                    drawing_output_gallery = gr.Gallery(label="Generated Drawings", columns=2, preview=True)
                    drawing_download = gr.File(label="Download DXF Files (ZIP)")
    
    # Connect all the components
    generate_btn.click(fn=generate_pavilion, inputs=[input_text, num_steps, seed_input, model_choice], outputs=[output_image, error_output])
    bg_remove_btn.click(fn=remove_background_fn, inputs=bg_input_image, outputs=[bg_output_image, bg_status])
    model_generate_btn.click(
        fn=generate_3d_model,
        inputs=[model_input_image, model_num_steps],
        outputs=[model_output_3d, model_status, model_download]
    )
    drawing_btn.click(
        fn=generate_2d_drawings,
        inputs=drawing_input_model,
        outputs=[drawing_output_gallery, drawing_download]
    )
    output_image.change(fn=lambda img: (img, img), inputs=output_image, outputs=[bg_input_image, model_input_image])
    bg_output_image.change(fn=lambda img: img, inputs=bg_output_image, outputs=model_input_image)
    model_output_3d.change(fn=lambda f: f, inputs=model_output_3d, outputs=drawing_input_model)

if __name__ == "__main__":
    demo.launch(debug=True)