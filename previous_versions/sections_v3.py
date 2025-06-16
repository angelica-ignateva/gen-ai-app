import gradio as gr
import torch
import numpy as np
from diffusers import DiffusionPipeline
from rembg import remove
import tempfile
import os
from PIL import Image, ImageDraw
import shutil
import subprocess
import time
import io

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Initialize pipelines
pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
hy3d_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')

def generate_pavilion(description, num_steps=1, seed=None):
    try:
        prompt = f"{description}, aerial view, photorealistic, high detail, architectural visualization, futuristic design, intricate details, elegant structure"
        negative_prompt = "blurry, distorted, deformed, text"

        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0,
            generator=generator
        ).images[0]

        return result, f"✅ Image generated successfully! Seed: {seed if seed is not None else 'random'}"
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

        preview_image = Image.new("RGB", (500, 500), color=(100, 100, 100))
        draw = ImageDraw.Draw(preview_image)
        draw.text((50, 250), "3D Model Generated", fill="white")

        with open("generated_model_url.txt", "w") as f:
            f.write(output_3d_path)

        return preview_image, f"✅ 3D model generated successfully with {num_steps} steps!"
    except Exception as e:
        return None, f"❌ Error generating 3D model: {str(e)}"

def load_3d_model_viewer():
    try:
        with open("generated_model_url.txt", "r") as f:
            model_url = f.read().strip()
        html_content = f"""
        <model-viewer src="file://{model_url}" alt="3D Model" auto-rotate camera-controls style="width: 100%; height: 500px;"></model-viewer>
        """
        return html_content
    except Exception as e:
        return f"<p style='color: red;'>❌ Error loading 3D model: {str(e)}</p>"

# Blender-based 2D drawings generator
def get_blender_path():
    blender_path = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
    if os.path.exists(blender_path):
        return blender_path
    raise FileNotFoundError("Blender not found. Please install Blender 4.4 at the default location.")

def process_model(model_file):
    try:
        blender_path = get_blender_path()
        script_path = os.path.join(os.path.dirname(__file__), "generate_drawings.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Blender script not found at {script_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Handle both file path strings and file-like objects
            if isinstance(model_file, str):
                model_path = os.path.join(temp_dir, "input_model" + os.path.splitext(model_file)[1])
                shutil.copy(model_file, model_path)
            else:
                # Handle file-like objects (gradio might pass this)
                model_path = os.path.join(temp_dir, "input_model.glb")  # Default to .glb extension
                if hasattr(model_file, 'name'):
                    model_path = os.path.join(temp_dir, "input_model" + os.path.splitext(model_file.name)[1])
                with open(model_path, 'wb') as f:
                    if hasattr(model_file, 'read'):
                        f.write(model_file.read())
                    else:
                        shutil.copyfileobj(model_file, f)
            
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
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Blender failed with no output"
                raise RuntimeError(error_msg)
            time.sleep(1)
            images = []
            for f in os.listdir(output_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(output_dir, f)
                    with Image.open(img_path) as img:
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        images.append(img_bytes.getvalue())
            if not images:
                raise RuntimeError("No render output created.")
            return [Image.open(io.BytesIO(img_data)) for img_data in images]
    except subprocess.TimeoutExpired:
        raise gr.Error("Processing timed out after 5 minutes. The model might be too complex.")
    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")

# Pavilion Examples
pavilion_examples = [
    ["Glass pavilion with bamboo structure, modern and minimalist design"],
    ["Biophilic pavilion with living green walls and exposed timber beams"],
    ["Futuristic pavilion with carbon fiber frame and integrated LED lighting"]
]

# Main Gradio UI
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
                    model_num_steps = gr.Slider(label="3D Generation Steps", minimum=1, maximum=10, value=1, step=1)
                    model_generate_btn = gr.Button("Generate 3D Model", variant="primary")
                with gr.Column():
                    model_output_image = gr.Image(label="3D Model Preview", type="pil")
                    model_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.TabItem("📝 2D Drawings Generator"):
            with gr.Row():
                with gr.Column():
                    drawing_input_model = gr.Model3D(label="Upload 3D Model (.glb, .obj, .fbx)")
                    drawing_btn = gr.Button("Generate Drawings", variant="primary")
                with gr.Column():
                    drawing_output_gallery = gr.Gallery(label="Generated Drawings", columns=3, preview=True)
    
    # Button actions
    generate_btn.click(fn=generate_pavilion, inputs=[input_text, num_steps, seed_input], outputs=[output_image, error_output])
    bg_remove_btn.click(fn=remove_background, inputs=bg_input_image, outputs=[bg_output_image, bg_status])
    model_generate_btn.click(fn=generate_3d_model, inputs=[model_input_image, model_num_steps], outputs=[model_output_image, model_status])
    drawing_btn.click(fn=process_model, inputs=drawing_input_model, outputs=drawing_output_gallery)
    
    # Auto-fill background and 3D model with generated 2D image
    output_image.change(fn=lambda x: x, inputs=output_image, outputs=[bg_input_image, model_input_image])

if __name__ == "__main__":
    demo.launch(debug=True)
