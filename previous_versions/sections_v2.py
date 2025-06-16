import gradio as gr
import os
import tempfile
import subprocess
from PIL import Image

def get_blender_path():
    """Find Blender executable"""
    paths = [
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
        "/usr/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender"
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Blender not found. Please install Blender.")

def process_model(model_file):
    """Process 3D model through Blender"""
    try:
        blender_path = get_blender_path()
        script_path = os.path.join(os.path.dirname(__file__), "generate_drawings.py")
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Blender script not found at {script_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            model_path = os.path.join(temp_dir, os.path.basename(model_file.name))
            with open(model_path, 'wb') as f:
                f.write(model_file.read())
            
            # Prepare output
            output_dir = os.path.join(temp_dir, "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            # Run Blender
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
                timeout=120
            )
            
            # Check results
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Blender failed with no output"
                raise RuntimeError(error_msg)
            
            # Get output images
            images = []
            for f in os.listdir(output_dir):
                if f.lower().endswith('.png'):
                    images.append(Image.open(os.path.join(output_dir, f)))
            
            if not images:
                raise RuntimeError("No output images were generated")
            
            return images
            
    except Exception as e:
        raise gr.Error(f"Processing failed: {str(e)}")

# Gradio Interface
with gr.Blocks(title="Architectural Drawing Generator") as demo:
    gr.Markdown("""
    # Architectural Drawing Generator
    Upload a 3D model (GLB/OBJ/FBX) to generate sections and plans
    """)
    
    with gr.Row():
        with gr.Column():
            model_input = gr.File(
                label="Upload 3D Model",
                file_types=[".glb", ".obj", ".fbx"]
            )
            process_btn = gr.Button("Generate Drawings", variant="primary")
            
        with gr.Column():
            gallery = gr.Gallery(
                label="Generated Drawings",
                columns=3
            )
    
    process_btn.click(
        fn=process_model,
        inputs=model_input,
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch()