import gradio as gr
import os
import tempfile
import rhino3dm
import compute_rhino3d.Util
import compute_rhino3d.Brep
import shutil
import subprocess

def get_compute_server():
    """
    Configure the Rhino Compute server.
    Ensure your Compute server is running locally or remotely.
    """
    compute_rhino3d.Util.url = "http://localhost:8081/"
    # compute_rhino3d.Util.apiKey = ""  # If needed

    # compute_rhino3d.Util.url = "https://compute-server.iaac.net/"
    # compute_rhino3d.Util.apiKey = "datamgmt20242"

def load_model(model_path):
    """
    Load a 3D model (3dm format) using rhino3dm.
    """
    model = rhino3dm.File3dm.Read(model_path)
    if not model:
        raise Exception("Failed to load model.")
    return model

def create_section_planes(bbox):
    """
    Create planes for X, Y, Z sections.
    """
    center = bbox.Center
    planes = [
        rhino3dm.Plane(center, rhino3dm.Vector3d(1, 0, 0)),
        rhino3dm.Plane(center, rhino3dm.Vector3d(0, 1, 0)),
        rhino3dm.Plane(center, rhino3dm.Vector3d(0, 0, 1)),
    ]
    return planes

def generate_sections(model, output_dir):
    """
    Generate section curves for X, Y, Z planes.
    Save them as new 3dm files.
    """
    bbox = model.Objects.GetBoundingBox(True)
    planes = create_section_planes(bbox)
    
    section_files = []
    for i, plane in enumerate(planes):
        section_curves = compute_rhino3d.Brep.Section(model.Objects, plane.Origin, plane.Normal, 0.001)
        
        section_model = rhino3dm.File3dm()
        for curve in section_curves:
            section_model.Objects.AddCurve(curve)
        
        section_path = os.path.join(output_dir, f"Section_{i}.3dm")
        section_model.Write(section_path)
        section_files.append(section_path)
    
    return section_files

def convert_to_3dm(input_file, output_file):
    """
    Converts input model file to .3dm using Rhino's headless mode.
    """
    rhino_exe = r"C:\Program Files\Rhino 8\System\Rhino.exe"  # Update path if needed
    conversion_script = os.path.join(os.path.dirname(__file__), "convert_to_3dm.py")
    
    if not os.path.exists(rhino_exe):
        raise RuntimeError("Rhino executable not found. Please ensure Rhino is installed.")
    
    if not os.path.exists(conversion_script):
        raise RuntimeError("Conversion script not found.")
    
    cmd = [
        rhino_exe,
        "/nosplash",
        "/runscript=!-_RunPythonScript",
        conversion_script,
        input_file,
        output_file
    ]
    
    print("Running conversion command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    print("Conversion stdout:", result.stdout)
    print("Conversion stderr:", result.stderr)
    
    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr}")
    
    return output_file

def process_model(model_file):
    """
    Process uploaded model (any format) by converting to .3dm and generating sections.
    """
    try:
        get_compute_server()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Determine file extension
            input_ext = os.path.splitext(model_file)[1].lower()
            input_path = os.path.join(temp_dir, "input_model" + input_ext)
            shutil.copy(model_file, input_path)
            
            if input_ext != ".3dm":
                # Convert to .3dm
                converted_3dm = os.path.join(temp_dir, "converted_model.3dm")
                convert_to_3dm(input_path, converted_3dm)
                model_path = converted_3dm
            else:
                model_path = input_path
            
            model = load_model(model_path)
            if not model:
                raise Exception("Failed to load .3dm file after conversion.")
            
            output_dir = os.path.join(temp_dir, "sections")
            os.makedirs(output_dir, exist_ok=True)
            
            section_files = generate_sections(model, output_dir)
            return [gr.File(section_file) for section_file in section_files]
    
    except subprocess.TimeoutExpired:
        raise gr.Error("Conversion timed out.")
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


# Gradio Interface
with gr.Blocks(title="Rhino 3D Section Generator") as demo:
    gr.Markdown("""
    # Rhino 3D Section Generator
    Upload a `.3dm` file to generate X, Y, Z sections (as new 3dm files with curves).
    """)
    
    with gr.Row():
        with gr.Column():
            model_input = gr.File(label="Upload 3dm Model")
            process_btn = gr.Button("Generate Sections", variant="primary")
        
        with gr.Column():
            section_gallery = gr.Files(label="Downloadable Section Files")
    
    process_btn.click(
        fn=process_model,
        inputs=model_input,
        outputs=section_gallery
    )

if __name__ == "__main__":
    demo.launch(debug=True)
