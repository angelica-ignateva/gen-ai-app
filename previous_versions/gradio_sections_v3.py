import os
import numpy as np
import pyvista as pv
import tempfile
import gradio as gr
from PIL import Image

def load_and_prepare_mesh(file_path):
    mesh = pv.read(file_path)
    mesh.translate(-np.array(mesh.center))
    scale = 2.0 / mesh.length
    mesh.scale([scale, scale, scale])
    return mesh

def render_orthographic_view(mesh, view, view_name):
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512))

    # Add the mesh with visible edges
    plotter.add_mesh(mesh, color='black', show_edges=True, style='wireframe', line_width=0.5)

    # Set the camera to orthographic projection
    plotter.camera.SetParallelProjection(True)

    # Set the view direction
    if view == "top":
        plotter.view_vector((0, -1, 0))  # Top view
    elif view == "left":
        plotter.view_vector((-1, 0, 0))  # Left view
    elif view == "front":
        plotter.view_vector((0, 0, 1))   # Front view
    elif view == "axonometric":
        plotter.view_vector((5, -1, 0.5))   # Axonometry view

    # Set background to white
    plotter.set_background('white')

    # Save the screenshot
    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"{view_name}.png")
    plotter.screenshot(output_path)
    plotter.close()
    return output_path

def rotate_image(image_path, angle, output_path):
    img = Image.open(image_path)
    rotated = img.rotate(angle, expand=True, fillcolor='white')
    rotated.save(output_path)

def generate_2d_views(file_obj):
    mesh = load_and_prepare_mesh(file_obj.name)

    # Create view images
    top_img = render_orthographic_view(mesh, "top", "Top View")
    left_img = render_orthographic_view(mesh, "left", "Left View")
    front_img = render_orthographic_view(mesh, "front", "Front View")
    axo_img = render_orthographic_view(mesh, "axonometric", "Axonometric View")

    # Apply the requested rotations (no more stacking, just final rotations)
    tmp_dir = tempfile.gettempdir()

    left_final = os.path.join(tmp_dir, "Left_View_Final.png")
    rotate_image(left_img, -90, left_final)

    front_final = os.path.join(tmp_dir, "Front_View_Final.png")
    rotate_image(front_img, 90, front_final)

    axo_final = os.path.join(tmp_dir, "Axonometric_View_Final.png")
    rotate_image(axo_img, 90, axo_final)

    images = [
        top_img,       # Top View
        left_final,    # Left View rotated 180
        front_final,   # Front View rotated 180
        axo_final      # Axonometric View rotated 180
    ]
    return images

demo = gr.Interface(
    fn=generate_2d_views,
    inputs=gr.File(label="Upload 3D Model (.obj, .ply, .stl, .glb, etc.)"),
    outputs=[
        gr.Image(label="Top View Drawing"),
        gr.Image(label="Left View Drawing"),
        gr.Image(label="Front View Drawing"),
        gr.Image(label="Axonometric View Drawing"),
    ],
    title="2D Architectural Wireframe Views Generator",
    description="Upload a 3D model to generate 2D architectural wireframe views (Top, Left, Front, Axonometric)"
)

if __name__ == "__main__":
    demo.launch()

#gradio gradio_sections_v3.py