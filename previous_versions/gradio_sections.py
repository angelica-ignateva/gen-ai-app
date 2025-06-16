import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import tempfile
import gradio as gr

def load_and_prepare_mesh(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    mesh.apply_translation(-mesh.centroid)
    scale = 2.0 / max(mesh.extents)
    mesh.apply_scale(scale)
    return mesh

def project_and_draw_edges(mesh, direction, view_name):
    # Rotate mesh to align view direction with Z axis
    rotation = trimesh.geometry.align_vectors(direction, [0, 0, 1])
    mesh_proj = mesh.copy()
    mesh_proj.apply_transform(rotation)

    # 2D projection: drop Z
    vertices_2d = mesh_proj.vertices[:, :2]
    edges = mesh_proj.edges_unique

    fig, ax = plt.subplots(figsize=(6, 6))
    for edge in edges:
        p1, p2 = vertices_2d[edge[0]], vertices_2d[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=0.5)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(view_name)
    plt.tight_layout()

    tmp_dir = tempfile.gettempdir()
    output_path = os.path.join(tmp_dir, f"{view_name}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

def generate_2d_views(file_obj):
    mesh = load_and_prepare_mesh(file_obj.name)

    views = {
        "Front View": np.array([0, -1, 0]),
        "Left View": np.array([-1, 0, 0]),
        "Top View": np.array([0, 0, 1]),
        "Axonometric View": np.array([1, -1, 1]),
    }

    images = []
    for name, direction in views.items():
        img_path = project_and_draw_edges(mesh, direction, name)
        images.append(img_path)

    return images

demo = gr.Interface(
    fn=generate_2d_views,
    inputs=gr.File(label="Upload 3D Model (.obj, .glb, .ply, etc.)"),
    outputs=[
        gr.Image(label="Front View Drawing"),
        gr.Image(label="Left View Drawing"),
        gr.Image(label="Top View Drawing"),
        gr.Image(label="Axonometric View Drawing"),
    ],
    title="2D Architectural Drawings (Direct Projection)",
    description="Upload a 3D model to generate fast 2D architectural line drawings for front, left, top, and axonometric views."
)

if __name__ == "__main__":
    demo.launch()
