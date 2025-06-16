import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import tempfile
import gradio as gr
from scipy.spatial import ConvexHull
import alphashape

def load_and_prepare_mesh(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    mesh.apply_translation(-mesh.centroid)
    scale = 2.0 / max(mesh.extents)
    mesh.apply_scale(scale)
    return mesh


def project_and_draw_silhouette(mesh, direction, view_name, alpha=0.02):
    rotation = trimesh.geometry.align_vectors(direction, [0, 0, 1])
    mesh_proj = mesh.copy()
    mesh_proj.apply_transform(rotation)

    vertices_2d = mesh_proj.vertices[:, :2]

    # Compute concave hull (alpha shape)
    shape = alphashape.alphashape(vertices_2d, alpha)
    fig, ax = plt.subplots(figsize=(6, 6))
    if shape.geom_type == 'Polygon':
        x, y = shape.exterior.xy
        ax.fill(x, y, color='black', alpha=1.0)
    elif shape.geom_type == 'MultiPolygon':
        for polygon in shape.geoms:
            x, y = polygon.exterior.xy
            ax.fill(x, y, color='black', alpha=1.0)

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


def generate_2d_silhouette_views(file_obj):
    mesh = load_and_prepare_mesh(file_obj.name)

    views = {
        "Front View": np.array([0, -1, 0]),
        "Left View": np.array([-1, 0, 0]),
        "Top View": np.array([0, 0, 1]),
        "Axonometric View": np.array([1, -1, 1]),
    }

    images = []
    for name, direction in views.items():
        img_path = project_and_draw_silhouette(mesh, direction, name)
        images.append(img_path)

    return images

demo = gr.Interface(
    fn=generate_2d_silhouette_views,
    inputs=gr.File(label="Upload 3D Model (.obj, .glb, .ply, etc.)"),
    outputs=[
        gr.Image(label="Front View Silhouette"),
        gr.Image(label="Left View Silhouette"),
        gr.Image(label="Top View Silhouette"),
        gr.Image(label="Axonometric View Silhouette"),
    ],
    title="2D Architectural Silhouette Views (Ultra-Fast!)",
    description="Upload a 3D model to generate ultra-fast 2D silhouette views (front, left, top, axonometric) using convex hull projection."
)

if __name__ == "__main__":
    demo.launch()


#gradio gradio_sections_v2.py