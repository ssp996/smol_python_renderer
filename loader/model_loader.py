import trimesh 
import numpy as np

def load_model_as_mesh(file_path, default_color=(0.502, 0.502, 0.502)):
    mesh = trimesh.load(file_path, force="mesh")
    
    vertices = np.array(mesh.vertices)
    indices = np.array(mesh.faces)
    normals = np.array(mesh.vertex_normals)
    
    try:
        colors = np.array(mesh.visual.vertex_colors)
        if len(colors) != len(vertices):
            raise ValueError("Color array size mismatch")
    except (AttributeError, ValueError, TypeError):
        colors = np.tile(default_color, (len(vertices), 1)).astype(np.uint8)

    return vertices, colors, normals, indices

