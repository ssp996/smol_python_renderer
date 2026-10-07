import trimesh 
import numpy as np

def load_as_mesh(filepath):
    mesh = trimesh.load(filepath, force='mesh')

    vertices = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    indices = mesh.faces.flatten().astype(np.uint32)
    
    colors = mesh.visual.vertex_colors[:, :3] / 255.0
    colors = colors.astype(np.float32)

    return vertices, colors, normals, indices



