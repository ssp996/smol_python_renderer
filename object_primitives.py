import trimesh
from matrix_tools import directional_light_matrix, point_light_matrices
import numpy as np
from enum import Enum

class UniformType(Enum):
    # Scalars
    FLOAT = "float"
    INT = "int"
    UINT = "uint"
    BOOL = "bool"

    # Floating-point vectors
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"

    # Integer vectors
    IVEC2 = "ivec2"
    IVEC3 = "ivec3"
    IVEC4 = "ivec4"

    # Unsigned integer vectors
    UVEC2 = "uvec2"
    UVEC3 = "uvec3"
    UVEC4 = "uvec4"

    # Matrices
    MAT2 = "mat2"
    MAT3 = "mat3"
    MAT4 = "mat4"

    #Light type
    LIGHT_TYPE_POINT = "point_light"
    LIGHT_TYPE_DIRECTIONAL = "directional_light"

class UniformData:
    def __init__(self, data, type, name):
        self.data = data
        self.type = type
        self.name = name


class Object:
    def __init__(self, vertices, colors, normals, indices, model_matrix, uniform_data=[]):
        self.vertex_count = len(vertices)
        self.index_count = len(indices)

        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.colors = np.asarray(colors, dtype=np.float32)
        self.normals = np.asarray(normals, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.uint32)

        self.combined = np.empty((self.vertex_count, 9), dtype=np.float32)

        self.combined[:, 0:3] = self.vertices
        self.combined[:, 3:6] = self.colors
        self.combined[:, 6:9] = self.normals

        self.vao = None
        self.vbo = None
        self.ebo = None

        self.model_matrix = model_matrix #model matrix uniform
        self.model_loc = None

        self.uniform_data = uniform_data
        self.uniform_data.append(UniformData(model_matrix, UniformType.MAT4, "model"))
        self.uniform_locs = []

class Light:
    def __init__(self, pos_dir, color, intensity, type, light_matrices, far_plane=None): #enter far plane for point lights
        self.pos_dir = pos_dir
        self.color = color
        self.intensity = intensity
        self.type = type
        self.light_matrices = light_matrices
        self.far_plane = far_plane
        self.texture_unit_index = None

        self.fbo = None
        self.shadow_map = None

    def update_pos_dir(self, pos):
        self.pos_dir = pos
        if self.far_plane is not None:
            self.light_matrices = point_light_matrices(pos, self.far_plane)
        else:
            self.light_matrices = directional_light_matrix(pos)


class Color(Enum):
    BLACK       = [0.0, 0.0, 0.0]
    WHITE       = [1.0, 1.0, 1.0]
    RED         = [1.0, 0.0, 0.0]
    GREEN       = [0.0, 1.0, 0.0]
    BLUE        = [0.0, 0.0, 1.0]

    YELLOW      = [1.0, 1.0, 0.0]
    CYAN        = [0.0, 1.0, 1.0]
    MAGENTA     = [1.0, 0.0, 1.0]

    ORANGE      = [1.0, 0.5, 0.0]
    PURPLE      = [0.5, 0.0, 0.5]
    PINK        = [1.0, 0.75, 0.8]
    BROWN       = [0.6, 0.3, 0.1]

    GRAY        = [0.5, 0.5, 0.5]
    LIGHT_GRAY  = [0.75, 0.75, 0.75]
    DARK_GRAY   = [0.25, 0.25, 0.25]

    LIME        = [0.5, 1.0, 0.0]
    NAVY        = [0.0, 0.0, 0.5]
    TEAL        = [0.0, 0.5, 0.5]
    OLIVE       = [0.5, 0.5, 0.0]
    MAROON      = [0.5, 0.0, 0.0]

    GOLD        = [1.0, 0.84, 0.0]
    SILVER      = [0.75, 0.75, 0.75]
    CORAL       = [1.0, 0.5, 0.31]
    SALMON      = [1.0, 0.5, 0.5]
    VIOLET      = [0.56, 0.0, 1.0]
    INDIGO      = [0.29, 0.0, 0.51]

    SKY_BLUE    = [0.53, 0.81, 0.92]
    LIGHT_BLUE  = [0.68, 0.85, 0.90]
    DARK_BLUE   = [0.0, 0.0, 0.55]

    LIGHT_GREEN = [0.56, 0.93, 0.56]
    DARK_GREEN  = [0.0, 0.39, 0.0]

    BEIGE       = [0.76, 0.70, 0.50]
    CREAM       = [1.0, 0.99, 0.82]

""" def expand_mesh(mesh):
    new_vertices = []
    new_normals = []
    new_indices = []

    vertex_map = {}

    for face_index, face in enumerate(mesh.faces):
        normal = mesh.face_normals[face_index]

        triangle = []

        for vertex_index in face:
            position = mesh.vertices[vertex_index]

            key = (tuple(position), tuple(normal))

            if key not in vertex_map:
                vertex_map[key] = len(new_vertices)

                new_vertices.append(position)
                new_normals.append(normal)

            triangle.append(vertex_map[key])

        new_indices.append(triangle)

    return (np.array(new_vertices, dtype=np.float32), np.array(new_normals, dtype=np.float32), np.array(new_indices, dtype=np.float32).flatten()) """

def expand_mesh(mesh):
    v_expanded = mesh.vertices[mesh.faces].reshape(-1, 3)
    
    n_expanded = np.repeat(mesh.face_normals, 3, axis=0)
    
    combined = np.hstack((v_expanded, n_expanded))
    
    unique_combined, new_indices = np.unique(combined, axis=0, return_inverse=True)
    
    new_vertices = unique_combined[:, 0:3].astype(np.float32)
    new_normals = unique_combined[:, 3:6].astype(np.float32)
    
    new_indices = new_indices.astype(np.float32)
    
    return new_vertices, new_normals, new_indices

def expand_mesh_smooth(mesh: trimesh.Trimesh):
    mesh = mesh.smooth_shaded
    return (np.asarray(mesh.vertices, np.float32), np.asarray(mesh.vertex_normals, np.float32), np.asarray(mesh.faces, np.uint32).ravel())


def cube_object(color, model_matrix=None, extents=[1, 1, 1]):
    TRIMESH_BOX = trimesh.creation.box(extents=extents)
    BOX_VERTICES, BOX_NORMALS, BOX_INDICES = expand_mesh(TRIMESH_BOX)
    box_colors = np.tile(np.array(color.value, dtype=np.float32), (len(BOX_VERTICES), 1))

    cube_model_matrix = model_matrix if model_matrix is not None else np.eye(4, dtype=np.float32)
    cube = Object(vertices=BOX_VERTICES, colors=box_colors, normals=BOX_NORMALS, indices=BOX_INDICES, model_matrix=cube_model_matrix, uniform_data=[UniformData(cube_model_matrix, UniformType.MAT4, "model")])
    return cube

def cone_object(color: Color, model_matrix=None, radius=1.0, height=2.0, sections=32, shade_smooth=False):
    trimesh_cone = trimesh.creation.cone(radius=radius, height=height, sections=sections)
    cone_vertices, cone_normals, cone_indices = expand_mesh_smooth(mesh=trimesh_cone) if shade_smooth else expand_mesh(mesh=trimesh_cone)
    colors = np.tile(np.array(color.value, dtype=np.float32), (len(cone_vertices), 1))

    cone_model_matrix = model_matrix if model_matrix is not None else np.eye(4, dtype=np.float32)
    cone = Object(vertices=cone_vertices, colors=colors, normals=cone_normals, indices=cone_indices, model_matrix=cone_model_matrix, uniform_data=[UniformData(cone_model_matrix, UniformType.MAT4, "model")])    
    return cone

def cylinder_object(color: Color, model_matrix=None, radius=1.0, height=2.0, sections=32, shade_smooth=False):
    trimesh_cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    vertices, normals, indices = expand_mesh_smooth(mesh=trimesh_cylinder) if shade_smooth else expand_mesh(mesh=trimesh_cylinder)
    colors = np.tile(np.array(color.value, dtype=np.float32), (len(vertices), 1))

    cylinder_model_matrix = model_matrix if model_matrix is not None else np.eye(4, dtype=np.float32)
    return Object(vertices=vertices, colors=colors, normals=normals, indices=indices, model_matrix=cylinder_model_matrix, uniform_data=[UniformData(cylinder_model_matrix, UniformType.MAT4, "model")])

def icosphere_object(color: Color, model_matrix=None, radius=1.0, subdivisions=3, shade_smooth=True):
    trimesh_sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    vertices, normals, indices = expand_mesh_smooth(mesh=trimesh_sphere) if shade_smooth else expand_mesh(mesh=trimesh_sphere)
    colors = np.tile(np.array(color.value, dtype=np.float32), (len(vertices), 1))

    sphere_model_matrix = model_matrix if model_matrix is not None else np.eye(4, dtype=np.float32)
    return Object(vertices=vertices, colors=colors, normals=normals, indices=indices, model_matrix=sphere_model_matrix, uniform_data=[UniformData(sphere_model_matrix, UniformType.MAT4, "model")])

def uv_sphere_object(color: Color, model_matrix=None, radius=1.0, count=(32, 32), shade_smooth=True):
    trimesh_sphere = trimesh.creation.uv_sphere(radius=radius, count=count)
    vertices, normals, indices = expand_mesh_smooth(mesh=trimesh_sphere) if shade_smooth else expand_mesh(mesh=trimesh_sphere)
    colors = np.tile(np.array(color.value, dtype=np.float32), (len(vertices), 1))

    sphere_model_matrix = model_matrix if model_matrix is not None else np.eye(4, dtype=np.float32)
    return Object(vertices=vertices, colors=colors, normals=normals, indices=indices, model_matrix=sphere_model_matrix, uniform_data=[UniformData(sphere_model_matrix, UniformType.MAT4, "model")])

def capsule_object(color: Color, model_matrix=None, height=2.0, radius=1.0, count=(32, 32), shade_smooth=True):
    trimesh_capsule = trimesh.creation.capsule(height=height, radius=radius, count=count)
    vertices, normals, indices = expand_mesh_smooth(mesh=trimesh_capsule) if shade_smooth else expand_mesh(mesh=trimesh_capsule)
    colors = np.tile(np.array(color.value, dtype=np.float32), (len(vertices), 1))

    capsule_model_matrix = model_matrix if model_matrix is not None else np.eye(4, dtype=np.float32)
    return Object(vertices=vertices, colors=colors, normals=normals, indices=indices, model_matrix=capsule_model_matrix, uniform_data=[UniformData(capsule_model_matrix, UniformType.MAT4, "model")])

def sun(direction=None, color=None, intensity=0.4):
    direction = np.array([-0.4, -1.0, -0.3] if direction is None else direction, dtype=np.float32)
    color = np.array([0.9, 0.9, 1.0] if color is None else color, dtype=np.float32)
    return Light(pos_dir=direction, color=color, intensity=intensity, type=UniformType.LIGHT_TYPE_DIRECTIONAL, light_matrices=directional_light_matrix(direction))

def bulb(position, color=None, intensity=2.0):
    POINT_FAR = 30.0
    bulb_color = color if color is not None else Color.ORANGE.value
    bulb_pos = np.array(position, dtype=np.float32)
    lightbulb = Light(pos_dir=bulb_pos, color=np.array(bulb_color, dtype=np.float32), intensity=intensity, type=UniformType.LIGHT_TYPE_POINT, light_matrices=point_light_matrices(bulb_pos, POINT_FAR), far_plane=POINT_FAR)
    return lightbulb
