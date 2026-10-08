from renderer import *
from matrix_tools import *
from object_primitives import *
from loader import model_loader
import trimesh


cat_trimesh = trimesh.load("loader/scene.gltf", force="mesh")
cat_vertices2, cat_normals2, cat_indices2 = expand_mesh(cat_trimesh)
cat_colors2 = np.tile(np.array(Color.BLUE.value, dtype=np.float32), (len(cat_vertices2), 1))

cat_object = Object(vertices=cat_vertices2, colors=cat_colors2, normals= cat_normals2, indices=cat_indices2, model_matrix=np.eye(4, dtype=np.float32))

sun2 = sun(intensity=.9)
renderer = Renderer(800, 800, [cat_object], lights=[sun2], default_sun=False)


def cat_spin():
    t = glfw.get_time()
    cat_scale = scale(5, 3 * abs(math.sin(7 * t)) + 2, 5)
    cat_rotate = rotate_y(7 * t)

    cat_model = cat_rotate @ cat_scale 
    
    renderer.objects[0].model_matrix = cat_model 

renderer.run_app(update_callback=cat_spin)