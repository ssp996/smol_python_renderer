from renderer import *
from object_primitives import *
from matrix_tools import *

cone = cone_object(Color.RED)
cube = cube_object(Color.BLUE, translate(0, -3, 0), [10, 1, 10])
icosphere = icosphere_object(Color.CORAL, translate(-4, 0, 0), shade_smooth=True)

renderer = Renderer(800, 800, [cone, cube, icosphere])

def update_callback():
    t = glfw.get_time()
    renderer.objects[0].model_matrix = rotate_x(0.5 * t)

renderer.run_app(update_callback=update_callback)

