from renderer import *
from object_primitives import *
from matrix_tools import *

cone = cone_object(Color.RED, translate(-3, 0, 0))
cube = cube_object(Color.WHITE, translate(0, -3, 0), [100, 1, 100])
icosphere = icosphere_object(Color.CORAL, translate(3, 0, 0), shade_smooth=True)
uv_sphere = uv_sphere_object(Color.CYAN, translate(0, 11, 0), radius=0.1)

lightbulb = bulb([0, 10, 0])
renderer = Renderer(1920, 1080, [cone, cube, icosphere, uv_sphere], lights=[lightbulb], default_sun=False)

def update_callback():
    t = glfw.get_time()
    renderer.objects[0].model_matrix = rotate_x(0.5 * t) @ translate(-8, 0, 0)
    lightbulb_pos = [15 * math.sin(t), 6, 15 * math.cos(t)]
    renderer.objects[3].model_matrix = translate(lightbulb_pos[0], lightbulb_pos[1] + 1, lightbulb_pos[2])
    renderer.point_lights[0].update_position(lightbulb_pos)
    

renderer.run_app(update_callback=update_callback)

