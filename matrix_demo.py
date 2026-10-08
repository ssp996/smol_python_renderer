from matrix_tools import *

fov = 60
aspect_ratio = 1.7778
perspective_projection_matrix = perspective(math.radians(fov), aspect_ratio, 0.1, 100.0)
orthographic_matrix = directional_light_matrix([-0.4, -1.0, -0.3])
print(perspective_projection_matrix)
print(orthographic_matrix)
