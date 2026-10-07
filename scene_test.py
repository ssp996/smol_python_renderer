from renderer import *
from object_primitives import *

cone = cone_object(Color.RED, shade_smooth=True)

renderer = Renderer(800, 800, [cone])

renderer.run_app()

