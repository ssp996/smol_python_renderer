from OpenGL.GL import *
from OpenGL.GL import shaders
import glfw
import zlib
import numpy as np
import math
from enum import Enum
import ctypes
from matrix_tools import *
from camera import Camera
from object_primitives import *

UNIFORM_UPLOADERS = {
    # Scalars
    UniformType.FLOAT: lambda loc, x:
        glUniform1f(loc, x),

    UniformType.INT: lambda loc, x:
        glUniform1i(loc, x),

    UniformType.UINT: lambda loc, x:
        glUniform1ui(loc, x),

    UniformType.BOOL: lambda loc, x:
        glUniform1i(loc, x),

    # Floating-point vectors
    UniformType.VEC2: lambda loc, x:
        glUniform2f(loc, *x),

    UniformType.VEC3: lambda loc, x:
        glUniform3f(loc, *x),

    UniformType.VEC4: lambda loc, x:
        glUniform4f(loc, *x),

    # Integer vectors
    UniformType.IVEC2: lambda loc, x:
        glUniform2i(loc, *x),

    UniformType.IVEC3: lambda loc, x:
        glUniform3i(loc, *x),

    UniformType.IVEC4: lambda loc, x:
        glUniform4i(loc, *x),

    # Unsigned integer vectors
    UniformType.UVEC2: lambda loc, x:
        glUniform2ui(loc, *x),

    UniformType.UVEC3: lambda loc, x:
        glUniform3ui(loc, *x),

    UniformType.UVEC4: lambda loc, x:
        glUniform4ui(loc, *x),

    # Matrices
    UniformType.MAT2: lambda loc, x:
        glUniformMatrix2fv(loc, 1, GL_TRUE, x),

    UniformType.MAT3: lambda loc, x:
        glUniformMatrix3fv(loc, 1, GL_TRUE, x),

    UniformType.MAT4: lambda loc, x:
        glUniformMatrix4fv(loc, 1, GL_TRUE, x),
}

POINT_FACE_NAMES = ["px", "nx", "py", "ny", "pz", "nz"]

POINT_FACE_DISPLAY = [
    ((1, 0, 0),  (0, 1, 0),  (0, 1)),   # +X
    ((-1, 0, 0), (0, 1, 0),  (2, 1)),   # -X
    ((0, 1, 0),  (0, 0, -1), (1, 0)),   # +Y
    ((0, -1, 0), (0, 0, 1),  (1, 2)),   # -Y
    ((0, 0, 1),  (0, 1, 0),  (1, 1)),   # +Z
    ((0, 0, -1), (0, 1, 0),  (3, 1)),   # -Z
]



def upload_uniforms(uniform_data, locs):
    for i, data in enumerate(uniform_data):
        uploader = UNIFORM_UPLOADERS[data.type]
        uploader(locs[i], data.data)

def load_shader_source(filepath):
    with open(filepath, "r") as file:
        return file.read()


class Renderer:
    def __init__(self, width, height, objects: list[Object], uniform_data: list[UniformData]=[], lights: list[Light]=[]):
        self.width = width
        self.height = height

        self.shadow_width = 2048
        self.shadow_height = 2048

        self.window = None

        self.vertex_shader_path = "shaders/main_vertex.vert"
        self.fragment_shader_path = "shaders/main_fragment.frag"

        self.objects = objects
        self.lights = lights

        self.global_uniform_data = []

        self.perspective_projection_matrix = UniformData(perspective(math.radians(60), self.width / self.height, 0.1, 100.0), UniformType.MAT4, "projection")
        if self.perspective_projection_matrix is not None: self.global_uniform_data.append(self.perspective_projection_matrix)

        self.camera = Camera()
        self.view_uniform = UniformData(self.camera.get_view_matrix(), UniformType.MAT4, "view")
        self.global_uniform_data.append(self.view_uniform)

        self.camera_pos = self.camera.position
        self.global_uniform_data.extend(uniform_data)
        self.locs = []
        
        self.directional_lights_program_locs = {}
        self.point_light_program_locs = {}
        self.main_shader_program_locs = {}

        self.main_shader_program = None
        self.directional_lights_program = None
        self.point_lights_program = None

        self.clear_color = (0.1, 0.1, 0.1, 1.0)

        self.show_shadow_debug = False      
        self.shadow_export_dir = "shadow_maps"   
        self.debug_cell = 90                
        self.debug_dir_size = 180          
        self.debug_dir_scale = 0.5         
        self.debug_point_scale = 1.0     
        self.debug_program = None
        self.debug_vao = None
        self.debug_locs = {}

        self.uniform_uploaders = UNIFORM_UPLOADERS

        self.point_lights: list[Light] = []
        self.directional_lights:list[Light] = []

        self.sun = sun()
        self.directional_lights.append(self.sun)

        self.max_directional_lights = 2
        self.max_point_lights = 4

        for light in lights:
            if light.type == UniformType.LIGHT_TYPE_POINT:
                self.point_lights.append(light)
            elif light.type == UniformType.LIGHT_TYPE_DIRECTIONAL:
                self.directional_lights.append(light)

        if len(self.point_lights) > 4 or len(self.directional_lights) > 2:
            raise ValueError("maximum lights exceeded")

    def init_glfw(self):
        if not glfw.init():
            print("glfw initiation error")
            return

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(self.width, self.height, "bruh", None, None)
        if not self.window:
            print("error creating window")
            glfw.terminate()
            return

        glfw.make_context_current(self.window)

        glfw.set_key_callback(self.window, self.on_key)

        glClearColor(*self.clear_color)
        glEnable(GL_DEPTH_TEST)
        
    def compile_shaders(self):
        vertex_shader = shaders.compileShader(load_shader_source(self.vertex_shader_path), GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(load_shader_source(self.fragment_shader_path), GL_FRAGMENT_SHADER)
        self.main_shader_program = shaders.compileProgram(vertex_shader, fragment_shader, validate=False)

        directional_lights_vertex_shader = shaders.compileShader(load_shader_source("shaders/lighting_and_shadows/directional_vertex.vert"), GL_VERTEX_SHADER)
        directional_lights_fragment_shader = shaders.compileShader(load_shader_source("shaders/lighting_and_shadows/directional_fragment.frag"), GL_FRAGMENT_SHADER)
        self.directional_lights_program = shaders.compileProgram(directional_lights_vertex_shader, directional_lights_fragment_shader)

        point_lights_vertex_shader = shaders.compileShader(load_shader_source("shaders/lighting_and_shadows/point_vertex.vert"), GL_VERTEX_SHADER)
        point_lights_geometry_shader = shaders.compileShader(load_shader_source("shaders/lighting_and_shadows/point_geometry.geom"), GL_GEOMETRY_SHADER)
        point_lights_fragment_shader = shaders.compileShader(load_shader_source("shaders/lighting_and_shadows/point_fragment.frag"), GL_FRAGMENT_SHADER)
        self.point_lights_program = shaders.compileProgram(point_lights_vertex_shader, point_lights_geometry_shader, point_lights_fragment_shader)

        debug_vertex_shader = shaders.compileShader(load_shader_source("debug/debug_vertex.vert"), GL_VERTEX_SHADER)
        debug_fragment_shader = shaders.compileShader(load_shader_source("debug/debug_fragment.frag"), GL_FRAGMENT_SHADER)

        self.debug_program = shaders.compileProgram(debug_vertex_shader, debug_fragment_shader, validate=False)


    def create_buffers(self):
        self.debug_vao = glGenVertexArrays(1)
        
        for obj in self.objects:
            obj.vao = glGenVertexArrays(1)
            obj.vbo = glGenBuffers(1)
            obj.ebo = glGenBuffers(1)

            glBindVertexArray(obj.vao)

            glBindBuffer(GL_ARRAY_BUFFER, obj.vbo)
            glBufferData(GL_ARRAY_BUFFER, obj.combined.nbytes, obj.combined, GL_STATIC_DRAW)

            float_size = obj.combined.itemsize 
            stride = 9 * float_size

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            color_offset = 3 * float_size
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(color_offset))
            glEnableVertexAttribArray(1)


            normal_offset = 6 * float_size
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(normal_offset))
            glEnableVertexAttribArray(2)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, obj.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, obj.indices.nbytes, obj.indices, GL_STATIC_DRAW)

            glBindVertexArray(0)

        if self.point_lights:
            for i, point_light in enumerate(self.point_lights):
                point_light.fbo = glGenFramebuffers(1)
                point_light.shadow_map = glGenTextures(1)
                glBindTexture(GL_TEXTURE_CUBE_MAP, point_light.shadow_map)

                for j in range(6):
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + j, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

                glBindFramebuffer(GL_FRAMEBUFFER, point_light.fbo)    
                glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, point_light.shadow_map, 0)
                glDrawBuffer(GL_NONE)
                glReadBuffer(GL_NONE)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)

        if self.directional_lights:
            for i, directional_light in enumerate(self.directional_lights):
                directional_light.fbo = glGenFramebuffers(1)
                directional_light.shadow_map = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, directional_light.shadow_map)

                glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.shadow_width, self.shadow_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)

                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
                borderColor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
                glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor)

                glBindFramebuffer(GL_FRAMEBUFFER, directional_light.fbo)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, directional_light.shadow_map, 0)
                glDrawBuffer(GL_NONE)
                glReadBuffer(GL_NONE)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
    

    def get_locations(self):
        """ directional light program's locs """

        model_loc = glGetUniformLocation(self.directional_lights_program, "model")
        self.directional_lights_program_locs["model_loc"] = model_loc

        light_space_matrix_loc = glGetUniformLocation(self.directional_lights_program, "lightSpaceMatrix")
        self.directional_lights_program_locs["light_space_matrix_loc"] = light_space_matrix_loc

        """point light program's locs"""

        model_loc = glGetUniformLocation(self.point_lights_program, "model")
        self.point_light_program_locs["model_loc"] = model_loc

        shadow_matrix_locs = []

        for i in range(6):
            matrix_loc = glGetUniformLocation(self.point_lights_program, f"shadowMatrices[{i}]")
            shadow_matrix_locs.append(matrix_loc)

        self.point_light_program_locs["shadow_matrix_locs"] = shadow_matrix_locs

        light_pos_loc = glGetUniformLocation(self.point_lights_program, "lightPos")
        self.point_light_program_locs["light_pos_loc"] = light_pos_loc

        far_plane_loc = glGetUniformLocation(self.point_lights_program, "far_plane")
        self.point_light_program_locs["far_plane_loc"] = far_plane_loc

        """main shader program locs"""

        #vertex shader uniforms (user defined)
        for data in self.global_uniform_data:
            loc = glGetUniformLocation(self.main_shader_program, data.name)
            self.locs.append(loc)

        for obj in self.objects:
            for data in obj.uniform_data:
                loc = glGetUniformLocation(self.main_shader_program, data.name)
                obj_model_loc = glGetUniformLocation(self.main_shader_program, "model")
                obj.uniform_locs.append(loc)
                obj.model_loc = obj_model_loc

        #shadow related
        self.main_shader_program_locs["active_point_lights_loc"] = glGetUniformLocation(self.main_shader_program, "activePointLights")
        self.main_shader_program_locs["active_dir_lights_loc"] = glGetUniformLocation(self.main_shader_program, "activeDirLights")

        camera_pos_loc = glGetUniformLocation(self.main_shader_program, "cameraPos")
        self.main_shader_program_locs["camera_pos_loc"] = camera_pos_loc

        #directional light struct data
        directional_light_locs = []
        for i, light in enumerate(self.directional_lights):
            light_locs_dict = {}
            
            light_locs_dict["dir_loc"] = glGetUniformLocation(self.main_shader_program, f"dirLights[{i}].direction")
        
            light_locs_dict["color_loc"] = glGetUniformLocation(self.main_shader_program, f"dirLights[{i}].color")
           
            light_locs_dict["intensity_loc"] = glGetUniformLocation(self.main_shader_program, f"dirLights[{i}].intensity")
            
            light_locs_dict["matrix_loc"] = glGetUniformLocation(self.main_shader_program, f"dirLights[{i}].lightSpaceMatrix")

            directional_light_locs.append(light_locs_dict)

        self.main_shader_program_locs["directional_light_locs"] = directional_light_locs

        self.main_shader_program_locs["dir_sampler_locs"] = [glGetUniformLocation(self.main_shader_program, f"dirShadowMaps[{i}]") for i in range(self.max_directional_lights)]

        #point light struct data
        point_light_locs = []

        for i, light in enumerate(self.point_lights):
            light_locs_dict = {}
            
            light_locs_dict["pos_loc"] = glGetUniformLocation(self.main_shader_program, f"pointLights[{i}].position")
        
            light_locs_dict["color_loc"] = glGetUniformLocation(self.main_shader_program, f"pointLights[{i}].color")
            
            light_locs_dict["intensity_loc"] = glGetUniformLocation(self.main_shader_program, f"pointLights[{i}].intensity")
            
            light_locs_dict["far_plane"] = glGetUniformLocation(self.main_shader_program, f"pointLights[{i}].farPlane")

            point_light_locs.append(light_locs_dict)

        self.main_shader_program_locs["point_light_locs"] = point_light_locs

        self.main_shader_program_locs["point_sampler_locs"] = [glGetUniformLocation(self.main_shader_program, f"pointShadowMaps[{i}]") for i in range(self.max_point_lights)]

        for name in ("rect", "mode", "faceForward", "faceRight", "faceUp", "scale", "tex2D", "texCube"):
            self.debug_locs[name] = glGetUniformLocation(self.debug_program, name)

    def upload_main_shader_uniforms(self):
        upload_uniforms(self.global_uniform_data, self.locs)
        self.uniform_uploaders[UniformType.INT](self.main_shader_program_locs["active_point_lights_loc"], len(self.point_lights))
        self.uniform_uploaders[UniformType.INT](self.main_shader_program_locs["active_dir_lights_loc"], len(self.directional_lights))

        self.uniform_uploaders[UniformType.VEC3](self.main_shader_program_locs["camera_pos_loc"], self.camera_pos) 

        for i, light in enumerate(self.directional_lights):
            light_locs = self.main_shader_program_locs["directional_light_locs"][i]

            self.uniform_uploaders[UniformType.VEC3](light_locs["dir_loc"], light.pos_dir)

            self.uniform_uploaders[UniformType.VEC3](light_locs["color_loc"], light.color)

            self.uniform_uploaders[UniformType.FLOAT](light_locs["intensity_loc"], light.intensity)

            self.uniform_uploaders[UniformType.MAT4](light_locs["matrix_loc"], light.light_matrices)

        for i, light in enumerate(self.point_lights):
            light_locs = self.main_shader_program_locs["point_light_locs"][i]

            self.uniform_uploaders[UniformType.VEC3](light_locs["pos_loc"], light.pos_dir)

            self.uniform_uploaders[UniformType.VEC3](light_locs["color_loc"], light.color)

            self.uniform_uploaders[UniformType.FLOAT](light_locs["intensity_loc"], light.intensity)

            self.uniform_uploaders[UniformType.FLOAT](light_locs["far_plane"], light.far_plane)

        for i, loc in enumerate(self.main_shader_program_locs["dir_sampler_locs"]):
            self.uniform_uploaders[UniformType.INT](loc, i)

        for i, loc in enumerate(self.main_shader_program_locs["point_sampler_locs"]):
            self.uniform_uploaders[UniformType.INT](loc, self.max_directional_lights + i)

    def execute_directional_lights_program(self):
        for directional_light in self.directional_lights:
            glBindFramebuffer(GL_FRAMEBUFFER, directional_light.fbo)
            glViewport(0, 0, self.shadow_width, self.shadow_height)
            glClear(GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.directional_lights_program)

            self.uniform_uploaders[UniformType.MAT4](self.directional_lights_program_locs["light_space_matrix_loc"], directional_light.light_matrices)

            for obj in self.objects:
                self.uniform_uploaders[UniformType.MAT4](self.directional_lights_program_locs["model_loc"], obj.model_matrix)
                glBindVertexArray(obj.vao)
                glDrawElements(GL_TRIANGLES, obj.index_count, GL_UNSIGNED_INT, None)

    def execute_point_lights_program(self):
        for point_light in self.point_lights:
            glBindFramebuffer(GL_FRAMEBUFFER, point_light.fbo)
            glViewport(0, 0, self.shadow_width, self.shadow_height)
            glClear(GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.point_lights_program)

            self.uniform_uploaders[UniformType.VEC3](self.point_light_program_locs["light_pos_loc"], point_light.pos_dir)
            self.uniform_uploaders[UniformType.FLOAT](self.point_light_program_locs["far_plane_loc"], point_light.far_plane)

            for i in range(6):
                self.uniform_uploaders[UniformType.MAT4](self.point_light_program_locs["shadow_matrix_locs"][i], point_light.light_matrices[i])

            for obj in self.objects:
                self.uniform_uploaders[UniformType.MAT4](self.point_light_program_locs["model_loc"], obj.model_matrix)
                glBindVertexArray(obj.vao)
                glDrawElements(GL_TRIANGLES, obj.index_count, GL_UNSIGNED_INT, None)

    def execute_main_shader_program(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.main_shader_program)

        for i, light in enumerate(self.directional_lights):
            texture_unit = i

            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, light.shadow_map)

            light.texture_unit_index = texture_unit


        for i, light in enumerate(self.point_lights):
            texture_unit = self.max_directional_lights + i

            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_CUBE_MAP, light.shadow_map)

            light.texture_unit_index = texture_unit
        self.upload_main_shader_uniforms()
        
        for i, obj in enumerate(self.objects):
            upload_uniforms(obj.uniform_data, obj.uniform_locs)
            self.uniform_uploaders[UniformType.MAT4](obj.model_loc, obj.model_matrix)
            glBindVertexArray(obj.vao)
            glDrawElements(GL_TRIANGLES, obj.index_count, GL_UNSIGNED_INT, None)


    def on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_T:
            self.show_shadow_debug = not self.show_shadow_debug

    def draw_shadow_debug(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.width, self.height)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(self.debug_program)
        glBindVertexArray(self.debug_vao)
 
        L = self.debug_locs
        glUniform1i(L["tex2D"], 6)
        glUniform1i(L["texCube"], 7)
 
        margin = 8
        cursor = {"x": margin, "y": margin, "row_h": 0}
 
        def place(w, h):
            if cursor["x"] + w > self.width and cursor["x"] > margin:
                cursor["x"] = margin
                cursor["y"] += cursor["row_h"] + margin
                cursor["row_h"] = 0
            px, py = cursor["x"], cursor["y"]
            cursor["x"] += w + margin
            cursor["row_h"] = max(cursor["row_h"], h)
            return px, py
 
        def quad(px, py, size):
            glUniform4f(L["rect"],
                        2.0 * px / self.width - 1.0, 2.0 * py / self.height - 1.0,
                        2.0 * size / self.width, 2.0 * size / self.height)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
 
        for light in self.directional_lights:
            glUniform1i(L["mode"], 0)
            glUniform1f(L["scale"], self.debug_dir_scale)
            glActiveTexture(GL_TEXTURE6)
            glBindTexture(GL_TEXTURE_2D, light.shadow_map)
            px, py = place(self.debug_dir_size, self.debug_dir_size)
            quad(px, py, self.debug_dir_size)
 
        c = self.debug_cell
        for light in self.point_lights:
            glUniform1i(L["mode"], 1)
            glUniform1f(L["scale"], self.debug_point_scale)
            glActiveTexture(GL_TEXTURE7)
            glBindTexture(GL_TEXTURE_CUBE_MAP, light.shadow_map)
            px, py = place(4 * c, 3 * c)
            for forward, up, (col, row) in POINT_FACE_DISPLAY:
                f = np.array(forward, dtype=np.float32)
                u = np.array(up, dtype=np.float32)
                r = np.cross(f, u)
                glUniform3f(L["faceForward"], *f)
                glUniform3f(L["faceRight"], *r)
                glUniform3f(L["faceUp"], *u)
                quad(px + col * c, py + (2 - row) * c, c)
 
        glEnable(GL_DEPTH_TEST)


    def draw_loop(self, update_callback=None):
        while not glfw.window_should_close(self.window):

            if update_callback:
                update_callback()

            if self.camera is not None:
                self.camera.update(self.window)
                self.view_uniform.data = self.camera.get_view_matrix()
                self.camera_pos = self.camera.position
           
            self.execute_directional_lights_program()
            self.execute_point_lights_program()
            self.execute_main_shader_program()

            if self.show_shadow_debug:
                self.draw_shadow_debug()

            glfw.swap_buffers(self.window)
            glfw.poll_events()

    def cleanup(self):
        for obj in self.objects:
            glDeleteVertexArrays(1, [obj.vao])
            glDeleteBuffers(1, [obj.vbo])

        for light in (self.point_lights + self.directional_lights):
            glDeleteFramebuffers(1, [light.fbo])
            glDeleteTextures(1, [light.shadow_map])
        glfw.terminate()

    def run_app(self, update_callback=None):
        self.init_glfw()
        self.compile_shaders()
        self.get_locations()
        self.create_buffers()
        self.draw_loop(update_callback)
        self.cleanup()    