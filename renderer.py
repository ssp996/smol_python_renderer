from OpenGL.GL import *
from OpenGL.GL import shaders
import glfw
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

def upload_uniforms(uniform_data, locs):
    for i, data in enumerate(uniform_data):
        uploader = UNIFORM_UPLOADERS[data.type]
        uploader(locs[i], data.data)

class Object:
    def __init__(self, vertices, colors, normals, indices, uniform_data):
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

        self.uniform_data = uniform_data
        self.uniform_locs = []

class UniformData:
    def __init__(self, data, type, name):
        self.data = data
        self.type = type
        self.name = name

class Renderer:
    def __init__(self, objects: list[Object], uniform_data: list[UniformData]):
        self.width = 800
        self.height = 800

        self.window = None

        self.vertex_shader_path = "shaders/vertex_shader.vert"
        self.fragment_shader_path = "shaders/fragment_shader.frag"

        self.objects = objects
        self.global_uniform_data = uniform_data
        self.locs = []
        self.shader_program = None

        self.clear_color = (0.1, 0.1, 0.1, 1.0)

        self.uniform_uploaders = UNIFORM_UPLOADERS

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

        glClearColor(*self.clear_color)
        glEnable(GL_DEPTH_TEST)
        
    def compile_shaders(self):
        with open(self.vertex_shader_path, encoding='utf-8') as f:
            vertex_shader_source = f.read()

        with open(self.fragment_shader_path, encoding='utf-8') as f:
            fragment_shader_source = f.read()

        vertex_shader = shaders.compileShader(vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
        self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

    def create_buffers(self):
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

    def get_locations(self):
        for data in self.global_uniform_data:
            loc = glGetUniformLocation(self.shader_program, data.name)
            self.locs.append(loc)

        for obj in self.objects:
            for data in obj.uniform_data:
                loc = glGetUniformLocation(self.shader_program, data.name)
                obj.uniform_locs.append(loc)

    def draw_loop(self, model_update=None):
        while not glfw.window_should_close(self.window):

            if model_update:
                model_update()

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.shader_program)
            upload_uniforms(self.global_uniform_data, self.locs)

            for obj in self.objects:
                upload_uniforms(obj.uniform_data, obj.uniform_locs)
                glBindVertexArray(obj.vao)
                glDrawElements(GL_TRIANGLES, obj.index_count, GL_UNSIGNED_INT, None)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

    def cleanup(self):
        for obj in self.objects:
            glDeleteVertexArrays(1, [obj.vao])
            glDeleteBuffers(1, [obj.vbo])
        glfw.terminate()

    def run_app(self, model_update=None):
        self.init_glfw()
        self.compile_shaders()
        self.get_locations()
        self.create_buffers()
        self.draw_loop(model_update)
        self.cleanup()    