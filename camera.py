from matrix_tools import *
import glfw

class Camera:
    def __init__(self, position=(0.0, 4.5, 9.0), target=(0.0, 1.0, 0.0)):
        self.position = np.asarray(position, dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.pitch = 0
        self.yaw = -90 #start pointing at -Z

        self.sens = 0.1
        self.speed = 5.0

        self.window = None
        self.captured = False
        self.last_mouse = None
        self.last_time = None

        self.point_at(target)


    def point_at(self, target):
        d = normalize(np.asarray(target, dtype=np.float32) - self.position)
        self.pitch = math.degrees(math.asin(float(d[1])))
        self.yaw = math.degrees(math.atan2(float(d[2]), float(d[0])))

    @property
    def forward(self):
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        return normalize([math.cos(pitch) * math.cos(yaw), math.sin(pitch), math.cos(pitch) * math.sin(yaw)])

    @property
    def right(self):
        return normalize(np.cross(self.forward, self.world_up))

    def get_view_matrix(self):
        return look_at(self.position, self.position + self.forward, self.world_up)

    def attach(self, window):
        self.window = window
        self.capture(True)

    def capture(self, capture):
        self.captured = capture
        self.last_mouse = None
        mode = glfw.CURSOR_DISABLED if capture else glfw.CURSOR_NORMAL
        glfw.set_input_mode(self.window, glfw.CURSOR, mode)

    def update(self, window):
        if self.window is None:
            self.attach(window)

        now = glfw.get_time()

        dt = 0.0 if self.last_time is None else min(now - self.last_time, 0.1)
        self.last_time = now

        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS and self.captured:
            self.capture(False)
        elif not self.captured and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.capture(True)

        if self.captured:
            x, y = glfw.get_cursor_pos(window)
            if self.last_mouse is not None:
                dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
                self.yaw += dx * self.sens
                self.pitch = float(np.clip(self.pitch - dy * self.sens, -89.0, 89.0))

            self.last_mouse = (x, y)

        down = lambda key: glfw.get_key(window, key) == glfw.PRESS

        move = np.zeros(3, dtype=np.float32)
        if down(glfw.KEY_W): move += self.forward
        if down(glfw.KEY_S): move -= self.forward
        if down(glfw.KEY_D): move += self.right
        if down(glfw.KEY_A): move -= self.right
        if down(glfw.KEY_SPACE): move += self.world_up
        if down(glfw.KEY_LEFT_CONTROL): move -= self.world_up

        if np.linalg.norm(move) > 0:
            speed = self.speed
            self.position += normalize(move) * speed * dt


