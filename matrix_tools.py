import numpy as np
import math

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(fov / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(left, right, bottom, top, near, far):
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def translate(x, y, z):
    m = np.identity(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def scale(x, y, z):
    return np.diag([x, y, z, 1.0]).astype(np.float32)

def rotate_x(angle):
    c, s = math.cos(angle), math.sin(angle)

    return np.array([
        [ 1, 0, 0, 0],
        [ 0, c,-s, 0],
        [ 0, s, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)


def rotate_y(angle):
    c, s = math.cos(angle), math.sin(angle)

    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)


def rotate_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    
    return np.array([
        [ c,-s, 0, 0],
        [ s, c, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float32)
    


def look_at(eye, target, up=(0.0, 1.0, 0.0)):

    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = normalize(target - eye)           # camera -Z axis
    right = normalize(np.cross(forward, up))    # camera +X axis
    true_up = np.cross(right, forward)          # camera +Y axis

    view = np.identity(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(true_up, eye)
    view[2, 3] = np.dot(forward, eye)
    return view

def directional_light_matrix(direction, distance=20.0, extent=12.0):
    direction = normalize(direction)
    eye = -direction * distance
    view = look_at(eye, (0.0, 0.0, 0.0))
    proj = orthographic(-extent, extent, -extent, extent, 1.0, distance * 2.0)
    return proj @ view


def point_light_matrices(position, far_plane, near_plane=0.1):

    # cubemap face order: +X, -X, +Y, -Y, +Z, -Z (with the up vectors opengl expects)
    CUBE_FACE_DIRS = [
        (( 1, 0, 0), (0, -1,  0)),
        ((-1, 0, 0), (0, -1,  0)),
        (( 0, 1, 0), (0,  0,  1)),
        (( 0, -1, 0), (0, 0, -1)),
        (( 0, 0, 1), (0, -1,  0)),
        (( 0, 0, -1), (0, -1, 0)),
    ]

    position = np.asarray(position, dtype=np.float32)
    proj = perspective(math.radians(90.0), 1.0, near_plane, far_plane)
    return [proj @ look_at(position, position + np.array(d, dtype=np.float32), up) for d, up in CUBE_FACE_DIRS]

