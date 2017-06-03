import pyglet
import math
from math import pi, sin, cos
import numpy as np
from pyglet.gl import *


try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True,)
    window = pyglet.window.Window(resizable=True, config=config)
except pyglet.window.NoSuchConfigException:
    print('fall back to no multisampling')
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True)


# window = pyglet.window.Window()
# # image = pyglet.resource.image('kitten.jpg')
# image = pyglet.image.load('/norep/basicsampling_1.png')
# label = pyglet.text.Label('Hello, world',
#                           font_name='Times New Roman',
#                           font_size=36,
#                           x=window.width // 2, y=window.height // 2 * 1.9,
#                           anchor_x='center', anchor_y='top')

window.push_handlers(pyglet.window.event.WindowEventLogger())


# @window.event
# def on_key_press(symbol, modifiers):
#     print(symbol, modifiers)


# @window.event
# def on_mouse_press(x, y, button, modifiers):
#     print(x, y, button, modifiers)


@window.event
def on_mouse_drag(x, y, dx, dy, button, modifiers):
    global rx, ry, rz
    global points

    R1 = np.identity(K)
    theta = dx * 3.1416 / 180

    # # rotates around axis into screen
    # R[0, 0] = math.cos(theta)
    # R[1, 1] = R[0, 0]
    # R[1, 0] = math.sin(theta)
    # R[0, 1] = - math.sin(theta)

    # rotates around vertical axis (vertical, coplanar with the screen surface)
    R1[0, 0] = math.cos(theta)
    R1[2, 2] = R1[0, 0]
    R1[2, 0] = math.sin(theta)
    R1[0, 2] = - R1[2, 0]

    phi = dy * 3.1416 / 180
    R2 = np.identity(K)
    # rotates around left-right axis (axis coplanar with screen surface)
    R2[1, 1] = math.cos(phi)
    R2[2, 2] = R2[0, 0]
    R2[2, 1] = math.sin(phi)
    R2[1, 2] = - R2[2, 1]

    points = points.dot(R1)
    points = points.dot(R2)

    # ry += dx
    # rx += dy
    # # rz += dt * 30

    # rx %= 360
    # ry %= 360
    # rz %= 360

    # print(x, y, dx, dy, button, modifiers)

# @window.event
# def on_draw():
#     window.clear()
#     # image.blit(0, 0)
#     # label.draw()


@window.event
def on_resize(width, height):
    # Override the default on_resize handler to create a 3D projection
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED


def update(dt):
    global rx, ry, rz
    rx += dt * 1
    ry += dt * 80
    rz += dt * 30
    rx %= 360
    ry %= 360
    rz %= 360

# pyglet.clock.schedule(update)


sphere = gluNewQuadric()


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0, -4)
    glRotatef(rz, 0, 0, 1)
    glRotatef(rx, 1, 0, 0)
    glRotatef(ry, 0, 1, 0)
    # torus.draw()
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(1.0, 1.0, 1.0, 0))
    for point in points:
        glPushMatrix()
        glTranslatef(point[0], point[1], point[2])
        glScalef(0.01, 0.01, 0.01)
        # torus.draw()
        gluSphere(sphere, 1.0, 10, 10)
        glPopMatrix()
    # for point in squared:
    #     glPushMatrix()
    #     glTranslatef(point[0], point[1], point[2])
    #     glScalef(0.05, 0.05, 0.05)
    #     # torus.draw()
    #     gluSphere(sphere, 1.0, 10, 10)
    #     glPopMatrix()

    AXIS_THICKNESS = 0.03

    # glColor3f(1, 1, 0)
    # pyglet.gl.glLineWidth(10.5)
    # pyglet.graphics.draw(
    #     2, pyglet.gl.GL_LINES,
    #     ('v3f', (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    # )

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(1.0, 0.0, 0.0, 0))
    gluCylinder(sphere, AXIS_THICKNESS, AXIS_THICKNESS, 1.0, 10, 10)

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0, 1.0, 0.0, 0))
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    glColor3f(0, 1, 0)
    # pyglet.gl.glLineWidth(10.5)
    # pyglet.graphics.draw(
    #     2, pyglet.gl.GL_LINES,
    #     ('v3f', (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    # )
    gluCylinder(sphere, AXIS_THICKNESS, AXIS_THICKNESS, 1.0, 10, 10)
    glPopMatrix()

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0, 0, 1.0, 0))
    glPushMatrix()
    glRotatef(-90, 0, 1, 0)
    glColor3f(0, 1, 0)
    # pyglet.gl.glLineWidth(10.5)
    # pyglet.graphics.draw(
    #     2, pyglet.gl.GL_LINES,
    #     ('v3f', (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    # )
    gluCylinder(sphere, AXIS_THICKNESS, AXIS_THICKNESS, 1.0, 10, 10)
    glPopMatrix()

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.5, 0, 0.3, 1))


# Define a simple function to create ctypes arrays of floats:
def vec(*args):
    return (GLfloat * len(args))(*args)


def setup():
    # One-time GL setup
    # glClearColor(1, 1, 1, 1)
    glClearColor(0, 0, 0, 1)
    glColor3f(1, 0, 0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    # Uncomment this line for a wireframe view
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always
    # include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    glLightfv(GL_LIGHT0, GL_POSITION, vec(.5, .5, 1, 0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, vec(.1, .1, 0.1, 0.1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1, 1, 1, 1))
    glLightfv(GL_LIGHT1, GL_POSITION, vec(1, 0, .5, 0))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.5, .5, .5, 1))
    glLightfv(GL_LIGHT1, GL_SPECULAR, vec(0.1, 0.1, 0.1, 0.1))

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.5, 0, 0.3, 1))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 0)


class Torus(object):
    def __init__(self, radius, inner_radius, slices, inner_slices):
        # Create the vertex and normal arrays.
        vertices = []
        normals = []

        u_step = 2 * pi / (slices - 1)
        v_step = 2 * pi / (inner_slices - 1)
        u = 0.
        for i in range(slices):
            cos_u = cos(u)
            sin_u = sin(u)
            v = 0.
            for j in range(inner_slices):
                cos_v = cos(v)
                sin_v = sin(v)

                d = (radius + inner_radius * cos_v)
                x = d * cos_u
                y = d * sin_u
                z = inner_radius * sin_v

                nx = cos_u * cos_v
                ny = sin_u * cos_v
                nz = sin_v

                vertices.extend([x, y, z])
                normals.extend([nx, ny, nz])
                v += v_step
            u += u_step

        # Create ctypes arrays of the lists
        vertices = (GLfloat * len(vertices))(*vertices)
        normals = (GLfloat * len(normals))(*normals)

        # Create a list of triangle indices.
        indices = []
        for i in range(slices - 1):
            for j in range(inner_slices - 1):
                p = i * inner_slices + j
                indices.extend([p, p + inner_slices, p + inner_slices + 1])
                indices.extend([p, p + inner_slices + 1, p + 1])
        indices = (GLuint * len(indices))(*indices)

        # Compile a display list
        self.list = glGenLists(1)
        glNewList(self.list, GL_COMPILE)

        glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glNormalPointer(GL_FLOAT, 0, normals)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)
        glPopClientAttrib()

        glEndList()

    def draw(self):
        glCallList(self.list)


setup()
torus = Torus(1, 0.3, 50, 30)
N = 1000
K = 4
# points = np.random.randn(N, K) * 0.5
A = np.random.randn(K)
cov = A.T.dot(A) + 0.1 * np.identity(K)
points = np.random.multivariate_normal(
    mean=np.zeros(K),
    cov=cov,
    size=(N,)
)

squared_side = 5
squared = np.zeros((squared_side * squared_side, K))
for i in range(squared_side):
    for j in range(squared_side):
        squared[i * squared_side + j][0] = i / squared_side
        squared[i * squared_side + j][1] = j / squared_side
        squared[i * squared_side + j][2] = (i * i + j * j) / squared_side / squared_side

rx = ry = rz = 0


pyglet.app.run()
