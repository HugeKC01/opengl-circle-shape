from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18
import numpy as np
import glm  # For matrix operations

# Shader sources for normal sphere
vertex_shader_source_a = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


out vec3 Normal;
out vec3 FragPos;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

fragment_shader_source_a = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main()
{
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""

# Shader sources for star-patterned sphere
vertex_shader_source_b = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 Normal;
out vec3 FragPos;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

fragment_shader_source_b = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

uniform vec4 LightDir;
uniform vec4 HVector;
uniform vec4 BallCenter;
uniform vec4 SpecularColor;
uniform vec4 Red, Yellow, Blue;
uniform vec4 HalfSpace0, HalfSpace1, HalfSpace2, HalfSpace3, HalfSpace4;
uniform float InOrOutInit;
uniform float StripeWidth;
uniform float FWidth;

void main()
{
    vec4 normal;
    vec4 p;
    vec4 surfColor;
    float intensity;
    vec4 distance;
    float inorout;

    p.xyz = normalize(FragPos - BallCenter.xyz);
    p.w = 1.0;

    inorout = InOrOutInit;

    distance[0] = dot(p, HalfSpace0);
    distance[1] = dot(p, HalfSpace1);
    distance[2] = dot(p, HalfSpace2);
    distance[3] = dot(p, HalfSpace3);

    distance = smoothstep(-FWidth, FWidth, distance);

    inorout += dot(distance, vec4(1.0));

    distance.x = dot(p, HalfSpace4);
    distance.y = StripeWidth - abs(p.z);

    distance = smoothstep(-FWidth, FWidth, distance);

    inorout += distance.x;

    inorout = clamp(inorout, 0.0, 1.0);

    surfColor = mix(Yellow, Red, inorout);
    surfColor = mix(surfColor, Blue, distance.y);

    normal = p;

    intensity = 0.2;
    intensity += 0.8 * clamp(dot(LightDir.xyz, normal.xyz), 0.0, 1.0);
    surfColor *= intensity;

    intensity = clamp(dot(HVector.xyz, normal.xyz), 0.0, 1.0);
    intensity = pow(intensity, SpecularColor.a);
    surfColor += SpecularColor * intensity;

    FragColor = surfColor;
}
"""

# Shader sources for brick-patterned sphere
brick_vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

brick_fragment_shader_source = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 BrickColor;
uniform vec3 MortarColor;
uniform vec2 BrickSize;
uniform vec2 BrickPct;
uniform vec3 LightPosition;

void main()
{
    vec3 lightDir = normalize(LightPosition - FragPos);
    vec3 norm = normalize(Normal);
    float diff = max(dot(norm, lightDir), 0.0);

    // Compute brick pattern
    vec3 color;
    vec2 position = FragPos.xy / BrickSize;
    if (fract(position.y * 0.5) > 0.5)
        position.x += 0.5;
    position = fract(position);

    if (position.x > BrickPct.x || position.y > BrickPct.y)
        color = MortarColor;
    else
        color = BrickColor;

    color *= diff;

    FragColor = vec4(color, 1.0);
}
"""

shader_program_a = None
shader_program_b = None
brick_shader_program = None
current_shader_program = None
sphere_vao = None
sphere_vbo = None
sphere_ebo = None
indices = None
show_three_spheres = False
background_color = [0.0, 0.0, 0.0, 1.0]  # Start with black background

def create_sphere(radius, sectors, stacks):
    vertices = []
    normals = []
    indices = []

    for stack in range(stacks + 1):
        phi = np.pi / 2 - stack * np.pi / stacks
        for sector in range(sectors + 1):
            theta = sector * 2 * np.pi / sectors
            x = radius * np.cos(phi) * np.cos(theta)
            y = radius * np.cos(phi) * np.sin(theta)
            z = radius * np.sin(phi)
            vertices.extend([x, y, z])
            normals.extend([x, y, z])  # Normal is the same as position for a sphere

    for stack in range(stacks):
        for sector in range(sectors):
            first = stack * (sectors + 1) + sector
            second = first + sectors + 1
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(indices, dtype=np.uint32)

def init():
    global shader_program_a, shader_program_b, brick_shader_program, current_shader_program, sphere_vao, sphere_vbo, sphere_ebo, indices

    shader_program_a = compileProgram(compileShader(vertex_shader_source_a, GL_VERTEX_SHADER), compileShader(fragment_shader_source_a, GL_FRAGMENT_SHADER))
    shader_program_b = compileProgram(compileShader(vertex_shader_source_b, GL_VERTEX_SHADER), compileShader(fragment_shader_source_b, GL_FRAGMENT_SHADER))
    brick_shader_program = compileProgram(compileShader(brick_vertex_shader_source, GL_VERTEX_SHADER), compileShader(brick_fragment_shader_source, GL_FRAGMENT_SHADER))
    current_shader_program = shader_program_a

    vertices, normals, indices = create_sphere(0.5, 36, 18)  # Create a sphere with radius 0.5
    indices = np.array(indices, dtype=np.uint32)  # ensure numpy array of correct type.

    sphere_vao = glGenVertexArrays(1)
    sphere_vbo = glGenBuffers(1)
    sphere_ebo = glGenBuffers(1)

    glBindVertexArray(sphere_vao)

    glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    normal_vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, normal_vbo)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glBindVertexArray(0)

def set_uniforms_for_shader(shader_program):
    model = glm.mat4(1.0)
    view = glm.lookAt(glm.vec3(0, 0, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    projection = glm.perspective(glm.radians(45.0), 800.0 / 600.0, 0.1, 100.0)

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
    view_pos_loc = glGetUniformLocation(shader_program, "viewPos")
    light_color_loc = glGetUniformLocation(shader_program, "lightColor")
    object_color_loc = glGetUniformLocation(shader_program, "objectColor")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))
    glUniform3f(light_pos_loc, 2.0, 2.0, 2.0)
    glUniform3f(view_pos_loc, 0.0, 0.0, 3.0)
    glUniform3f(light_color_loc, 1.0, 1.0, 1.0)
    glUniform3f(object_color_loc, 1.0, 0.5, 0.31)

    if shader_program == shader_program_b:
        # Set additional uniforms for star-patterned sphere
        glUniform4f(glGetUniformLocation(shader_program, "LightDir"), 0.57735, 0.57735, 0.57735, 0.0)
        glUniform4f(glGetUniformLocation(shader_program, "HVector"), 0.32506, 0.32506, 0.88808, 0.0)
        glUniform4f(glGetUniformLocation(shader_program, "BallCenter"), 0.0, 0.0, 0.0, 1.0)
        glUniform4f(glGetUniformLocation(shader_program, "SpecularColor"), 0.4, 0.4, 0.4, 60.0)
        glUniform4f(glGetUniformLocation(shader_program, "Red"), 0.6, 0.0, 0.0, 1.0)
        glUniform4f(glGetUniformLocation(shader_program, "Blue"), 0.0, 0.3, 0.6, 1.0)
        glUniform4f(glGetUniformLocation(shader_program, "Yellow"), 0.6, 0.5, 0.0, 1.0)
        glUniform4f(glGetUniformLocation(shader_program, "HalfSpace0"), 1.0, 0.0, 0.0, 0.2)
        glUniform4f(glGetUniformLocation(shader_program, "HalfSpace1"), 0.309016994, 0.951056516, 0.0, 0.2)
        glUniform4f(glGetUniformLocation(shader_program, "HalfSpace2"), -0.809016994, 0.587785252, 0.0, 0.2)
        glUniform4f(glGetUniformLocation(shader_program, "HalfSpace3"), -0.809016994, -0.587785252, 0.0, 0.2)
        glUniform4f(glGetUniformLocation(shader_program, "HalfSpace4"), 0.309016994, -0.951056516, 0.0, 0.2)
        glUniform1f(glGetUniformLocation(shader_program, "InOrOutInit"), -3.0)
        glUniform1f(glGetUniformLocation(shader_program, "StripeWidth"), 0.3)
        glUniform1f(glGetUniformLocation(shader_program, "FWidth"), 0.005)

    elif shader_program == brick_shader_program:
        # Set additional uniforms for brick-patterned sphere
        glUniform3f(glGetUniformLocation(shader_program, "BrickColor"), 1.0, 0.3, 0.2)
        glUniform3f(glGetUniformLocation(shader_program, "MortarColor"), 0.85, 0.86, 0.84)
        glUniform2f(glGetUniformLocation(shader_program, "BrickSize"), 0.30, 0.15)
        glUniform2f(glGetUniformLocation(shader_program, "BrickPct"), 0.90, 0.85)
        glUniform3f(glGetUniformLocation(shader_program, "LightPosition"), 0.0, 0.0, 4.0)

def display():
    global current_shader_program, sphere_vao, indices, show_three_spheres, background_color

    glClearColor(*background_color)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    if show_three_spheres:
        # Draw first sphere with shader_program_a
        glUseProgram(shader_program_a)
        set_uniforms_for_shader(shader_program_a)
        model = glm.translate(glm.mat4(1.0), glm.vec3(-1.0, 0.0, 0.0))
        model = glm.scale(model, glm.vec3(0.7, 0.7, 0.7))  # Scale down
        glUniformMatrix4fv(glGetUniformLocation(shader_program_a, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glBindVertexArray(sphere_vao)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        # Draw second sphere with shader_program_b
        glUseProgram(shader_program_b)
        set_uniforms_for_shader(shader_program_b)
        model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.0, 0.0))
        model = glm.scale(model, glm.vec3(0.7, 0.7, 0.7))  # Scale down
        glUniformMatrix4fv(glGetUniformLocation(shader_program_b, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glBindVertexArray(sphere_vao)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        # Draw third sphere with brick_shader_program
        glUseProgram(brick_shader_program)
        set_uniforms_for_shader(brick_shader_program)
        model = glm.translate(glm.mat4(1.0), glm.vec3(1.0, 0.0, 0.0))
        model = glm.scale(model, glm.vec3(0.7, 0.7, 0.7))  # Scale down
        glUniformMatrix4fv(glGetUniformLocation(brick_shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glBindVertexArray(sphere_vao)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    else:
        glUseProgram(current_shader_program)
        set_uniforms_for_shader(current_shader_program)
        glBindVertexArray(sphere_vao)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glBindVertexArray(0)
    render_mode_label()
    glutSwapBuffers()

def reshape(width, height):
    glViewport(0, 0, width, height)

def keyboard(key, x, y):
    global current_shader_program, show_three_spheres, background_color
    if key == b'1':
        current_shader_program = shader_program_a
        show_three_spheres = False
    elif key == b'2':
        current_shader_program = shader_program_b
        show_three_spheres = False
    elif key == b'3':
        current_shader_program = brick_shader_program
        show_three_spheres = False
    elif key == b'4':
        show_three_spheres = True
    elif key == b'd' or key == b'D':
        # Toggle background color between black and white
        if background_color[0] == 0.0:
            background_color = [1.0, 1.0, 1.0, 1.0]
        else:
            background_color = [0.0, 0.0, 0.0, 1.0]
    glutPostRedisplay()

def render_mode_label():
    glUseProgram(0)
    # Use window coordinates for robust text placement if available
    try:
        glWindowPos2f(10, 570)
        use_windowpos = True
    except Exception:
        use_windowpos = False
    if not use_windowpos:
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 800, 0, 600)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

    # Switch text color based on background color
    if background_color[0] >= 0.5:
        glColor3f(0.0, 0.0, 0.0)  # Black text on light bg
    else:
        glColor3f(1.0, 1.0, 1.0)  # White text on dark bg

    if show_three_spheres:
        label = b"All Modes"
    elif current_shader_program == shader_program_a:
        label = b"Normal Mode"
    elif current_shader_program == shader_program_b:
        label = b"Toy Mode"
    elif current_shader_program == brick_shader_program:
        label = b"Brick Mode"
    else:
        label = b"Unknown Mode"

    if not use_windowpos:
        glRasterPos2f(10, 570)
    for c in label:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c)

    if not use_windowpos:
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"OpenGL Sphere")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glEnable(GL_DEPTH_TEST)

    init()

    glutMainLoop()