#version 330 core

uniform mat4 mvp;

in vec3 position;
in float alpha;

out float v_alpha;

void main() {
    v_alpha = alpha;
    gl_Position = mvp * vec4(position, 1.0);
}
