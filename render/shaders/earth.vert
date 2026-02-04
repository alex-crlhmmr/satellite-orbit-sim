#version 330 core

uniform mat4 mvp;
uniform mat4 model;

in vec3 position;
in vec3 normal;
in vec2 texcoord;

out vec3 v_normal;
out vec2 v_texcoord;
out vec3 v_position;

void main() {
    vec4 world_pos = model * vec4(position, 1.0);
    v_position = world_pos.xyz;
    v_normal = normalize(mat3(model) * normal);
    v_texcoord = texcoord;
    gl_Position = mvp * vec4(position, 1.0);
}
