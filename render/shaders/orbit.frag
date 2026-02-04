#version 330 core

in float v_alpha;

out vec4 frag_color;

void main() {
    // Bright cyan/white core that fades with alpha
    vec3 core_color = vec3(0.4, 0.9, 1.0);
    vec3 white = vec3(1.0, 1.0, 1.0);
    // Brighter toward full alpha
    vec3 color = mix(core_color, white, v_alpha * 0.5);
    frag_color = vec4(color, v_alpha);
}
