#version 330 core

uniform sampler2D day_texture;
uniform sampler2D night_texture;
uniform vec3 sun_direction;

in vec3 v_normal;
in vec2 v_texcoord;

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 sun_dir = normalize(sun_direction);

    // Direct solar incidence on a Lambertian surface. Decode texture samples
    // from sRGB before applying illumination, then encode for display.
    float ndot = dot(normal, sun_dir);
    float solar_incidence = max(ndot, 0.0);
    float night_visibility = 1.0 - smoothstep(-0.08, 0.02, ndot);

    vec3 day_linear = pow(texture(day_texture, v_texcoord).rgb, vec3(2.2));
    vec3 night_linear = pow(texture(night_texture, v_texcoord).rgb, vec3(2.2));
    vec3 radiance = day_linear * solar_incidence
                  + night_linear * night_visibility;

    frag_color = vec4(pow(max(radiance, vec3(0.0)), vec3(1.0 / 2.2)), 1.0);
}
