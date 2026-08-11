#version 330 core

uniform sampler2D day_texture;
uniform sampler2D night_texture;
uniform vec3 sun_direction;
uniform vec3 camera_position;

in vec3 v_normal;
in vec2 v_texcoord;
in vec3 v_position;

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 sun_dir = normalize(sun_direction);

    // Day/night factor based on angle between surface normal and sun direction
    float ndot = dot(normal, sun_dir);
    float blend = smoothstep(-0.1, 0.3, ndot);

    vec3 day_col = texture(day_texture, v_texcoord).rgb;
    vec3 night_col = texture(night_texture, v_texcoord).rgb;
    // Keep the surface legible in onboard nadir views even when the camera is
    // over the night side. The night texture remains dominant; this is a
    // small visualization floor rather than physical illumination.
    night_col = max(night_col, day_col * 0.35);

    vec3 color = mix(night_col, day_col, blend);

    // Atmospheric rim glow: stronger at grazing angles
    vec3 view_dir = normalize(camera_position - v_position);
    float rim = 1.0 - max(dot(normal, view_dir), 0.0);
    rim = pow(rim, 3.0);
    vec3 atmosphere = vec3(0.3, 0.5, 1.0) * rim * 0.6;

    color += atmosphere;

    frag_color = vec4(color, 1.0);
}
