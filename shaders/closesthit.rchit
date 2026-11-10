#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;

// Distinct, saturated color per cube (golden-ratio hue spacing)
vec3 instanceColor(uint id)
{
    float hue = fract(float(id) * 0.61803398875);
    vec3 rgb = clamp(abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(1.0), rgb, 0.8);
}

void main()
{
    // Flat color, no shading yet
    hitValue = instanceColor(gl_InstanceCustomIndexEXT);
}
