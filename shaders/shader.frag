#version 450
#extension GL_KHR_vulkan_glsl: enable

layout(constant_id = 0) const int NUM_WAVES = 30;

layout(location = 0) out vec4 FragColor;

layout(location = 0) in vec4 color;
layout(location = 1) in vec4 pos;

struct Wave {
    float amp;
    float phase;
    float dir_x;
    float dir_z;
    float freq;
    float sharpness;
};

layout(binding = 0) uniform WavesUBO {
    Wave waves[NUM_WAVES];
} waves;

layout( push_constant ) uniform constants
{
    float time;
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
} PC;

vec3 sun_direction = normalize(vec3(0.1, 1.0, 0.3));
vec4 water_color = vec4(0.05,0.23,0.70,1.0);
vec4 specular_color = vec4(0.8,0.8,0.9,1.0);

void main()
{
  vec3 normal = vec3(0);
  
  for (int i = 0; i < NUM_WAVES; i++) {
      Wave wave = waves.waves[i];
      float wave_coord = wave.dir_x * pos.x + wave.dir_z * pos.z;
      float angle = wave.freq * wave_coord + wave.phase * PC.time;
      float d = wave.amp * exp(sin(angle)*wave.sharpness) * cos(angle) * wave.freq * wave.sharpness;
      float dx = d*wave.dir_x;
      float dz = d*wave.dir_z;

      normal += vec3(-dx, 0.0, -dz);
  }
  normal.y = 1.0;
  normal = normalize(normal);
  
  float diffuse = max(dot(normal, sun_direction), 0.0);

  vec3 view_dir = normalize(PC.cameraPos.xyz-pos.xyz);
  vec3 halfway =  normalize(sun_direction + view_dir);

  float specular = pow(max(dot(normal, halfway), 0.0),3);

  FragColor = diffuse * water_color + specular_color * specular;
  //FragColor = water_color;
}