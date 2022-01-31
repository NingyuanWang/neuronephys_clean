#version 450 core

out vec4 color;

layout (location=2) uniform mat4 pMat;
layout (location=3) uniform mat4 vMat;
layout (location=4) uniform mat4 mMat;
uniform int render_lines;
uniform float color_divisor;
uniform float color_off;
uniform float color_cutoff;

in VS_OUT
{   
    vec3 position;
    flat float voltage;
} fs_in;

void main(void)
{
    if(render_lines == 1)
    {
        float t = fs_in.voltage;
        if(t < 0.5)
            color = vec4(0.9, 0.0, 0.0, 0.9);
        else if(t < 1.5)
            color = vec4(0.0, 0.0, 0.9, 0.9);
        else if(t < 2.5)
            color = vec4(0.0, 0.9, 0.9, 0.9);
        else if(t < 3.5)
            color = vec4(0.0, 0.9, 0.0, 0.9);
        else 
            color = vec4(0.9, 0.9, 0.9, 0.9);
        return;
    }
    float t = (fs_in.voltage - color_off) / color_divisor;
    t = clamp(t, 0, 1);
    t = 1 - t;
    if(t < color_cutoff)
        discard;
    color = mix(vec4(1.0, 0.0, 1.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), (t - color_cutoff) / (1 - color_cutoff));
}