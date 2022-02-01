#version 450 core

layout (location=0) in vec3 position;
layout (location=1) in float voltage;

layout (location=2) uniform mat4 pMat;
layout (location=3) uniform mat4 vMat;
layout (location=4) uniform mat4 mMat;
uniform int render_lines;

out VS_OUT
{   
    vec3 position;
    smooth float voltage;
} vs_out;

void main(void)
{
    gl_Position = pMat*vMat*mMat*vec4(position, 1.0);
    vs_out.position = position; 

    if(render_lines == 1)
        vs_out.voltage = gl_VertexID % 5;
    else
        vs_out.voltage = voltage;
}
