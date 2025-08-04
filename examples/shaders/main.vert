#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) out vec2 outUV;

struct Vertex {
    vec4 position;
    vec4 normal;
    vec2 uv;
};

layout(std430, buffer_reference, buffer_reference_align = 16) readonly buffer VertexBuffer
{
    Vertex vertices[];
};

struct Material {
    vec4 baseColourFactor;
    uint baseColourTextureID;
    uint normalTextureID;
    uint metallicRoughnessTextureID;
    uint aoTextureID;
};

layout(scalar, buffer_reference, buffer_reference_align = 16) readonly buffer MaterialBuffer
{
    Material material;
};

layout(scalar, push_constant) uniform Registers {
    mat4 mvp;
    VertexBuffer vertexBuffer;
    MaterialBuffer materialBuffer;
} registers;

void main() {
    Vertex vertex = registers.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position = registers.mvp * vertex.position;
    outUV = vertex.uv;
}
