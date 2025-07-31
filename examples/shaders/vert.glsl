#version 450
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) out vec2 outUV;

struct Vertex {
    vec4 position;
    vec2 uv;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer
{
    Vertex vertices[];
};

layout(scalar, push_constant) uniform Registers {
    mat4 mvp;
    VertexBuffer vertexBuffer;
    uint textureId;
} registers;

void main() {
    Vertex vertex = registers.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position = registers.mvp * vertex.position;
    outUV = vertex.uv;
}
