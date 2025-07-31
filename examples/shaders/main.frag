#version 450
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D textures[];

struct Vertex {
    vec3 position;
    vec3 normal;
    vec2 uv;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer
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

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MaterialBuffer
{
    Material material;
};

layout(scalar, push_constant) uniform Registers {
    mat4 mvp;
    VertexBuffer vertexBuffer;
    MaterialBuffer materialBuffer;
} registers;

void main() {
    Material material = registers.materialBuffer.material;
    outColor = texture(textures[nonuniformEXT(material.baseColourTextureID)], inUV);
}
