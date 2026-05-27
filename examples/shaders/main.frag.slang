static const uint NO_TEXTURE = 4294967295;

struct Material
{
  float4 baseColourFactor;
  float3 emissiveFactor;
  uint   baseColourTextureID;
  uint   normalTextureID;
  uint   metallicRoughnessTextureID;
  uint   aoTextureID;
};

struct Vertex
{
  float3 position;
  float3 normal;
  float2 uv;
};

struct Registers
{
  float4x4 mvp;
  Vertex*  vertexBuffer;
  Material* materialBuffer;
};

struct FragmentInput
{
  float2 uv : TEXCOORD0;
};

[[vk::push_constant]]
ConstantBuffer<Registers> registers;

[[vk::binding(0, 0)]]
Sampler2D textures[];

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0
{
  Material material = *registers.materialBuffer;
  float4 baseColor = material.baseColourFactor;

  if(material.baseColourTextureID != NO_TEXTURE)
  {
    baseColor *= textures[material.baseColourTextureID].Sample(input.uv);
  }

  return baseColor;
}
