struct Vertex
{
  float3 position;
  float3 normal;
  float2 uv;
};

struct Material
{
  float4 baseColourFactor;
  float3 emissiveFactor;
  uint   baseColourTextureID;
  uint   normalTextureID;
  uint   metallicRoughnessTextureID;
  uint   aoTextureID;
};

struct Registers
{
  float4x4 mvp;
  Vertex*  vertexBuffer;
  Material* materialBuffer;
};

struct VertexOutput
{
  float4 position : SV_Position;
  float2 uv : TEXCOORD0;
};

[[vk::push_constant]]
ConstantBuffer<Registers> registers;

[shader("vertex")]
VertexOutput main(uint vertexId : SV_VertexID)
{
  Vertex vertex = registers.vertexBuffer[vertexId];

  VertexOutput output;
  output.position = mul(float4(vertex.position, 1.0), registers.mvp);
  output.uv = vertex.uv;
  return output;
}
