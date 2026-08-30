#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float2 worldPos;
    float2 center;
};

struct Uniforms {
    float radius;
};

struct VertexOut {
    float4 position [[position]];
    float2 worldPos;
    float2 center;
    float4 color;
};

vertex VertexOut vertex_main(const device Vertex *vertices [[buffer(0)]],
                             uint vertex_id [[vertex_id]]) {
    VertexOut out;
    out.position = float4(vertices[vertex_id].position, 0.0f, 1.0f);
    out.worldPos = vertices[vertex_id].worldPos;
    out.center = vertices[vertex_id].center;
    out.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],const device Uniforms& uniforms [[buffer(1)]]) {
    float d = distance(in.worldPos, in.center);
    if (d > uniforms.radius)
        return float4(0.0, 0.0, 0.0, 0.0);
    if (d > uniforms.radius - uniforms.radius * 0.2f)
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    return in.color;
}
