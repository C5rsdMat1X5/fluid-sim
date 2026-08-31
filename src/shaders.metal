#include "../include/shader_types.hpp"
#include <metal_stdlib>

using namespace metal;

constant float2 kVertices[3] = {float2(-1.7320508f, -1.0f),
                                float2(1.7320508f, -1.0f), float2(0.0f, 2.0f)};

struct VertexOut {
    float4 position [[position]];
    float2 localPosition;
};

#define TILE_SIZE 256

kernel void compute_kernel(const device Particle *pIn [[buffer(0)]],
                           device Particle *pOut [[buffer(1)]],
                           constant Uniforms &uniforms [[buffer(2)]],
                           uint id [[thread_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tgSize [[threads_per_threadgroup]]) {
    threadgroup float2 sharedPos[TILE_SIZE];

    const int numParticles = uniforms.numParticles;
    const bool active = (id < uint(numParticles));

    Particle p;
    float2 oldPos = float2(0.0f);

    // Tiny per-particle lateral wobble. Pure normal-only collisions have no
    // friction/torque, so a perfectly vertical stack of particles is a valid
    // zero-horizontal-force equilibrium and never topples on its own; this
    // breaks that symmetry so tall narrow piles settle into a wider mound.
    const float kJitterAccel = 12.0f;
    const float jitterPhase =
        float(id) * 12.9898f + uniforms.time * 3.7f;
    const float jitterAccel = sin(jitterPhase) * kJitterAccel;

    if (active) {
        p = pIn[id];
        oldPos = p.pos;

        p.vel.y += uniforms.gravity * uniforms.subDt;
        p.vel.x += jitterAccel * uniforms.subDt;

        float speedSqr = dot(p.vel, p.vel);
        if (speedSqr > uniforms.maxSpeedSqr) {
            p.vel *= uniforms.maxSpeed * rsqrt(speedSqr);
        }

        p.pos += p.vel * uniforms.subDt;
    }

    const float radius = uniforms.radius;
    const float collisionDist = radius * 2.0f;
    const float collisionDistSqr = collisionDist * collisionDist;

    const float relaxationFactor = 0.45f;

    const int numTiles = (numParticles + TILE_SIZE - 1) / TILE_SIZE;
    float2 correction = float2(0.0f);

    for (int t = 0; t < numTiles; ++t) {
        int loadIdx = t * TILE_SIZE + int(tid);
        if (loadIdx < numParticles) {

            Particle other = pIn[loadIdx];
            other.vel.y += uniforms.gravity * uniforms.subDt;
            sharedPos[tid] = other.pos + other.vel * uniforms.subDt;
        } else {
            sharedPos[tid] = float2(1e8f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            int particlesInTile = min(TILE_SIZE, numParticles - t * TILE_SIZE);

#pragma unroll(4)
            for (int j = 0; j < particlesInTile; ++j) {
                int otherIdx = t * TILE_SIZE + j;
                if (otherIdx == int(id))
                    continue;

                float2 p2Pos = sharedPos[j];
                float2 d = p.pos - p2Pos;
                float distSqr = dot(d, d);

                if (distSqr < collisionDistSqr) {
                    float dist = sqrt(distSqr);
                    float2 n;

                    if (dist > 1e-5f) {
                        n = d / dist;
                    } else {

                        float angle =
                            float((id * 928371 + otherIdx * 12345) & 511) *
                            (6.283185f / 512.0f);
                        n = float2(cos(angle), sin(angle));
                        dist = 1e-5f;
                    }

                    float penetration = collisionDist - dist;

                    correction += n * (penetration * 0.5f * relaxationFactor);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        const float maxCorrection = collisionDist;
        float correctionLenSqr = dot(correction, correction);
        if (correctionLenSqr > maxCorrection * maxCorrection) {
            correction *= maxCorrection * rsqrt(correctionLenSqr);
        }
        p.pos += correction;

        const float minX = radius;
        const float maxX = float(uniforms.windowWidth) - radius;
        const float minY = radius;
        const float maxY = float(uniforms.windowHeight) - radius;
        const float wallFriction = 0.98f;

        if (p.pos.x < minX) {
            p.pos.x = minX;
            oldPos.x = p.pos.x + (p.pos.x - oldPos.x) * 0.5f;
        } else if (p.pos.x > maxX) {
            p.pos.x = maxX;
            oldPos.x = p.pos.x + (p.pos.x - oldPos.x) * 0.5f;
        }

        if (p.pos.y < minY) {
            p.pos.y = minY;
            oldPos.y = p.pos.y + (p.pos.y - oldPos.y) * 0.5f;
        } else if (p.pos.y > maxY) {
            p.pos.y = maxY;
            oldPos.y = p.pos.y + (p.pos.y - oldPos.y) * 0.5f;
            p.pos.x = oldPos.x + (p.pos.x - oldPos.x) * wallFriction;
        }

        p.vel = (p.pos - oldPos) / uniforms.subDt;

        const float drag = 0.05f;
        p.vel *= max(0.0f, 1.0f - drag * uniforms.subDt);

        pOut[id] = p;
    }
}

vertex VertexOut vertex_main(const device Particle *particles [[buffer(0)]],
                             constant Uniforms &uniforms [[buffer(1)]],
                             uint vertexID [[vertex_id]],
                             uint pID [[instance_id]]) {
    Particle p = particles[pID];
    float2 invWindow = float2(2.0f / float(uniforms.windowWidth),
                              -2.0f / float(uniforms.windowHeight));
    float2 clipPos = fma(p.pos, invWindow, float2(-1.0f, 1.0f));

    VertexOut out;
    float2 localPos = kVertices[vertexID];
    out.position = float4(
        fma(localPos, uniforms.particleScale, clipPos), 0.0f, 1.0f);
    out.localPosition = localPos;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    if (dot(in.localPosition, in.localPosition) > 1.0f)
        discard_fragment();

    return float4(0.2f, 0.6f, 1.0f, 1.0f);
}