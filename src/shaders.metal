#include "../include/shader_types.hpp"
#include <metal_stdlib>

using namespace metal;

constant float2 kVertices[3] = {float2(-1.7320508f, -1.0f),
                                float2(1.7320508f, -1.0f), float2(0.0f, 2.0f)};

constant float kJitterAccel = 16.0f;
constant float kRelaxationFactor = 0.45f;
constant float kWallRestitution = 0.8f;
constant float kWallFriction = 0.98f;
constant float kDrag = 0.05f;
constant float kSpeedColorScale = 0.45f;

constant float3 kRestColor = float3(0.38f, 0.40f, 0.98f);
constant float3 kMotionColor = float3(1.00f, 0.40f, 0.45f);

#define TILE_SIZE 256

struct VertexOut {
    float4 position [[position]];
    float2 localPosition;
    float speed;
};

inline void integrateVelocity(thread Particle &p, uint id, float time, float dt,
                              float gravDt) {
    float phase = fract(float(id) * 0.754877666f + time * 3.7f);
    float jitterAccel = (phase * 2.0f - 1.0f) * kJitterAccel;

    p.vel.y += gravDt;
    p.vel.x += jitterAccel * dt;
}

inline void clampSpeed(thread Particle &p, float maxSpeed, float maxSpeedSqr) {
    float speedSqr = dot(p.vel, p.vel);
    if (speedSqr > maxSpeedSqr) {
        p.vel *= maxSpeed * rsqrt(speedSqr);
    }
}

inline float2 resolveNeighborCollisions(float2 selfPos,
                                        threadgroup const float2 *sharedPos,
                                        int particlesInTile, int selfTileIndex,
                                        bool isSelfTile, float collisionDist,
                                        float collisionDistSqr, float kCoeff) {
    float2 correction = float2(0.0f);

#pragma unroll(8)
    for (int j = 0; j < particlesInTile; ++j) {
        if (isSelfTile && j == selfTileIndex)
            continue;

        float2 d = selfPos - sharedPos[j];
        float distSqr = dot(d, d);

        if (distSqr < collisionDistSqr) {
            float invDist = rsqrt(max(distSqr, 1e-8f));
            float dist = distSqr * invDist;
            float penetration = collisionDist - dist;

            float2 n =
                (distSqr > 1e-8f) ? (d * invDist) : float2(0.7071f, 0.7071f);

            correction += n * (penetration * kCoeff);
        }
    }

    return correction;
}

inline void resolveWallCollisions(thread Particle &p, thread float2 &oldPos,
                                  float radius, float windowWidth,
                                  float windowHeight) {
    const float minX = radius;
    const float maxX = windowWidth - radius;
    const float minY = radius;
    const float maxY = windowHeight - radius;

    if (p.pos.x < minX) {
        p.pos.x = minX;
        oldPos.x = p.pos.x + (p.pos.x - oldPos.x) * kWallRestitution;
    } else if (p.pos.x > maxX) {
        p.pos.x = maxX;
        oldPos.x = p.pos.x + (p.pos.x - oldPos.x) * kWallRestitution;
    }

    if (p.pos.y < minY) {
        p.pos.y = minY;
        oldPos.y = p.pos.y + (p.pos.y - oldPos.y) * kWallRestitution;
    } else if (p.pos.y > maxY) {
        p.pos.y = maxY;
        oldPos.y = p.pos.y + (p.pos.y - oldPos.y) * kWallRestitution;
        p.pos.x = oldPos.x + (p.pos.x - oldPos.x) * kWallFriction;
    }
}

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

    const float dt = uniforms.subDt;
    const float gravDt = uniforms.gravity * dt;

    if (active) {
        p = pIn[id];
        oldPos = p.pos;

        integrateVelocity(p, id, uniforms.time, dt, gravDt);
        clampSpeed(p, uniforms.maxSpeed, uniforms.maxSpeedSqr);

        p.pos += p.vel * dt;
    }

    const float radius = uniforms.radius;
    const float collisionDist = radius * 2.0f;
    const float collisionDistSqr = collisionDist * collisionDist;
    const float maxCorrectionSqr = collisionDistSqr;
    const float kCoeff = 0.5f * kRelaxationFactor;

    const int numTiles = (numParticles + TILE_SIZE - 1) / TILE_SIZE;
    const int myTileIdx = int(id) / TILE_SIZE;
    float2 correction = float2(0.0f);

    const float2 gravStep = float2(0.0f, gravDt * dt);

    for (int t = 0; t < numTiles; ++t) {
        int loadIdx = t * TILE_SIZE + int(tid);
        if (loadIdx < numParticles) {
            Particle other = pIn[loadIdx];
            sharedPos[tid] = fma(other.vel, float2(dt), other.pos) + gravStep;
        } else {
            sharedPos[tid] = float2(1e8f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            const int particlesInTile =
                min(TILE_SIZE, numParticles - t * TILE_SIZE);
            const bool isSelfTile = (t == myTileIdx);

            correction += resolveNeighborCollisions(
                p.pos, sharedPos, particlesInTile, int(tid), isSelfTile,
                collisionDist, collisionDistSqr, kCoeff);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        float correctionLenSqr = dot(correction, correction);
        if (correctionLenSqr > maxCorrectionSqr) {
            correction *= collisionDist * rsqrt(correctionLenSqr);
        }
        p.pos += correction;

        resolveWallCollisions(p, oldPos, radius, float(uniforms.windowWidth),
                              float(uniforms.windowHeight));

        p.vel = (p.pos - oldPos) / dt;
        p.vel *= max(0.0f, 1.0f - kDrag * dt);

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
        fma(localPos, float2(uniforms.particleScale), clipPos), 0.0f, 1.0f);
    out.localPosition = localPos;

    float speed = length(p.vel);
    out.speed =
        clamp(speed / (uniforms.maxSpeed * kSpeedColorScale), 0.0f, 1.0f);

    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    float distSq = dot(in.localPosition, in.localPosition);

    if (distSq > 1.0f) {
        discard_fragment();
    }

    float3 color = mix(kRestColor, kMotionColor, in.speed);
    float alpha = smoothstep(1.0f, 0.85f, distSq);

    return float4(color, alpha);
}
