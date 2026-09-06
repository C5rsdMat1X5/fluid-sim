#include "../include/shader_types.hpp"
#include <metal_stdlib>

using namespace metal;

constant float2 kVertices[3] = {float2(-1.7320508f, -1.0f),
                                float2(1.7320508f, -1.0f), float2(0.0f, 2.0f)};

constant float kJitterAccel = 200.0f;
constant float kRelaxationFactor = 0.45f;
constant float kWallRestitution = 0.8f;
constant float kWallFriction = 0.98f;
constant float kDrag = 0.05f;
constant float kSpeedColorScale = 0.45f;

constant float kStiffness = 10000.0f;
constant float kTargetDensity = 1.5f;
constant float kPressureScale = 1000.0f;
constant float kViscosityScale = 0.01f;

constant float3 kRestColor = float3(0.38f, 0.40f, 0.98f);
constant float3 kMotionColor = float3(1.00f, 0.40f, 0.45f);

#define SCAN_TG 256

inline int2 cellCoord(float2 p, constant Uniforms &u) {
    float fx = clamp(p.x * u.invCellSize, 0.0f, float(u.gridW) - 1.0f);
    float fy = clamp(p.y * u.invCellSize, 0.0f, float(u.gridH) - 1.0f);
    int cx = clamp(int(fx), 0, int(u.gridW) - 1);
    int cy = clamp(int(fy), 0, int(u.gridH) - 1);
    return int2(cx, cy);
}

inline uint cellIndex(float2 p, constant Uniforms &u) {
    int2 c = cellCoord(p, u);
    return uint(c.y * u.gridW + c.x);
}

kernel void k_predict(const device float2 *posIn [[buffer(0)]],
                      const device float2 *velIn [[buffer(1)]],
                      const device uint *idIn [[buffer(2)]],
                      device float2 *predOut [[buffer(3)]],
                      device uint *cellOut [[buffer(4)]],
                      device atomic_uint *counts [[buffer(5)]],
                      constant Uniforms &u [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(u.numParticles))
        return;

    const float dt = u.subDt;

    float2 pos = posIn[gid];
    float2 vel = velIn[gid];
    const uint pid = idIn[gid];

    float phase = fract(float(pid) * 0.754877666f + u.time * 3.7f);

    vel.y += u.gravity * dt;
    vel.x += (phase * 2.0f - 1.0f) * kJitterAccel * dt;

    float speedSqr = dot(vel, vel);
    vel *= (speedSqr > u.maxSpeedSqr) ? (u.maxSpeed * rsqrt(speedSqr)) : 1.0f;

    float2 pred = fma(vel, float2(dt), pos);

    predOut[gid] = pred;

    uint c = cellIndex(pred, u);
    cellOut[gid] = c;
    atomic_fetch_add_explicit(&counts[c], 1u, memory_order_relaxed);
}

kernel void k_scan_local(const device uint *in [[buffer(0)]],
                         device uint *out [[buffer(1)]],
                         device uint *blockSums [[buffer(2)]],
                         constant uint &n [[buffer(3)]],
                         uint gid [[thread_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint bid [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]],
                         uint sgCount [[simdgroups_per_threadgroup]]) {
    threadgroup uint sgSums[32];

    uint v = (gid < n) ? in[gid] : 0u;

    uint pre = simd_prefix_exclusive_sum(v);
    uint sgTotal = simd_sum(v);

    if (lane == 0)
        sgSums[sgid] = sgTotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint offset = 0u;
    for (uint s = 0; s < sgid; ++s)
        offset += sgSums[s];

    if (gid < n)
        out[gid] = offset + pre;

    if (tid == 0) {
        uint total = 0u;
        for (uint s = 0; s < sgCount; ++s)
            total += sgSums[s];
        blockSums[bid] = total;
    }
}

kernel void k_scan_blocks(device uint *blockSums [[buffer(0)]],
                          constant uint &numBlocks [[buffer(1)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tgSize [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sgid [[simdgroup_index_in_threadgroup]],
                          uint sgCount [[simdgroups_per_threadgroup]]) {
    threadgroup uint sgSums[32];
    threadgroup uint running;

    if (tid == 0)
        running = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0; base < numBlocks; base += tgSize) {
        uint i = base + tid;
        uint v = (i < numBlocks) ? blockSums[i] : 0u;

        uint pre = simd_prefix_exclusive_sum(v);
        uint sgTotal = simd_sum(v);

        if (lane == 0)
            sgSums[sgid] = sgTotal;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint offset = 0u;
        for (uint s = 0; s < sgid; ++s)
            offset += sgSums[s];

        uint chunkTotal = 0u;
        for (uint s = 0; s < sgCount; ++s)
            chunkTotal += sgSums[s];

        uint value = running + offset + pre;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (i < numBlocks)
            blockSums[i] = value;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0)
            running += chunkTotal;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void k_scan_add(device uint *data [[buffer(0)]],
                       const device uint *blockOffsets [[buffer(1)]],
                       constant uint &n [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint bid [[threadgroup_position_in_grid]]) {
    if (gid >= n)
        return;
    data[gid] += blockOffsets[bid];
}

kernel void k_scatter(const device float2 *posIn [[buffer(0)]],
                      const device float2 *predIn [[buffer(1)]],
                      const device uint *idIn [[buffer(2)]],
                      const device uint *cellIn [[buffer(3)]],
                      const device uint *cellStart [[buffer(4)]],
                      device atomic_uint *counts [[buffer(5)]],
                      device float2 *sortedOld [[buffer(6)]],
                      device float2 *sortedPred [[buffer(7)]],
                      device uint *sortedId [[buffer(8)]],
                      constant Uniforms &u [[buffer(9)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(u.numParticles))
        return;

    uint c = cellIn[gid];
    uint remaining =
        atomic_fetch_sub_explicit(&counts[c], 1u, memory_order_relaxed);
    if (remaining == 0u)
        return;

    uint dst = cellStart[c] + remaining - 1u;
    if (dst >= uint(u.numParticles))
        return;

    sortedOld[dst] = posIn[gid];
    sortedPred[dst] = predIn[gid];
    sortedId[dst] = idIn[gid];
}

kernel void k_dens(const device float2 *sortedPred [[buffer(0)]],
                   const device uint *cellStart [[buffer(1)]],
                   device float *densOut [[buffer(2)]],
                   constant Uniforms &u [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(u.numParticles))
        return;

    const float2 self = sortedPred[gid];
    const int2 c = cellCoord(self, u);
    const int cellRadius = max(1, int(ceil(u.smoothRad * u.invCellSize)));

    const int x0 = max(c.x - cellRadius, 0);
    const int x1 = min(c.x + cellRadius, int(u.gridW) - 1);

    const int y0 = max(c.y - cellRadius, 0);
    const int y1 = min(c.y + cellRadius, int(u.gridH) - 1);

    float density = 0.0f;

    const float h = u.smoothRad;
    const float h2 = h * h;
    const float invH = 1.0f / h;
    const uint totalCells = uint(u.gridW * u.gridH);
    const uint numParticles = uint(u.numParticles);

    for (int y = y0; y <= y1; ++y) {
        const uint rowBase = uint(y * u.gridW);
        const uint begin = cellStart[rowBase + uint(x0)];
        const uint endIdx = rowBase + uint(x1) + 1u;
        const uint end =
            (endIdx < totalCells) ? cellStart[endIdx] : numParticles;

        if (begin >= end)
            continue;

        for (uint j = begin; j < end; ++j) {
            const float2 d = self - sortedPred[j];
            const float distSqr = dot(d, d);

            if (distSqr < h2) {
                const float dist = sqrt(distSqr);
                const float x = 1.0f - (dist * invH);
                const float x2 = x * x;
                density += x2 * x2;
            }
        }
    }

    densOut[gid] = max(density, 0.0001f);
}

kernel void k_solve(const device float2 *sortedOld [[buffer(0)]],
                    const device float2 *sortedPred [[buffer(1)]],
                    const device uint *cellStart [[buffer(2)]],
                    device float2 *posOut [[buffer(3)]],
                    device float2 *velOut [[buffer(4)]],
                    const device float *densOut [[buffer(5)]],
                    constant Uniforms &u [[buffer(6)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(u.numParticles))
        return;

    const float radius = u.radius;
    const float collisionDist = radius * 2.0f;
    const float collisionDistSqr = collisionDist * collisionDist;
    const float kCoeff = 0.5f * kRelaxationFactor;

    const float2 self = sortedPred[gid];
    float2 oldPos = sortedOld[gid];

    const int2 c = cellCoord(self, u);
    const int cellRadius = max(1, int(ceil(u.smoothRad * u.invCellSize)));

    const int x0 = max(c.x - cellRadius, 0);
    const int x1 = min(c.x + cellRadius, int(u.gridW) - 1);

    const int y0 = max(c.y - cellRadius, 0);
    const int y1 = min(c.y + cellRadius, int(u.gridH) - 1);

    const uint totalCells = uint(u.gridW * u.gridH);
    const uint numParticles = uint(u.numParticles);

    const float rhoI = max(densOut[gid], 0.0001f);
    const float pressureI = max(0.0f, kStiffness * (rhoI - kTargetDensity));
    const float termI = pressureI / (rhoI * rhoI);

    const float h = u.smoothRad;
    const float maxDist = max(h, collisionDist);
    const float maxDistSqr = maxDist * maxDist;

    const float2 vi = (sortedPred[gid] - sortedOld[gid]) * u.invDt;
    const float spikyConst =
        (-30.0f / (M_PI_F * h * h * h * h * h)) * kPressureScale;

    const float pi_const = 20.0f / (3.0f * M_PI_F * h * h * h * h * h) * kViscosityScale;

    float2 pressureAccel = float2(0.0f);
    float2 viscosityAccel = float2(0.0f);
    float2 correction = float2(0.0f);

    for (int y = y0; y <= y1; ++y) {
        const uint rowBase = uint(y * u.gridW);
        const uint begin = cellStart[rowBase + uint(x0)];
        const uint endIdx = rowBase + uint(x1) + 1u;
        const uint end =
            (endIdx < totalCells) ? cellStart[endIdx] : numParticles;

        if (begin >= end)
            continue;

        for (uint j = begin; j < end; ++j) {
            if (j == gid)
                continue;

            const float2 d = self - sortedPred[j];
            const float distSqr = dot(d, d);

            if (distSqr >= maxDistSqr)
                continue;

            const float invDist = rsqrt(distSqr);
            const float dist = distSqr * invDist;
            const float2 n = d * invDist;

            if (dist < h) {
                const float rhoJ = max(densOut[j], 0.0001f);
                const float pressureJ =
                    max(0.0f, kStiffness * (rhoJ - kTargetDensity));
                const float termJ = pressureJ / (rhoJ * rhoJ);

                const float hMinusR = h - dist;
                const float kernelGrad = spikyConst * (hMinusR * hMinusR);

                pressureAccel -= ((termI + termJ) * kernelGrad) * n;

                const float2 vj = (sortedPred[j] - sortedOld[j]) * u.invDt;
                const float viscosityLaplacian = pi_const * (h - dist);
        
                viscosityAccel +=
                    (vj - vi) /
                    rhoJ *
                    viscosityLaplacian;
            }

            if (distSqr < collisionDistSqr) {
                const float penetration = collisionDist - dist;
                correction += n * (penetration * kCoeff);
            }
        }
    }

    float corrLenSqr = dot(correction, correction);
    correction *= (corrLenSqr > collisionDistSqr)
                      ? (collisionDist * rsqrt(corrLenSqr))
                      : 1.0f;

    float2 pos = self + pressureAccel * (u.subDt * u.subDt) + correction + viscosityAccel;

    const float minX = radius;
    const float maxX = float(u.windowWidth) - radius;
    const float minY = radius;
    const float maxY = float(u.windowHeight) - radius;

    float cx = clamp(pos.x, minX, maxX);
    bool hitX = (cx != pos.x);
    oldPos.x = hitX ? (cx + (cx - oldPos.x) * kWallRestitution) : oldPos.x;
    pos.x = cx;
    pos.y = hitX ? (oldPos.y + (pos.y - oldPos.y) * kWallFriction) : pos.y;

    float cy = clamp(pos.y, minY, maxY);
    bool hitY = (cy != pos.y);
    oldPos.y = hitY ? (cy + (cy - oldPos.y) * kWallRestitution) : oldPos.y;
    pos.y = cy;
    pos.x = hitY ? (oldPos.x + (pos.x - oldPos.x) * kWallFriction) : pos.x;

    float2 vel = (pos - oldPos) * u.invDt;
    vel *= max(0.0f, 1.0f - kDrag * u.subDt);

    posOut[gid] = pos;
    velOut[gid] = vel;
}

struct VertexOut {
    float4 position [[position]];
    float2 localPosition;
    float speed;
};

vertex VertexOut vertex_main(const device float2 *positions [[buffer(0)]],
                             const device float2 *velocities [[buffer(1)]],
                             constant Uniforms &uniforms [[buffer(2)]],
                             uint vertexID [[vertex_id]],
                             uint pID [[instance_id]]) {
    float2 pos = positions[pID];
    float2 vel = velocities[pID];

    float2 invWindow = float2(2.0f / float(uniforms.windowWidth),
                              -2.0f / float(uniforms.windowHeight));
    float2 clipPos = fma(pos, invWindow, float2(-1.0f, 1.0f));

    VertexOut out;
    float2 localPos = kVertices[vertexID];
    out.position = float4(
        fma(localPos, float2(uniforms.particleScale), clipPos), 0.0f, 1.0f);
    out.localPosition = localPos;

    float maxSpeedNorm = max(uniforms.maxSpeed * kSpeedColorScale, 1e-4f);
    out.speed = clamp(length(vel) / maxSpeedNorm, 0.0f, 1.0f);
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]]) {
    float distSq = dot(in.localPosition, in.localPosition);
    float3 color = mix(kRestColor, kMotionColor, in.speed);
    float alpha = 1.0f - smoothstep(0.85f, 1.0f, distSq);
    return float4(color, alpha);
}