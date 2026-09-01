#pragma once

#ifdef __METAL_VERSION__
using vec2 = float2;
#else
#include <simd/simd.h>
using vec2 = simd::float2;
#endif

struct Uniforms {

    vec2 particleScale;

    float radius;
    float subDt;
    float invDt;
    float gravity;
    float maxSpeed;
    float maxSpeedSqr;
    float time;

    float invCellSize;

    int numParticles;
    int gridW;
    int gridH;
    int numCells;
    int windowWidth;
    int windowHeight;
};