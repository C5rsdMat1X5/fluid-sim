#pragma once

// Shared between C++ (simulation/renderer) and Metal shader code.
// Layouts must stay identical on both sides.

#ifdef __METAL_VERSION__
using vec2 = float2;
#else
#include <simd/simd.h>
using vec2 = simd::float2;
#endif

struct Particle {
    vec2 pos; // world space
    vec2 vel; // world space
};

struct Uniforms {
    float radius;
    vec2 particleScale;
    float subDt;
    float gravity;
    float maxSpeed;
    float maxSpeedSqr;
    float gridSize;
    float time;
    int numParticles;
    int windowWidth;
    int windowHeight;
};