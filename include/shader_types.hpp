#pragma once


#ifdef __METAL_VERSION__
using vec2 = float2;
#else
#include <simd/simd.h>
using vec2 = simd::float2;
#endif

struct Particle {
    vec2 pos; 
    vec2 vel; 
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