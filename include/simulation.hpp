#pragma once

#include <Metal.hpp>
#include <shader_types.hpp>

constexpr int kParticleCount = 8192;
constexpr int kSimulationIterations = 4;
constexpr int kMaxTicksPerFrame = 4;
constexpr float kParticleRadius = 3.0f;
constexpr float kCellSize = kParticleRadius * 2.0f;
constexpr float kParticlePadding = 1.0f;
constexpr float kMaxSpeed = 2000.0f;
constexpr float kGravity = 98.1f;
constexpr float kFixedDt = 1.0f / 60.0f;

struct RenderBuffers {
    MTL::Buffer *particleBufferA;
    MTL::Buffer *particleBufferB;
    MTL::Buffer *uniformsBuffer;
    uint32_t particleCount;
};

RenderBuffers initSimulation(MTL::Device *device, int particleCount);
const RenderBuffers &stepSimulation();
int simulationStepsToRun();
