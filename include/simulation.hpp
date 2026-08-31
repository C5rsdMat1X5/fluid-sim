#pragma once

#include <Metal.hpp>
#include <shader_types.hpp>

constexpr int kParticleCount = 8192 / 2;
constexpr int kSimulationIterations = 16;
constexpr int kMaxTicksPerFrame = 4;

struct RenderBuffers {
    MTL::Buffer *particleBufferA;
    MTL::Buffer *particleBufferB;
    MTL::Buffer *uniformsBuffer;
    uint32_t particleCount;
};

RenderBuffers initSimulation(MTL::Device *device, int particleCount);
const RenderBuffers &stepSimulation();
int simulationStepsToRun();
