#pragma once

#include <Metal.hpp>
#include <shader_types.hpp>

constexpr int kParticleCount = 500000;
constexpr int kSimulationIterations = 18;
constexpr int kMaxTicksPerFrame = 4;
constexpr float kParticleRadius = 0.5f;
constexpr float kCellSize = kParticleRadius * 2.0f;
constexpr float kParticlePadding = 0.05f;
constexpr float kMaxSpeed = 2000.0f;
constexpr float kGravity = 98.1f;
constexpr float kFixedDt = 1.0f / 60.0f;

struct RenderBuffers {

    MTL::Buffer *posA;
    MTL::Buffer *velA;
    MTL::Buffer *idA;
    MTL::Buffer *posB;
    MTL::Buffer *velB;
    MTL::Buffer *idB;

    MTL::Buffer *predPos;
    MTL::Buffer *cellOf;
    MTL::Buffer *counts;
    MTL::Buffer *cellStart;
    MTL::Buffer *blockSums;
    MTL::Buffer *sortedOld;
    MTL::Buffer *sortedPred;

    MTL::Buffer *uniformsBuffer;
    uint32_t particleCount;
    int gridW;
    int gridH;
    int numCells;
    int numScanBlocks;
};

RenderBuffers initSimulation(MTL::Device *device, int particleCount);
const RenderBuffers &stepSimulation();
int simulationStepsToRun();
