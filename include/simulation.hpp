#pragma once

#include <Metal.hpp>
#include <shader_types.hpp>

constexpr int kParticleCount = 20000;
constexpr int kSimulationIterations = 10;
constexpr int kMaxTicksPerFrame = 4;
constexpr float kParticleRadius = 1.0f;
constexpr float kParticleSmoothRad = kParticleRadius * 8.0f;
constexpr float kCellSize = kParticleRadius * 2.0f;
constexpr float kParticlePadding = 0.01f;
constexpr float kMaxSpeed = 2000.0f;
constexpr float kGravity = 981.0f;
constexpr float kFixedDt = 1.0f / 120.0f;

struct RenderBuffers {

    MTL::Buffer *posA;
    MTL::Buffer *velA;
    MTL::Buffer *densA;
    MTL::Buffer *idA;

    MTL::Buffer *posB;
    MTL::Buffer *velB;
    MTL::Buffer *densB;
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
