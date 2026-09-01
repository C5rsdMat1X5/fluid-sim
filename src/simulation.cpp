#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <simulation.hpp>
#include <vector>
#include <window.hpp>

namespace {
struct SimulationState {
    RenderBuffers buffers{};
    int particleCount = 0;
    float timeAccumulator = 0.0f;
    int stepsToRun = 0;
    float simTime = 0.0f;
};

SimulationState gState;
std::chrono::steady_clock::time_point gLastTime =
    std::chrono::steady_clock::now();

float getFrameDt() {
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<float> diff = now - gLastTime;
    gLastTime = now;

    return std::min(diff.count(), 0.1f);
}

int advanceFixedTimestep(float frameDt) {
    gState.timeAccumulator += frameDt;

    int ticks = 0;
    while (gState.timeAccumulator >= kFixedDt && ticks < kMaxTicksPerFrame) {
        gState.timeAccumulator -= kFixedDt;
        ++ticks;
    }
    gState.simTime += static_cast<float>(ticks) * kFixedDt;
    return ticks;
}

Uniforms buildUniforms() {
    Uniforms uniforms{};
    uniforms.particleScale = {kParticleRadius / (0.5f * gWindowWidth),
                              kParticleRadius / (0.5f * gWindowHeight)};
    uniforms.radius = kParticleRadius;
    uniforms.subDt = kFixedDt / static_cast<float>(kSimulationIterations);
    uniforms.invDt = 1.0f / uniforms.subDt;
    uniforms.gravity = kGravity;
    uniforms.maxSpeed = kMaxSpeed;
    uniforms.maxSpeedSqr = kMaxSpeed * kMaxSpeed;
    uniforms.time = gState.simTime;
    uniforms.invCellSize = 1.0f / kCellSize;
    uniforms.numParticles = gState.particleCount;
    uniforms.gridW = gState.buffers.gridW;
    uniforms.gridH = gState.buffers.gridH;
    uniforms.numCells = gState.buffers.numCells;
    uniforms.windowWidth = static_cast<int>(gWindowWidth);
    uniforms.windowHeight = static_cast<int>(gWindowHeight);
    return uniforms;
}

void layoutParticleGrid(int particleCount, std::vector<vec2> &pos,
                        std::vector<vec2> &vel, std::vector<uint32_t> &ids) {
    pos.resize(particleCount);
    vel.resize(particleCount);
    ids.resize(particleCount);

    int squareSide = static_cast<int>(
        std::ceil(std::sqrt(static_cast<float>(particleCount))));
    if (squareSide < 1)
        squareSide = 1;

    const float cellSize = kParticleRadius * 2.0f + kParticlePadding;
    const float gridWidth = static_cast<float>(squareSide) * cellSize;
    const float startX = (gWindowWidth - gridWidth) * 0.5f;
    const float startY = (gWindowHeight - gridWidth) * 0.5f;

    for (int i = 0; i < particleCount; ++i) {
        const int gridX = i % squareSide;
        const int gridY = i / squareSide;

        pos[i] = {
            startX + static_cast<float>(gridX) * cellSize + cellSize * 0.5f,
            startY + static_cast<float>(gridY) * cellSize + cellSize * 0.5f};
        vel[i] = {0.0f, 0.0f};
        ids[i] = static_cast<uint32_t>(i);
    }
}

MTL::Buffer *makeBuffer(MTL::Device *device, const void *data,
                        std::size_t size) {
    return device->newBuffer(data, size, MTL::ResourceStorageModeShared);
}

MTL::Buffer *makeZeroedBuffer(MTL::Device *device, std::size_t size) {
    MTL::Buffer *buffer =
        device->newBuffer(size, MTL::ResourceStorageModeShared);
    std::memset(buffer->contents(), 0, size);
    return buffer;
}

} // namespace

RenderBuffers initSimulation(MTL::Device *device, int particleCount) {
    gState = SimulationState{};
    gState.particleCount = particleCount;

    std::vector<vec2> pos, vel;
    std::vector<uint32_t> ids;
    layoutParticleGrid(particleCount, pos, vel, ids);

    const std::size_t vec2BufSize = particleCount * sizeof(vec2);
    const std::size_t idBufSize = particleCount * sizeof(uint32_t);

    RenderBuffers buffers{};
    buffers.posA = makeBuffer(device, pos.data(), vec2BufSize);
    buffers.velA = makeBuffer(device, vel.data(), vec2BufSize);
    buffers.idA = makeBuffer(device, ids.data(), idBufSize);
    buffers.posB = makeBuffer(device, pos.data(), vec2BufSize);
    buffers.velB = makeBuffer(device, vel.data(), vec2BufSize);
    buffers.idB = makeBuffer(device, ids.data(), idBufSize);

    buffers.gridW =
        std::max(1, static_cast<int>(std::ceil(gWindowWidth / kCellSize)));
    buffers.gridH =
        std::max(1, static_cast<int>(std::ceil(gWindowHeight / kCellSize)));
    buffers.numCells = buffers.gridW * buffers.gridH;

    const std::size_t histBufSize = (buffers.numCells + 1) * sizeof(uint32_t);
    constexpr int kScanGroupSize = 256;
    buffers.numScanBlocks =
        (buffers.numCells + 1 + kScanGroupSize - 1) / kScanGroupSize;

    buffers.predPos = device->newBuffer(vec2BufSize, MTL::ResourceStorageModeShared);
    buffers.cellOf = device->newBuffer(idBufSize, MTL::ResourceStorageModeShared);
    buffers.counts = makeZeroedBuffer(device, histBufSize);
    buffers.cellStart = makeZeroedBuffer(device, histBufSize);
    buffers.blockSums = makeZeroedBuffer(
        device, buffers.numScanBlocks * sizeof(uint32_t));
    buffers.sortedOld = device->newBuffer(vec2BufSize, MTL::ResourceStorageModeShared);
    buffers.sortedPred = device->newBuffer(vec2BufSize, MTL::ResourceStorageModeShared);

    buffers.uniformsBuffer =
        device->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);
    buffers.particleCount = static_cast<uint32_t>(particleCount);

    gState.buffers = buffers;
    gLastTime = std::chrono::steady_clock::now();
    return gState.buffers;
}

const RenderBuffers &stepSimulation() {
    const int ticks = advanceFixedTimestep(getFrameDt());
    gState.stepsToRun = ticks * kSimulationIterations;

    const Uniforms uniforms = buildUniforms();
    std::memcpy(gState.buffers.uniformsBuffer->contents(), &uniforms,
                sizeof(Uniforms));

    return gState.buffers;
}

int simulationStepsToRun() { return gState.stepsToRun; }
