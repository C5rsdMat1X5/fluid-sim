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
    uniforms.radius = kParticleRadius;
    uniforms.particleScale = {kParticleRadius / (0.5f * gWindowWidth),
                              kParticleRadius / (0.5f * gWindowHeight)};
    uniforms.subDt = kFixedDt / static_cast<float>(kSimulationIterations);
    uniforms.gravity = kGravity;
    uniforms.maxSpeed = kMaxSpeed;
    uniforms.maxSpeedSqr = kMaxSpeed * kMaxSpeed;
    uniforms.gridSize = kCellSize;
    uniforms.time = gState.simTime;
    uniforms.numParticles = gState.particleCount;
    uniforms.windowWidth = static_cast<int>(gWindowWidth);
    uniforms.windowHeight = static_cast<int>(gWindowHeight);
    return uniforms;
}

std::vector<Particle> layoutParticleGrid(int particleCount) {
    std::vector<Particle> particles(particleCount);

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

        particles[i].pos = {
            startX + static_cast<float>(gridX) * cellSize + cellSize * 0.5f,
            startY + static_cast<float>(gridY) * cellSize + cellSize * 0.5f};
        particles[i].vel = {0.0f, 0.0f};
    }

    return particles;
}

} // namespace

RenderBuffers initSimulation(MTL::Device *device, int particleCount) {
    gState = SimulationState{};
    gState.particleCount = particleCount;

    const std::vector<Particle> particles = layoutParticleGrid(particleCount);
    const std::size_t bufferSize = particles.size() * sizeof(Particle);

    MTL::Buffer *particleBufferA = device->newBuffer(
        particles.data(), bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer *particleBufferB = device->newBuffer(
        particles.data(), bufferSize, MTL::ResourceStorageModeShared);
    MTL::Buffer *uniformsBuffer =
        device->newBuffer(sizeof(Uniforms), MTL::ResourceStorageModeShared);

    gState.buffers = {particleBufferA, particleBufferB, uniformsBuffer,
                      static_cast<uint32_t>(particleCount)};
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