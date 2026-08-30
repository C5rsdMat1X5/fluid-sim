#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <simulation.hpp>
#include <vector>
#include <window_state.hpp>

namespace {

struct Vertex {
    float position[2];
    float worldPos[2];
    float center[2];
};

struct Particle {
    std::size_t vertexIndex;
    float pos[2];
    float vel[2];
    float halfSize;
};

struct LayoutTransform {
    float padding;
    float cellSize;
    float worldWidth;
    float worldHeight;
};

struct SimulationState {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Particle> particles;
    std::vector<RenderBuffers> renderBuffers;
    LayoutTransform layout;
};

SimulationState gState;

float toClipX(float worldX, const LayoutTransform &layout) {
    return (worldX / layout.worldWidth) * 2.0f - 1.0f;
}

float toClipY(float worldY, const LayoutTransform &layout) {
    return 1.0f - (worldY / layout.worldHeight) * 2.0f;
}

LayoutTransform layoutParticleGrid(int particleCount) {
    gState.vertices.assign(particleCount * 4, Vertex{});
    gState.indices.assign(particleCount * 6, 0);
    gState.particles.clear();
    gState.particles.reserve(particleCount);

    int squareSide = static_cast<int>(std::ceil(std::sqrt(particleCount)));
    if (squareSide < 1) {
        squareSide = 1;
    }

    const float padding = 8.0f;
    const float cellSize = 10.0f;
    const float gridWidth = static_cast<float>(squareSide) * cellSize +
                            (static_cast<float>(squareSide) - 1.0f) * padding;
    const float gridHeight = gridWidth;
    const float startX = (gWindowWidth - gridWidth) * 0.5f;
    const float startY = (gWindowHeight - gridHeight) * 0.5f;

    LayoutTransform layout{
        padding,
        cellSize,
        gWindowWidth,
        gWindowHeight,
    };
    gState.layout = layout;

    for (int i = 0; i < particleCount; ++i) {
        const int gridX = i % squareSide;
        const int gridY = i / squareSide;

        const float left =
            startX + static_cast<float>(gridX) * (cellSize + padding);
        const float top =
            startY + static_cast<float>(gridY) * (cellSize + padding);
        const float centerX = left + cellSize * 0.5f;
        const float centerY = top + cellSize * 0.5f;

        const std::size_t vertexIndex = 4 * static_cast<std::size_t>(i);
        const float right = left + cellSize;
        const float bottom = top + cellSize;

        gState.vertices[vertexIndex + 0] = {
            {toClipX(left, layout), toClipY(top, layout)},
            {left, top},
            {centerX, centerY},
        };
        gState.vertices[vertexIndex + 1] = {
            {toClipX(left, layout), toClipY(bottom, layout)},
            {left, bottom},
            {centerX, centerY},
        };
        gState.vertices[vertexIndex + 2] = {
            {toClipX(right, layout), toClipY(top, layout)},
            {right, top},
            {centerX, centerY},
        };
        gState.vertices[vertexIndex + 3] = {
            {toClipX(right, layout), toClipY(bottom, layout)},
            {right, bottom},
            {centerX, centerY},
        };

        Particle particle;
        particle.vertexIndex = vertexIndex;
        particle.pos[0] = centerX;
        particle.pos[1] = centerY;
        particle.vel[0] = 10.0f;
        particle.vel[1] = 0.0f;
        particle.halfSize = cellSize * 0.5f;
        gState.particles.push_back(particle);

        const std::size_t indexBase = 6 * static_cast<std::size_t>(i);
        gState.indices[indexBase + 0] = static_cast<uint32_t>(vertexIndex + 0);
        gState.indices[indexBase + 1] = static_cast<uint32_t>(vertexIndex + 1);
        gState.indices[indexBase + 2] = static_cast<uint32_t>(vertexIndex + 2);
        gState.indices[indexBase + 3] = static_cast<uint32_t>(vertexIndex + 2);
        gState.indices[indexBase + 4] = static_cast<uint32_t>(vertexIndex + 1);
        gState.indices[indexBase + 5] = static_cast<uint32_t>(vertexIndex + 3);
    }

    return layout;
}

void syncParticleVertices(Particle &particle, Vertex *vertices,
                          const LayoutTransform &layout) {

    const float left = particle.pos[0] - particle.halfSize;
    const float right = particle.pos[0] + particle.halfSize;
    const float top = particle.pos[1] - particle.halfSize;
    const float bottom = particle.pos[1] + particle.halfSize;

    Vertex *corners = &vertices[particle.vertexIndex];
    corners[0] = {
        {toClipX(left, layout), toClipY(top, layout)},
        {left, top},
        {particle.pos[0], particle.pos[1]},
    };
    corners[1] = {
        {toClipX(left, layout), toClipY(bottom, layout)},
        {left, bottom},
        {particle.pos[0], particle.pos[1]},
    };
    corners[2] = {
        {toClipX(right, layout), toClipY(top, layout)},
        {right, top},
        {particle.pos[0], particle.pos[1]},
    };
    corners[3] = {
        {toClipX(right, layout), toClipY(bottom, layout)},
        {right, bottom},
        {particle.pos[0], particle.pos[1]},
    };
}

} // namespace

static auto lastTime = std::chrono::steady_clock::now();
float getDt() {
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<float> diff = now - lastTime;
    lastTime = now;
    return diff.count();
}

void initSimulation(MTL::Device *device, int particleCount) {
    gState = SimulationState{};

    const LayoutTransform layout = layoutParticleGrid(particleCount);

    MTL::Buffer *vertexBuffer = device->newBuffer(
        gState.vertices.data(), gState.vertices.size() * sizeof(Vertex),
        MTL::ResourceStorageModeShared);
    MTL::Buffer *indexBuffer = device->newBuffer(
        gState.indices.data(), gState.indices.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);

    gState.renderBuffers.push_back(
        {vertexBuffer, indexBuffer, Uniforms{layout.cellSize * 0.5f}});
}

void computeWallColisions(Particle &particle, float radius){
    float worldH = gState.layout.worldHeight;
    float worldW = gState.layout.worldWidth;

    const float friction = 0.65f;
    const float sleepVelocity = 2.0f;
    const float restitution = 0.95f;
    
    if (particle.pos[0] - radius < 0.0f) {
        particle.pos[0] = radius;
        if (std::abs(particle.vel[0]) < sleepVelocity) {
            particle.vel[0] = 0.0f;

        } else {
            particle.vel[0] = -particle.vel[0] * restitution;
        }
        particle.vel[1] *= friction;
    }

    if (particle.pos[0] + radius > worldW) {
        particle.pos[0] = worldW - radius;
        if (std::abs(particle.vel[0]) < sleepVelocity) {
            particle.vel[0] = 0.0f;
        } else {
            particle.vel[0] = -particle.vel[0] * restitution;
        }
        particle.vel[1] *= friction;
    }

    if (particle.pos[1] - radius < 0.0f) {
        particle.pos[1] = radius;
        if (std::abs(particle.vel[1]) < sleepVelocity) {
            particle.vel[1] = 0.0f;
        } else {
            particle.vel[1] = -particle.vel[1] * restitution;
        }
        particle.vel[0] *= friction;
    }

    if (particle.pos[1] + radius > worldH) {
        particle.pos[1] = worldH - radius;
        if (std::abs(particle.vel[1]) < sleepVelocity) {
            particle.vel[1] = 0.0f;
        } else {
            particle.vel[1] = -particle.vel[1] * restitution;
        }
        particle.vel[0] *= friction;
    }
}

void computeCollisions(Particle &particle, int id) {
    float radius = gState.renderBuffers[0].uniforms.radius;

    const float restitution = 0.85f;

    computeWallColisions(particle, radius);
    for (int i = 0; i < gState.particles.size(); i++) {
        if (i <= id)
            continue;

        Particle &p2 = gState.particles[i];

        float dx = p2.pos[0] - particle.pos[0];
        float dy = p2.pos[1] - particle.pos[1];
        float distanceSqr = dx * dx + dy * dy;

        const float minDistance = 0.0001f;
        const float collisionDistance = radius * 2.0f;
        if (distanceSqr <= collisionDistance * collisionDistance &&
            distanceSqr > minDistance * minDistance) {
            float distance = sqrt(distanceSqr);
            float nx = dx / distance;
            float ny = dy / distance;
            float rvx = p2.vel[0] - particle.vel[0];
            float rvy = p2.vel[1] - particle.vel[1];
            float velocityAlongNormal = rvx * nx + rvy * ny;
            if (velocityAlongNormal > 0)
                continue;
            //float impulse = -(2.0f * velocityAlongNormal) / (1.0f / 1 + 1.0f / 1);
            float impulse = -(1 + restitution) * velocityAlongNormal / (1 + 1);
            float impulseX = impulse * nx;
            float impulseY = impulse * ny;

            particle.vel[0] -= impulseX / 1;
            particle.vel[1] -= impulseY / 1;

            p2.vel[0] += impulseX / 1;
            p2.vel[1] += impulseY / 1;

            const float correctionPercent = 0.8f;
            float penetration = collisionDistance - distance;
            if (penetration > 0.0f) {
                float correction = penetration * correctionPercent;
                particle.pos[0] -= nx * correction;
                particle.pos[1] -= ny * correction;
                p2.pos[0] += nx * correction;
                p2.pos[1] += ny * correction;
            }
        }
    }
}

const std::vector<RenderBuffers> &stepSimulation() {
    // here do physics
    float dt = getDt();
    const int iterations = 10;
    gState.layout.worldWidth = gWindowWidth;
    gState.layout.worldHeight = gWindowHeight;
    const float subDt = dt / iterations;
    const float maxSpeed = 2000.0f;
    const float maxSqrt = sqrt(maxSpeed / 2);
    for (int it = 0; it < iterations; it++) {
        for (int i = 0; i < gState.particles.size(); i++) {
            Particle &particle = gState.particles[i];
            particle.vel[1] += 981.0f * subDt;
            computeCollisions(particle, i);

            if (particle.vel[0] > maxSqrt || particle.vel[1] > maxSqrt) {
                float speedSqr = particle.vel[0] * particle.vel[0] +
                                 particle.vel[1] * particle.vel[1];
                if (speedSqr > maxSpeed * maxSpeed) {
                    float scale = maxSpeed / std::sqrt(speedSqr);
                    particle.vel[0] *= scale;
                    particle.vel[1] *= scale;
                }
            }

            particle.pos[0] += particle.vel[0] * subDt;
            particle.pos[1] += particle.vel[1] * subDt;
        }
    }
    for (Particle &p : gState.particles) {
        syncParticleVertices(p, gState.vertices.data(), gState.layout);
    }

    if (!gState.renderBuffers.empty()) {
        MTL::Buffer *vertexBuffer = gState.renderBuffers[0].vertexBuffer;
        std::memcpy(vertexBuffer->contents(), gState.vertices.data(),
                    gState.vertices.size() * sizeof(Vertex));
    }

    return gState.renderBuffers;
}
