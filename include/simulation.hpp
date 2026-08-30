#pragma once

#include <Metal.hpp>
#include <vector>

struct Uniforms {
    float radius;
};

struct RenderBuffers {
    MTL::Buffer *vertexBuffer;
    MTL::Buffer *indexBuffer;
    Uniforms uniforms;
};

void initSimulation(MTL::Device *device, int particleCount);
const std::vector<RenderBuffers> &stepSimulation();
