#pragma once

#include <Metal.hpp>
#include <simulation.hpp>

class Renderer {
public:
    Renderer(MTL::Device* device, CA::MetalLayer* layer);
    ~Renderer();

    void draw(MTL::Buffer* vertexBuffer, MTL::Buffer* indicesBuffer, Uniforms uniform);

private:
    MTL::Device* mDevice = nullptr;
    MTL::CommandQueue* mCommandQueue = nullptr;
    MTL::RenderPipelineState* mPipelineState = nullptr;
    CA::MetalLayer* mLayer = nullptr;
};

void setRenderer(Renderer* renderer);
void renderFrame();
