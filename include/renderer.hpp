#pragma once

#include <Metal.hpp>
#include <simulation.hpp>

class Renderer {
  public:
    Renderer(MTL::Device *device, CA::MetalLayer *layer);
    ~Renderer();

    void drawFrame();

  private:
    MTL::Device *mDevice = nullptr;
    MTL::CommandQueue *mCommandQueue = nullptr;
    MTL::RenderPipelineState *mPipelineState = nullptr;
    MTL::ComputePipelineState *mComputePipelineState = nullptr;
    CA::MetalLayer *mLayer = nullptr;

    RenderBuffers mBuffers{};
    bool mCurrentIsA = true;
};

void setRenderer(Renderer *renderer);
void renderFrame();
