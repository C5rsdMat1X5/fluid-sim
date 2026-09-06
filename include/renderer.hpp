#pragma once

#include <Metal.hpp>
#include <simulation.hpp>

class Renderer {
  public:
    Renderer(MTL::Device *device, CA::MetalLayer *layer);
    ~Renderer();

    void drawFrame();

  private:
    void createRenderPipeline(MTL::Library *library);
    void createComputePipeline(MTL::Library *library);
    MTL::ComputePipelineState *makeComputePipeline(MTL::Library *library,
                                                   const char *name);

    void encodeComputePass(MTL::CommandBuffer *commandBuffer);
    void encodeRenderPass(MTL::CommandBuffer *commandBuffer,
                          CA::MetalDrawable *drawable);
    void encodeStep(MTL::ComputeCommandEncoder *encoder, bool currentIsA);

    MTL::Device *mDevice = nullptr;
    MTL::CommandQueue *mCommandQueue = nullptr;
    MTL::RenderPipelineState *mPipelineState = nullptr;

    MTL::ComputePipelineState *mPredictPipeline = nullptr;
    MTL::ComputePipelineState *mScanLocalPipeline = nullptr;
    MTL::ComputePipelineState *mScanBlocksPipeline = nullptr;
    MTL::ComputePipelineState *mScanAddPipeline = nullptr;
    MTL::ComputePipelineState *mScatterPipeline = nullptr;
    MTL::ComputePipelineState *mDensityPipeline = nullptr;
    MTL::ComputePipelineState *mSolvePipeline = nullptr;

    CA::MetalLayer *mLayer = nullptr;

    RenderBuffers mBuffers{};
    bool mCurrentIsA = true;
};

void setRenderer(Renderer *renderer);
void renderFrame();
