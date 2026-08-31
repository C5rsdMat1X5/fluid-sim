#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTLFX_PRIVATE_IMPLEMENTATION

#include <Metal.hpp>
#include <algorithm>
#include <iostream>
#include <renderer.hpp>

static Renderer *gRenderer = nullptr;

Renderer::Renderer(MTL::Device *device, CA::MetalLayer *layer)
    : mDevice(device), mLayer(layer) {
    mCommandQueue = mDevice->newCommandQueue();
    if (mCommandQueue == nullptr) {
        std::cout << "Failed to create command queue\n";
        return;
    }

    MTL::Library *library = mDevice->newDefaultLibrary();
    if (library == nullptr) {
        std::cout << "Failed to load shader library\n";
        return;
    }

    createRenderPipeline(library);
    createComputePipeline(library);

    mBuffers = initSimulation(device, kParticleCount);

    library->release();
}

void Renderer::createRenderPipeline(MTL::Library *library) {
    MTL::Function *vertexFunction = library->newFunction(
        NS::String::string("vertex_main", NS::UTF8StringEncoding));
    MTL::Function *fragmentFunction = library->newFunction(
        NS::String::string("fragment_main", NS::UTF8StringEncoding));

    MTL::RenderPipelineDescriptor *descriptor =
        MTL::RenderPipelineDescriptor::alloc()->init();
    descriptor->setVertexFunction(vertexFunction);
    descriptor->setFragmentFunction(fragmentFunction);
    auto color = descriptor->colorAttachments()->object(0);
    color->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    color->setBlendingEnabled(true);
    color->setRgbBlendOperation(MTL::BlendOperationAdd);
    color->setAlphaBlendOperation(MTL::BlendOperationAdd);
    color->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
    color->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    color->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    color->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);

    NS::Error *error = nullptr;
    mPipelineState = mDevice->newRenderPipelineState(descriptor, &error);
    if (mPipelineState == nullptr) {
        std::cout << "Failed to create render pipeline state: "
                  << error->localizedDescription()->utf8String() << "\n";
    }

    vertexFunction->release();
    fragmentFunction->release();
    descriptor->release();
}

void Renderer::createComputePipeline(MTL::Library *library) {
    MTL::Function *computeFunction = library->newFunction(
        NS::String::string("compute_kernel", NS::UTF8StringEncoding));

    NS::Error *error = nullptr;
    mComputePipelineState =
        mDevice->newComputePipelineState(computeFunction, &error);
    if (mComputePipelineState == nullptr) {
        std::cout << "Failed to create compute pipeline state: "
                  << error->localizedDescription()->utf8String() << "\n";
    }

    computeFunction->release();
}

Renderer::~Renderer() {
    if (mPipelineState)
        mPipelineState->release();
    if (mComputePipelineState)
        mComputePipelineState->release();
    if (mCommandQueue)
        mCommandQueue->release();
}

void setRenderer(Renderer *renderer) { gRenderer = renderer; }

void renderFrame() {
    if (!gRenderer)
        return;
    gRenderer->drawFrame();
}

void Renderer::encodeComputePass(MTL::CommandBuffer *commandBuffer) {
    MTL::ComputePassDescriptor *cPass =
        MTL::ComputePassDescriptor::computePassDescriptor();
    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder(cPass);

    computeEncoder->setComputePipelineState(mComputePipelineState);
    computeEncoder->setBuffer(mBuffers.uniformsBuffer, 0, 2);

    NS::UInteger threadGroupSize =
        mComputePipelineState->maxTotalThreadsPerThreadgroup();
    threadGroupSize = std::min<NS::UInteger>(threadGroupSize, 256);
    threadGroupSize =
        std::min<NS::UInteger>(threadGroupSize, mBuffers.particleCount);

    const MTL::Size gridSize(mBuffers.particleCount, 1, 1);
    const MTL::Size groupSize(threadGroupSize, 1, 1);

    const int stepsToRun = simulationStepsToRun();
    for (int i = 0; i < stepsToRun; ++i) {
        MTL::Buffer *inBuffer =
            mCurrentIsA ? mBuffers.particleBufferA : mBuffers.particleBufferB;
        MTL::Buffer *outBuffer =
            mCurrentIsA ? mBuffers.particleBufferB : mBuffers.particleBufferA;
        computeEncoder->setBuffer(inBuffer, 0, 0);
        computeEncoder->setBuffer(outBuffer, 0, 1);
        computeEncoder->dispatchThreads(gridSize, groupSize);
        mCurrentIsA = !mCurrentIsA;
    }
    computeEncoder->endEncoding();
}

void Renderer::encodeRenderPass(MTL::CommandBuffer *commandBuffer,
                                CA::MetalDrawable *drawable) {
    MTL::Buffer *latestParticles =
        mCurrentIsA ? mBuffers.particleBufferA : mBuffers.particleBufferB;

    MTL::RenderPassDescriptor *pass =
        MTL::RenderPassDescriptor::renderPassDescriptor();
    pass->colorAttachments()->object(0)->setTexture(drawable->texture());
    pass->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    pass->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
    pass->colorAttachments()->object(0)->setClearColor(
        MTL::ClearColor(0.1, 0.1, 0.12, 1.0));
    MTL::RenderCommandEncoder *encoder =
        commandBuffer->renderCommandEncoder(pass);

    encoder->setRenderPipelineState(mPipelineState);
    encoder->setVertexBuffer(latestParticles, 0, 0);
    encoder->setVertexBuffer(mBuffers.uniformsBuffer, 0, 1);

    encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3,
                            mBuffers.particleCount);
    encoder->endEncoding();
}

void Renderer::drawFrame() {
    if (!mLayer || !mPipelineState || !mComputePipelineState ||
        !mBuffers.particleCount)
        return;

    mBuffers = stepSimulation();

    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    CA::MetalDrawable *drawable = mLayer->nextDrawable();
    if (drawable == nullptr) {
        pool->release();
        return;
    }

    MTL::CommandBuffer *commandBuffer = mCommandQueue->commandBuffer();

    encodeComputePass(commandBuffer);
    encodeRenderPass(commandBuffer, drawable);

    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();

    pool->release();
}
