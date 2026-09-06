#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTLFX_PRIVATE_IMPLEMENTATION

#include <Metal.hpp>
#include <algorithm>
#include <iostream>
#include <renderer.hpp>

namespace {
constexpr NS::UInteger kScanGroupSize = 256;
}

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

MTL::ComputePipelineState *Renderer::makeComputePipeline(MTL::Library *library,
                                                         const char *name) {
    MTL::Function *function =
        library->newFunction(NS::String::string(name, NS::UTF8StringEncoding));

    NS::Error *error = nullptr;
    MTL::ComputePipelineState *pipeline =
        mDevice->newComputePipelineState(function, &error);
    if (pipeline == nullptr) {
        std::cout << "Failed to create compute pipeline state '" << name
                  << "': " << error->localizedDescription()->utf8String()
                  << "\n";
    }

    function->release();
    return pipeline;
}

void Renderer::createComputePipeline(MTL::Library *library) {
    mPredictPipeline = makeComputePipeline(library, "k_predict");
    mScanLocalPipeline = makeComputePipeline(library, "k_scan_local");
    mScanBlocksPipeline = makeComputePipeline(library, "k_scan_blocks");
    mScanAddPipeline = makeComputePipeline(library, "k_scan_add");
    mScatterPipeline = makeComputePipeline(library, "k_scatter");
    mDensityPipeline = makeComputePipeline(library, "k_dens");
    mSolvePipeline = makeComputePipeline(library, "k_solve");
}

Renderer::~Renderer() {
    if (mPipelineState)
        mPipelineState->release();
    if (mPredictPipeline)
        mPredictPipeline->release();
    if (mScanLocalPipeline)
        mScanLocalPipeline->release();
    if (mScanBlocksPipeline)
        mScanBlocksPipeline->release();
    if (mScanAddPipeline)
        mScanAddPipeline->release();
    if (mScatterPipeline)
        mScatterPipeline->release();
    if (mDensityPipeline)
        mDensityPipeline->release();
    if (mSolvePipeline)
        mSolvePipeline->release();
    if (mCommandQueue)
        mCommandQueue->release();
}

void setRenderer(Renderer *renderer) { gRenderer = renderer; }

void renderFrame() {
    if (!gRenderer)
        return;
    gRenderer->drawFrame();
}

void Renderer::encodeStep(MTL::ComputeCommandEncoder *encoder,
                          bool currentIsA) {
    MTL::Buffer *inPos = currentIsA ? mBuffers.posA : mBuffers.posB;
    MTL::Buffer *inVel = currentIsA ? mBuffers.velA : mBuffers.velB;
    MTL::Buffer *inDens = currentIsA ? mBuffers.densA : mBuffers.densB;
    MTL::Buffer *inId = currentIsA ? mBuffers.idA : mBuffers.idB;
    MTL::Buffer *outPos = currentIsA ? mBuffers.posB : mBuffers.posA;
    MTL::Buffer *outVel = currentIsA ? mBuffers.velB : mBuffers.velA;
    MTL::Buffer *outDens = currentIsA ? mBuffers.densB : mBuffers.densA;
    MTL::Buffer *outId = currentIsA ? mBuffers.idB : mBuffers.idA;

    const uint32_t n = mBuffers.particleCount;
    const uint32_t histN = static_cast<uint32_t>(mBuffers.numCells + 1);
    const uint32_t numBlocks = static_cast<uint32_t>(mBuffers.numScanBlocks);

    auto particleGroupSize = [&](MTL::ComputePipelineState *pipeline) {
        NS::UInteger size = pipeline->maxTotalThreadsPerThreadgroup();
        size = std::min<NS::UInteger>(size, 256);
        size = std::min<NS::UInteger>(size, n);
        return MTL::Size(size, 1, 1);
    };
    const MTL::Size particleGrid(n, 1, 1);

    encoder->setComputePipelineState(mPredictPipeline);
    encoder->setBuffer(inPos, 0, 0);
    encoder->setBuffer(inVel, 0, 1);
    encoder->setBuffer(inId, 0, 2);
    encoder->setBuffer(mBuffers.predPos, 0, 3);
    encoder->setBuffer(mBuffers.cellOf, 0, 4);
    encoder->setBuffer(mBuffers.counts, 0, 5);
    encoder->setBuffer(mBuffers.uniformsBuffer, 0, 6);
    encoder->dispatchThreads(particleGrid, particleGroupSize(mPredictPipeline));

    const MTL::Size scanGroup(kScanGroupSize, 1, 1);
    const NS::UInteger scanGroups =
        (histN + kScanGroupSize - 1) / kScanGroupSize;

    encoder->setComputePipelineState(mScanLocalPipeline);
    encoder->setBuffer(mBuffers.counts, 0, 0);
    encoder->setBuffer(mBuffers.cellStart, 0, 1);
    encoder->setBuffer(mBuffers.blockSums, 0, 2);
    encoder->setBytes(&histN, sizeof(histN), 3);
    encoder->dispatchThreadgroups(MTL::Size(scanGroups, 1, 1), scanGroup);

    encoder->setComputePipelineState(mScanBlocksPipeline);
    encoder->setBuffer(mBuffers.blockSums, 0, 0);
    encoder->setBytes(&numBlocks, sizeof(numBlocks), 1);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1), scanGroup);

    encoder->setComputePipelineState(mScanAddPipeline);
    encoder->setBuffer(mBuffers.cellStart, 0, 0);
    encoder->setBuffer(mBuffers.blockSums, 0, 1);
    encoder->setBytes(&histN, sizeof(histN), 2);
    encoder->dispatchThreadgroups(MTL::Size(scanGroups, 1, 1), scanGroup);

    encoder->setComputePipelineState(mScatterPipeline);
    encoder->setBuffer(inPos, 0, 0);
    encoder->setBuffer(mBuffers.predPos, 0, 1);
    encoder->setBuffer(inId, 0, 2);
    encoder->setBuffer(mBuffers.cellOf, 0, 3);
    encoder->setBuffer(mBuffers.cellStart, 0, 4);
    encoder->setBuffer(mBuffers.counts, 0, 5);
    encoder->setBuffer(mBuffers.sortedOld, 0, 6);
    encoder->setBuffer(mBuffers.sortedPred, 0, 7);
    encoder->setBuffer(outId, 0, 8);
    encoder->setBuffer(mBuffers.uniformsBuffer, 0, 9);
    encoder->dispatchThreads(particleGrid, particleGroupSize(mScatterPipeline));

    encoder->setComputePipelineState(mDensityPipeline);
    encoder->setBuffer(mBuffers.sortedPred, 0, 0);
    encoder->setBuffer(mBuffers.cellStart, 0, 1);
    encoder->setBuffer(outDens, 0, 2);
    encoder->setBuffer(mBuffers.uniformsBuffer, 0, 3);
    encoder->dispatchThreads(particleGrid, particleGroupSize(mDensityPipeline));

    encoder->setComputePipelineState(mSolvePipeline);
    encoder->setBuffer(mBuffers.sortedOld, 0, 0);
    encoder->setBuffer(mBuffers.sortedPred, 0, 1);
    encoder->setBuffer(mBuffers.cellStart, 0, 2);
    encoder->setBuffer(outPos, 0, 3);
    encoder->setBuffer(outVel, 0, 4);
    encoder->setBuffer(outDens, 0, 5);
    encoder->setBuffer(mBuffers.uniformsBuffer, 0, 6);
    encoder->dispatchThreads(particleGrid, particleGroupSize(mSolvePipeline));
}

void Renderer::encodeComputePass(MTL::CommandBuffer *commandBuffer) {
    MTL::ComputePassDescriptor *cPass =
        MTL::ComputePassDescriptor::computePassDescriptor();
    MTL::ComputeCommandEncoder *computeEncoder =
        commandBuffer->computeCommandEncoder(cPass);

    const int stepsToRun = simulationStepsToRun();
    for (int i = 0; i < stepsToRun; ++i) {
        encodeStep(computeEncoder, mCurrentIsA);
        mCurrentIsA = !mCurrentIsA;
    }
    computeEncoder->endEncoding();
}

void Renderer::encodeRenderPass(MTL::CommandBuffer *commandBuffer,
                                CA::MetalDrawable *drawable) {
    MTL::Buffer *latestPos = mCurrentIsA ? mBuffers.posA : mBuffers.posB;
    MTL::Buffer *latestVel = mCurrentIsA ? mBuffers.velA : mBuffers.velB;

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
    encoder->setVertexBuffer(latestPos, 0, 0);
    encoder->setVertexBuffer(latestVel, 0, 1);
    encoder->setVertexBuffer(mBuffers.uniformsBuffer, 0, 2);

    encoder->drawPrimitives(MTL::PrimitiveTypeTriangle, 0, 3,
                            mBuffers.particleCount);
    encoder->endEncoding();
}

void Renderer::drawFrame() {
    if (!mLayer || !mPipelineState || !mPredictPipeline || !mDensityPipeline || !mSolvePipeline ||
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
