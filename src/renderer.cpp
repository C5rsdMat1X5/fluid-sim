#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTLFX_PRIVATE_IMPLEMENTATION

#include <Metal.hpp>
#include <renderer.hpp>
#include <vector>
#include <iostream>

static Renderer* gRenderer = nullptr;

Renderer::Renderer(MTL::Device* device, CA::MetalLayer* layer)
    : mDevice(device), mLayer(layer)
{
    mCommandQueue = mDevice->newCommandQueue();
    if (mCommandQueue == nullptr) {
        std::cout << "Failed to create command queue\n";
        return;
    }

    NS::Error* error = nullptr;
    MTL::Library* library = mDevice->newDefaultLibrary();
    if (library == nullptr) {
        std::cout << "Failed to load shader library\n";
        return;
    }

    MTL::Function* vertexFunction = library->newFunction(NS::String::string("vertex_main", NS::UTF8StringEncoding));
    MTL::Function* fragmentFunction = library->newFunction(NS::String::string("fragment_main", NS::UTF8StringEncoding));

    MTL::RenderPipelineDescriptor* descriptor = MTL::RenderPipelineDescriptor::alloc()->init();
    descriptor->setVertexFunction(vertexFunction);
    descriptor->setFragmentFunction(fragmentFunction);
    auto color = descriptor->colorAttachments()->object(0);
    color->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    color->setBlendingEnabled(true);
    color->setRgbBlendOperation(MTL::BlendOperationAdd);
    color->setAlphaBlendOperation(MTL::BlendOperationAdd);
    color->setSourceRGBBlendFactor(MTL::BlendFactorSourceAlpha);
    color->setSourceAlphaBlendFactor(MTL::BlendFactorSourceAlpha);
    color->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    color->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);

    mPipelineState = mDevice->newRenderPipelineState(descriptor, &error);
    if (mPipelineState == nullptr) {
        std::cout << "Failed to create render pipeline state: "
                   << error->localizedDescription()->utf8String() << "\n";
    }

    vertexFunction->release();
    fragmentFunction->release();
    descriptor->release();
    library->release();
}


Renderer::~Renderer()
{
    if (mPipelineState) mPipelineState->release();
    if (mCommandQueue) mCommandQueue->release();
}

void setRenderer(Renderer* renderer)
{
    gRenderer = renderer;
}

void renderFrame()
{
    if (!gRenderer) return;

    const std::vector<RenderBuffers>& buffers = stepSimulation();
    for (const RenderBuffers& buffer : buffers) {
        gRenderer->draw(buffer.vertexBuffer, buffer.indexBuffer, buffer.uniforms);
    }
}

void Renderer::draw(MTL::Buffer* vertexBuffer, MTL::Buffer* indicesBuffer, Uniforms uniform)
{
    if (!mLayer || !mPipelineState || !vertexBuffer || !indicesBuffer) return;

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    CA::MetalDrawable* drawable = mLayer->nextDrawable();
    if (drawable == nullptr) {
        pool->release();
        return;
    }

    MTL::RenderPassDescriptor* pass = MTL::RenderPassDescriptor::renderPassDescriptor();
    pass->colorAttachments()->object(0)->setTexture(drawable->texture());
    pass->colorAttachments()->object(0)->setLoadAction(MTL::LoadActionClear);
    pass->colorAttachments()->object(0)->setStoreAction(MTL::StoreActionStore);
    pass->colorAttachments()->object(0)->setClearColor(MTL::ClearColor(0.1, 0.1, 0.12, 1.0));

    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    MTL::RenderCommandEncoder* encoder = commandBuffer->renderCommandEncoder(pass);

    encoder->setRenderPipelineState(mPipelineState);
    encoder->setVertexBuffer(vertexBuffer, 0, 0);
    encoder->setFragmentBytes(&uniform, sizeof(uniform), 1);

    encoder->drawIndexedPrimitives(
        MTL::PrimitiveTypeTriangle,
        static_cast<NS::UInteger>(indicesBuffer->length() / sizeof(uint32_t)),
        MTL::IndexTypeUInt32,
        indicesBuffer,
        0
    );
    encoder->endEncoding();

    commandBuffer->presentDrawable(drawable);
    commandBuffer->commit();

    pool->release();
}
