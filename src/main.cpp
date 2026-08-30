#include <iostream>
#include <renderer.hpp>

void* createWindow();
void runApplication();

int main() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (device == nullptr) {
        std::cout << "No Metal device found\n";
        return 1;
    }

    CA::MetalLayer* layer =
        static_cast<CA::MetalLayer*>(createWindow());

    layer->setDevice(device);
    layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm);
    layer->setFramebufferOnly(true);

    Renderer renderer(device, layer);
    setRenderer(&renderer);
    initSimulation(device, 1000);

    runApplication();

    device->release();
    return 0;
}
