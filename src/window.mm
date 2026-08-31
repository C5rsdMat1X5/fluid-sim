#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#include <atomic>
#include <renderer.hpp>
#include <window.hpp>

static NSWindow *gWindow = nil;
static NSView *gView = nil;
static CAMetalLayer *gLayer = nil;
static std::atomic<bool> gRunning{false};
float gWindowWidth = 800.0f;
float gWindowHeight = 800.0f;

static NSInteger gFrameCount = 0;
static CFAbsoluteTime gLastFpsUpdate = 0;

static void updateDrawableSizeIfNeeded() {
    CGFloat scale = gWindow.backingScaleFactor;
    CGSize desiredDrawableSize =
        CGSizeMake(gWindowWidth * scale, gWindowHeight * scale);
    if (CGSizeEqualToSize(gLayer.drawableSize, desiredDrawableSize)) {
        return;
    }

    dispatch_sync(dispatch_get_main_queue(), ^{
      gLayer.contentsScale = scale;
      gLayer.drawableSize = desiredDrawableSize;
    });
}

static void updateFpsTitle() {
    gFrameCount++;
    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    CFAbsoluteTime elapsed = now - gLastFpsUpdate;
    if (elapsed < 0.5) {
        return;
    }

    double fps = gFrameCount / elapsed;
    NSString *title =
        [NSString stringWithFormat:@"Fluid Simulation - %.0f FPS", fps];
    dispatch_async(dispatch_get_main_queue(), ^{
      [gWindow setTitle:title];
    });
    gFrameCount = 0;
    gLastFpsUpdate = now;
}

static void renderLoopThread() {
    while (gRunning.load(std::memory_order_relaxed)) {
        gWindowWidth = static_cast<float>(gView.bounds.size.width);
        gWindowHeight = static_cast<float>(gView.bounds.size.height);

        updateDrawableSizeIfNeeded();
        renderFrame();
        updateFpsTitle();
    }
}

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:
    (NSApplication *)sender {
    gRunning.store(false, std::memory_order_relaxed);
    return YES;
}
@end

void *createWindow() {
    NSApplication *app = [NSApplication sharedApplication];
    static AppDelegate *delegate = nil;
    if (delegate == nil) {
        delegate = [[AppDelegate alloc] init];
        [app setDelegate:delegate];
    }

    [app setActivationPolicy:NSApplicationActivationPolicyRegular];

    if (gWindow != nil) {
        return (__bridge void *)gLayer;
    }

    NSRect frame = NSMakeRect(0, 0, 800, 800);
    gWindowWidth = static_cast<float>(frame.size.width);
    gWindowHeight = static_cast<float>(frame.size.height);

    gWindow = [[NSWindow alloc] initWithContentRect:frame
                                          styleMask:NSWindowStyleMaskTitled |
                                                    NSWindowStyleMaskClosable |
                                                    NSWindowStyleMaskResizable
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    [gWindow setReleasedWhenClosed:YES];

    gView = [[NSView alloc] initWithFrame:frame];
    gLayer = [CAMetalLayer layer];
    CGFloat scale = [gWindow backingScaleFactor];
    gLayer.contentsScale = scale;
    gLayer.drawableSize =
        CGSizeMake(frame.size.width * scale, frame.size.height * scale);
    gLayer.displaySyncEnabled = NO;

    [gView setWantsLayer:YES];
    [gView setLayer:gLayer];
    gLayer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
    gLayer.opaque = YES;

    [gWindow setContentView:gView];
    [gWindow center];
    [gWindow setTitle:@"Fluid Simulation"];
    [gWindow makeKeyAndOrderFront:nil];
    gWindowWidth = static_cast<float>(gView.bounds.size.width);
    gWindowHeight = static_cast<float>(gView.bounds.size.height);

    [app activateIgnoringOtherApps:YES];

    return (__bridge void *)gLayer;
}

void runApplication() {
    gLastFpsUpdate = CFAbsoluteTimeGetCurrent();
    gRunning.store(true, std::memory_order_relaxed);

    NSThread *renderThread = [[NSThread alloc] initWithBlock:^{
      renderLoopThread();
    }];
    [renderThread start];

    [NSApp run];

    gRunning.store(false, std::memory_order_relaxed);
}
