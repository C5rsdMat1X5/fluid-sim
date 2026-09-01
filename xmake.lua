add_rules("mode.debug", "mode.release")

set_defaultmode("release")
set_policy("build.ccache", true)

target("fluid-sim")
    add_rules("xcode.application")
    set_languages("c++17")

    add_includedirs("include")
    add_files("src/*.cpp")
    add_files("src/*.mm")
    add_files("src/*.metal")
    
    add_frameworks("Metal")
    add_frameworks("Foundation")
    add_frameworks("QuartzCore")
    add_frameworks("AppKit")

    if is_mode("release") then
        set_optimize("fastest") 
        set_symbols("hidden")
        set_strip("all")
        add_defines("NDEBUG")

        set_policy("build.optimization.lto", true)
    elseif is_mode("debug") then
        set_symbols("debug")
        set_optimize("none")
    end
