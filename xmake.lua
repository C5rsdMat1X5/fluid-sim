add_rules("mode.debug", "mode.release")

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