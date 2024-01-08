if (APPLE)
    include_directories(onnxruntime-osx-arm64-1.16.3/include/)
    link_directories(onnxruntime-osx-arm64-1.16.3/lib/)
else()
    include_directories(onnxruntime-linux-x64-1.12.1/include/)
    link_directories(onnxruntime-linux-x64-1.12.1/lib/)
endif()

if (WHISPER_SDL2)
    # Stream Tool
    set(TARGET stream)
    
    # Include VadIterator.cpp along with stream.cpp
    add_executable(${TARGET} stream.cpp VadIterator.cpp)

    include(DefaultTargetOptions)

    # Link ONNX Runtime library along with other libraries
    if (APPLE)
        target_link_libraries(${TARGET} PRIVATE common common-sdl whisper onnxruntime ${CMAKE_THREAD_LIBS_INIT})
        set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH "$ORIGIN/onnxruntime-osx-arm64-1.16.3/lib/")
    else()
        target_link_libraries(${TARGET} PRIVATE common common-sdl whisper onnxruntime ${CMAKE_THREAD_LIBS_INIT})
        set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH "$ORIGIN/onnxruntime-linux-x64-1.12.1/lib/")
    endif()
endif ()