# Specify the relative path for the ONNX Runtime include directory
include_directories(onnxruntime-linux-x64-1.12.1/include/)

# Specify the relative path for the ONNX Runtime link directory
link_directories(onnxruntime-linux-x64-1.12.1/lib/)

if (WHISPER_SDL2)
    # Stream Tool
    set(TARGET stream)
    
    # Include VadIterator.cpp along with stream.cpp
    add_executable(${TARGET} stream.cpp VadIterator.cpp)

    include(DefaultTargetOptions)

    # Link ONNX Runtime library along with other libraries
    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper onnxruntime ${CMAKE_THREAD_LIBS_INIT})

    # Set RPATH for finding the ONNX Runtime library at runtime
    set_target_properties(${TARGET} PROPERTIES INSTALL_RPATH "$ORIGIN/onnxruntime-linux-x64-1.12.1/lib/")
endif ()