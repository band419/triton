add_triton_library(TritonCustomToLLVM
    AllocateSharedMemory.cpp
    BarrierOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    TargetInfo.cpp
    TritonGPUToLLVM.cpp

    DEPENDS
    TritonCustomConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonAnalysis
    MLIRLLVMDialect
    MLIRPass
    MLIRTransforms
)
