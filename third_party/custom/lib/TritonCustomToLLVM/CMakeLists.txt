add_triton_library(TritonCustomToLLVM
    BarrierOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
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
