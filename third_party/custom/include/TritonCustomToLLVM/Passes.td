#ifndef TRITON_CONVERSION_TRITONCUSTOM_TO_LLVM_PASSES
#define TRITON_CONVERSION_TRITONCUSTOM_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonCustomToLLVM : Pass<"convert-triton-custom-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM for Custom SIMT backend";
    let description = [{
      This pass converts TritonGPU dialect to LLVM dialect for the Custom SIMT
      backend. It generates target-agnostic LLVM IR with custom intrinsics for
      SIMT operations (barrier, lane_id, program_id, etc.) that can be later
      lowered by a custom LLVM backend or external codegen.
    }];
    let constructor = "mlir::triton::createConvertTritonCustomToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"warpSize", "warp-size", "int32_t", /*default*/"32",
               "SIMT warp size (number of lanes per warp)">,
    ];
}

def AllocateCustomSharedMemory : Pass<"allocate-custom-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation on Custom SIMT backend";

  let description = [{
    This pass uses the `ModuleAllocation` analysis to:
      - Annotate modules with an attribute with the amount of shared/local
        memory used.
      - Annotate operations with an offset into the total shared/local memory.

    For the Custom SIMT backend, shared memory is allocated from global memory
    by the runtime, so this pass primarily tracks the required size.
  }];
}

#endif // TRITON_CONVERSION_TRITONCUSTOM_TO_LLVM_PASSES
