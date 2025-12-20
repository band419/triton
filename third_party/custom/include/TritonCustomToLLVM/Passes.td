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

#endif // TRITON_CONVERSION_TRITONCUSTOM_TO_LLVM_PASSES
