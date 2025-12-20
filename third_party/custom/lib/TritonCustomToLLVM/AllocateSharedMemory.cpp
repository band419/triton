//===- AllocateSharedMemory.cpp - Shared Memory Allocation for Custom ----===//
//
// Allocate shared memory for Custom SIMT backend.
// For this backend, "shared memory" is actually allocated from global memory
// by the runtime, so we just track the required size.
//
//===----------------------------------------------------------------------===//

#include "TritonCustomToLLVM/Passes.h"

#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATECUSTOMSHAREDMEMORY
#include "TritonCustomToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

struct AllocateCustomSharedMemory
    : public triton::impl::AllocateCustomSharedMemoryBase<
          AllocateCustomSharedMemory> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    
    // Run module allocation analysis
    ModuleAllocation allocation(mod);
    
    // Get the total shared memory size required
    size_t sharedMemSize = allocation.getSharedMemorySize();
    
    // Set module attribute with shared memory size
    OpBuilder b(mod.getContext());
    mod->setAttr("triton_custom.shared_memory_size",
                 b.getI64IntegerAttr(sharedMemSize));
    
    // For Custom backend, shared memory operations will be handled
    // during lowering to LLVM by offsetting from a base pointer
    // provided by the runtime.
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateCustomSharedMemoryPass() {
  return std::make_unique<AllocateCustomSharedMemory>();
}

} // namespace mlir::triton
