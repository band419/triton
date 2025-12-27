#include "mlir/Analysis/Liveness.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUGLOBALSCRATCHALLOCATIONPASS
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

static int32_t roundUp(int32_t val, int32_t step) {
  auto t = val + step - 1;
  return t - (t % step);
}

static void allocateGMem(Operation *parentOp,
                         llvm::SetVector<Operation *> &callStack,
                         bool useSharedOnGlobalScratch,
                         int32_t sharedMemorySize) {
  // Recursively visit any dependency functions
  parentOp->walk([&](triton::CallOp call) {
    auto callable = call.resolveCallable();
    if (!callable->hasAttr("ttg.global_scratch_memory_size")) {
      auto inserted = callStack.insert(parentOp);
      assert(inserted && "call cycle detected");
      allocateGMem(callable, callStack, useSharedOnGlobalScratch,
                   sharedMemorySize);
      callStack.remove(parentOp);
    }
  });

  MLIRContext *ctx = parentOp->getContext();
  OpBuilder builder(ctx);
  
  // When shared memory is emulated on global scratch, reserve the first
  // sharedMemorySize bytes for shared-emulation region. This ensures
  // GlobalScratchAllocOp allocations don't overlap with the shared region.
  int32_t offset = 0;
  uint32_t largestAlignment = 1;
  if (useSharedOnGlobalScratch && sharedMemorySize > 0) {
    // Reserve space for shared-emulation at the beginning.
    // We'll round up after we know the largest alignment.
    offset = sharedMemorySize;
  }

  // First pass: collect all alignments to determine the largest alignment
  parentOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    uint32_t align = 0;
    if (auto alloc = dyn_cast<triton::gpu::GlobalScratchAllocOp>(op)) {
      align = alloc.getAlignment();
    } else if (auto callOp = dyn_cast<triton::CallOp>(op)) {
      auto callable = callOp.resolveCallable();
      auto align_attr = callable->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_alignment");
      if (align_attr) {
        align = align_attr.getValue().getZExtValue();
      }
    }
    if (align > 0) {
      largestAlignment = std::max(largestAlignment, align);
    }
  });

  // If shared memory is emulated on global scratch, round up the reserved
  // region to the largest alignment
  if (useSharedOnGlobalScratch && sharedMemorySize > 0) {
    offset = roundUp(sharedMemorySize, largestAlignment);
  }

  // Dumb allocation that ignores liveness and makes no attempt to minimize
  // padding
  // TODO: Use a real algorithm
  parentOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    uint32_t nbytes = 0;
    uint32_t align = 0;
    if (auto alloc = dyn_cast<triton::gpu::GlobalScratchAllocOp>(op)) {
      nbytes = alloc.getNbytes();
      align = alloc.getAlignment();
    } else if (auto callOp = dyn_cast<triton::CallOp>(op)) {
      auto callable = callOp.resolveCallable();
      auto nbytes_attr = callable->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_size");
      auto align_attr = callable->getAttrOfType<IntegerAttr>(
          "ttg.global_scratch_memory_alignment");
      assert(nbytes_attr);
      assert(align_attr);

      nbytes = nbytes_attr.getValue().getZExtValue();
      align = align_attr.getValue().getZExtValue();
    }
    if (nbytes > 0) {
      offset = roundUp(offset, align);
      op->setAttr("ttg.global_scratch_memory_offset",
                  builder.getI32IntegerAttr(offset));
      offset += nbytes;
    }
  });
  int32_t totalMemorySize = roundUp(offset, largestAlignment);
  parentOp->setAttr("ttg.global_scratch_memory_size",
                    builder.getI32IntegerAttr(totalMemorySize));
  parentOp->setAttr("ttg.global_scratch_memory_alignment",
                    builder.getI32IntegerAttr(largestAlignment));
}

namespace {
class TritonGPUGlobalScratchAllocationPass
    : public mlir::triton::gpu::impl::TritonGPUGlobalScratchAllocationPassBase<
          TritonGPUGlobalScratchAllocationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Check if shared memory is emulated on global scratch.
    // This is indicated by the module attribute "ttg.shared_memory_model"
    // being set to "global_scratch".
    bool useSharedOnGlobalScratch = false;
    int32_t sharedMemorySize = 0;
    if (auto modelAttr =
            mod->getAttrOfType<StringAttr>("ttg.shared_memory_model")) {
      if (modelAttr.getValue() == "global_scratch") {
        useSharedOnGlobalScratch = true;
        // Get the shared memory size from ttg.shared attribute
        // (set by AllocateSharedMemory pass)
        if (auto sharedAttr = mod->getAttrOfType<IntegerAttr>("ttg.shared")) {
          sharedMemorySize = sharedAttr.getValue().getZExtValue();
        }
      }
    }

    bool seenKernel = false;

    SetVector<Operation *> callStack;
    mod->walk([&](triton::FuncOp func) {
      allocateGMem(func, callStack, useSharedOnGlobalScratch, sharedMemorySize);

      if (func.getVisibility() == SymbolTable::Visibility::Public) {
        assert(!seenKernel);
        seenKernel = true;
        auto size =
            func->getAttrOfType<IntegerAttr>("ttg.global_scratch_memory_size");
        auto align = func->getAttrOfType<IntegerAttr>(
            "ttg.global_scratch_memory_alignment");
        assert(size);
        assert(align);
        mod->setAttr("ttg.global_scratch_memory_size", size);
        mod->setAttr("ttg.global_scratch_memory_alignment", align);
      }
    });
    assert(seenKernel);
  }
};
} // namespace
                                                                                                                                                                                                                                                                                                                                                                                                              　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　