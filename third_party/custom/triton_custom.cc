//===- triton_custom.cc - Python bindings for Custom SIMT backend ---------===//
//
// Provides Python bindings to register passes and dialects for the Custom
// SIMT backend.
//
//===----------------------------------------------------------------------===//

#include "TritonCustomToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_custom_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton;

  m.def("add_to_llvmir",
        [](mlir::PassManager &pm, int32_t warpSize) {
          pm.addPass(createConvertTritonCustomToLLVMPass(warpSize));
        },
        py::arg("pm"), py::arg("warp_size") = 32);

  m.def("add_allocate_shared_memory",
        [](mlir::PassManager &pm) {
          pm.addPass(createAllocateCustomSharedMemoryPass());
        },
        py::arg("pm"));
}

void init_triton_custom(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_custom_passes_ttgpuir(passes.def_submodule("ttgpuir"));
}

PYBIND11_MODULE(libTritonCustom, m) {
  m.doc() = "Python bindings for Triton Custom SIMT backend";
  init_triton_custom(std::move(m.def_submodule("custom")));
}
