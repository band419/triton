//===- SPMDOpToLLVM.cpp - SPMD Ops to LLVM for Custom SIMT backend --------===//
//
// Convert SPMD operations (program_id, num_programs) to custom LLVM intrinsics.
//
//===----------------------------------------------------------------------===//

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace {

/// Convert GetProgramIdOp to llvm.riscv.simt.program.id intrinsic
struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  GetProgramIdOpConversion(LLVMTypeConverter &converter,
                           const mlir::triton::Custom::TargetInfo &targetInfo,
                           PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    
    // Map Triton axis to ProgramIDDim
    triton::ProgramIDDim axis;
    switch (op.getAxisAsInt()) {
    case 0:
      axis = triton::ProgramIDDim::X;
      break;
    case 1:
      axis = triton::ProgramIDDim::Y;
      break;
    case 2:
      axis = triton::ProgramIDDim::Z;
      break;
    default:
      return failure();
    }
    
    Value result = targetInfo.programId(rewriter, loc, moduleOp, axis);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const mlir::triton::Custom::TargetInfo &targetInfo;
};

/// Convert GetNumProgramsOp to llvm.riscv.simt.num.programs intrinsic
struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  GetNumProgramsOpConversion(LLVMTypeConverter &converter,
                             const mlir::triton::Custom::TargetInfo &targetInfo,
                             PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    
    triton::ProgramIDDim axis;
    switch (op.getAxisAsInt()) {
    case 0:
      axis = triton::ProgramIDDim::X;
      break;
    case 1:
      axis = triton::ProgramIDDim::Y;
      break;
    case 2:
      axis = triton::ProgramIDDim::Z;
      break;
    default:
      return failure();
    }
    
    Value result = targetInfo.getNumPrograms(rewriter, loc, axis);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const mlir::triton::Custom::TargetInfo &targetInfo;
};

} // namespace

namespace mlir::triton::Custom {

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfo &targetInfo,
                                 PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<GetNumProgramsOpConversion>(typeConverter, targetInfo, benefit);
}

} // namespace mlir::triton::Custom
