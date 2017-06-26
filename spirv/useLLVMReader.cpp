#include <iostream>
#include <fstream>
#include <stdexcept>

#ifndef _SPIRV_SUPPORT_TEXT_FMT
#define _SPIRV_SUPPORT_TEXT_FMT
#endif

#include "llvm/Support/SPIRV.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

void readSpirVText() {
    // llvm::LLVMContext context;
    // llvm::SMDiagnostic smDiagnostic;
    // // std::string llFilename = "cl_kernel1.ll";
    // // std::unique_ptr<llvm::Module> M = parseIRFile(llFilename, smDiagnostic, context);
    // std::string llFilename = "cl_kernel1.spt";
    // std::unique_ptr<llvm::Module> M = parseIRFile(llFilename, smDiagnostic, context);
    // if(!M) {
    //     smDiagnostic.print("irtoopencl", llvm::errs());
    //     throw std::runtime_error("failed to parse IR");
    // }

    // from https://github.com/KhronosGroup/SPIRV-LLVM/blob/khronos/spirv-3.6.1/tools/llvm-spirv/llvm-spirv.cpp
    std::string spvFilename = "cl_kernel1.spv";
    llvm::LLVMContext Context;
    std::ifstream IFS(spvFilename, std::ios::binary);
    llvm::Module *M;
    std::string Err;

    if (!llvm::ReadSPIRV(Context, IFS, M, Err)) {
        llvm::errs() << "Fails to load SPIRV as LLVM Module: " << Err << '\n';
        return;
    }

    // DEBUG(dbgs() << "Converted LLVM module:\n" << *M);

    llvm::raw_string_ostream ErrorOS(Err);
    if (llvm::verifyModule(*M, &ErrorOS)){
        llvm::errs() << "Fails to verify module: " << ErrorOS.str();
        return;
    }

    for(auto it=M->begin(); it != M->end(); it++) {
        llvm::Function *fn = &*it;
        std::cout << fn->getName().str() << std::endl;
        for(auto it2=fn->begin(); it2 != fn->end(); it2++) {
            llvm::BasicBlock *block = &*it2;
            std::cout << "block " << block->getName().str() << std::endl;
            for(auto it3=block->begin(); it3 != block->end(); it3++) {
                llvm::Instruction *inst = &*it3;
                std::cout << "instruction " << inst->getOpcodeName() << std::endl;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    readSpirVText();
    return 0;
}
