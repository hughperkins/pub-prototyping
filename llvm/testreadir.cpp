#include <iostream>
#include <string>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Support/raw_ostream.h>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <fstream>

using namespace llvm;
using namespace std;


void dumpBlock(BasicBlock *block) {
  for(auto it = block->begin(); it != block->end(); it++) {
    Instruction *inst = &*it;
    inst->dump();
    cout << endl;
  }
}

void dumpFunction(Function *F) {
  for(auto it = F->begin(); it != F->end(); it++) {
    BasicBlock *block = &*it;
    dumpBlock(block);
  }
}

int main(int argc, char *argv[]) {
  StringRef filename = "/home/ubuntu/prototyping/foo.ll";
  LLVMContext context;

    SMDiagnostic smDiagnostic;
    std::unique_ptr<llvm::Module> M = parseIRFile(filename, smDiagnostic, context);

    for(auto it=M->begin(); it != M->end(); it++) {
      Function *F = &*it;
      cout << "func " << F->getName().str() << endl;
      dumpFunction(F);
    }

  // ErrorOr<std::unique_ptr<MemoryBuffer>> fileOrErr =
  //   MemoryBuffer::getFileOrSTDIN(filename);
  // if (std::error_code ec = fileOrErr.getError()) {
  //   std::cerr << " Error opening input file: " + ec.message() << std::endl;
  //   return 2;
  // }
  // ErrorOr<std::unique_ptr<llvm::Module > > moduleOrErr =
  //     parseBitcodeFile(fileOrErr.get()->getMemBufferRef(), context);
  // if (std::error_code ec = fileOrErr.getError()) {
  //   std::cerr << "Error reading Moduule: " + ec.message() << std::endl;
  //   return 3;
  // }

  // Module *m = moduleOrErr.get();
  return 0;
}
