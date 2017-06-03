// #include "llvm/ADT/APFloat.h"
// #include "llvm/ADT/STLExtras.h"
// #include "llvm/IR/DerivedTypes.h"
// #include "llvm/IR/LLVMContext.h"
// #include "llvm/IR/IRBuilder.h"
// #include "llvm/IRReader/IRReader.h"
// #include "llvm/IR/Module.h"
// #include "llvm/IR/Verifier.h"
// #include "llvm/IR/Type.h"
// #include "llvm/IR/ValueSymbolTable.h"
// #include "llvm/IR/Instructions.h"
// #include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
// #include "llvm/Support/SourceMgr.h"
// // #include <iostream>
// // #include <cstdio>
// // #include <cstdlib>
// #include <vector>
// #include <map>
// #include <set>
// // #include <stdexcept>
// #include <sstream>
// #include <fstream>

using namespace llvm;
// using namespace std;

// #include <raw_ostream>

// static llvm::LLVMContext TheContext;

// #include <iostream>
#include <cassert>
// using namespace std;

int main(int argc, char *argv[]) {
    // SMDiagnostic Err;
    // TheModule = parseIRFile(target, Err, TheContext);
    // if(!TheModule) {
    //     Err.print(argv[0], errs());
    //     return 1;
    // }
    outs() << "argc " << argc << "\n";
    assert(argc < 2);
    // Value *myint = ConstantInt::getSigned(IntegerType::get(TheContext, 32), 123);
    // std::cout << "isa " << isa<PointerType>(myint->getType()) << std::endl;
    // PointerType *ptr = cast<PointerType>(myint->getType());
    // std::cout << "ptr " << ptr << std::endl;
    return 0;
}
