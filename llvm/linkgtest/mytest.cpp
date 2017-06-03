#include <iostream>

#include "gtest/gtest.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"

using namespace std;
using namespace llvm;

TEST(mytest, basic) {
    cout << "mytest " << endl;
    llvm::LLVMContext context;
    Value *a = ConstantInt::getSigned(IntegerType::get(context, 32), 123);

    SMDiagnostic smDiagnostic;
    std::unique_ptr<llvm::Module> M = parseIRFile("somefile.ll", smDiagnostic, context);
    if(!M) {
        smDiagnostic.print("irtoopencl", errs());
        // return 1;
        throw runtime_error("failed to parse IR");
    }

}
