#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
// #include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <set>
// #include <iostream>
#include <stdexcept>
#include <sstream>
#include <fstream>

using namespace llvm;
using namespace std;

void mutateGlobalConsructorNumElements(GlobalVariable *var, int numElements) {
    Type *constructorElementType = cast<ArrayType>(var->getType()->getElementType())->getElementType();
    Type *newVartype = PointerType::get(ArrayType::get(constructorElementType, numElements), 0);
    var->mutateType(newVartype);
}

void appendGlobalConstructorCall(Module *M, std::string functionName) {
    GlobalVariable *ctors = cast<GlobalVariable>(M->getNamedValue("llvm.global_ctors"));
    int oldNumConstructors = cast<ArrayType>(ctors->getType()->getPointerElementType())->getNumElements();
    outs() << "constructors " << oldNumConstructors << "\n";
    mutateGlobalConsructorNumElements(ctors, oldNumConstructors + 1);

    ConstantArray *initializer = cast<ConstantArray>(ctors->getInitializer());

    Constant **initializers = new Constant *[oldNumConstructors + 1];
    for(int i = 0; i < oldNumConstructors; i++) {
        initializers[i] = initializer->getAggregateElement((unsigned int)i);
    }
    Constant *structValues[3];
    structValues[0] = ConstantInt::get(IntegerType::get(M->getContext(), 32), 1000000);
    structValues[1] = M->getOrInsertFunction(
        functionName,
        Type::getVoidTy(M->getContext()),
        NULL);
    structValues[2] = ConstantPointerNull::get(PointerType::get(IntegerType::get(M->getContext(), 8), 0));
    initializers[oldNumConstructors] = ConstantStruct::getAnon(ArrayRef<Constant *>(&structValues[0], &structValues[3]));
    Constant *newinit = ConstantArray::get(initializer->getType(), ArrayRef<Constant *>(&initializers[0], &initializers[oldNumConstructors + 1]));
    ctors->setInitializer(newinit);
    delete[] initializers;
}

void patchModule(Module *M) {
    appendGlobalConstructorCall(M, "callfoo");
    appendGlobalConstructorCall(M, "callbar");
}

int main(int argc, char *argv[]) {
    LLVMContext llvmContext;
    // IRBuilder<> irBuilder(TheContext);
    unique_ptr<Module> module;

    SMDiagnostic smDiagnostic;
    module = parseIRFile("build/test.ll", smDiagnostic, llvmContext);
    if(!module) {
        smDiagnostic.print(argv[0], errs());
        return 1;
    }

    patchModule(module.get());

    AssemblyAnnotationWriter assemblyAnnotationWriter;
    ofstream ofile;
    ofile.open("build/test-patched.ll");
    raw_os_ostream my_raw_os_ostream(ofile);
    verifyModule(*module);
    module->print(my_raw_os_ostream, &assemblyAnnotationWriter);
    ofile.close();

    return 0;
}
