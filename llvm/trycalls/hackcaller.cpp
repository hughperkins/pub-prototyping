#include <iostream>
#include <fstream>
#include <sstream>

#include "llvm/IR/AssemblyAnnotationWriter.h"
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

using namespace llvm;
using namespace std;

static llvm::LLVMContext TheContext;
static llvm::IRBuilder<> Builder(TheContext);
static std::unique_ptr<llvm::Module> TheModule;

GlobalVariable *addGlobalVariable(Module *M, string name, string value) {
    // string name = ".mystr1";
    // string desiredString = value;
    int N = value.size() + 1;
    ArrayType *strtype = ArrayType::get(IntegerType::get(TheContext, 8), N);
    Constant *charConst = M->getOrInsertGlobal(StringRef(name), strtype);
    ConstantDataSequential *charConstSeq = cast<ConstantDataSequential>(charConst);

    ConstantDataArray *constchararray = cast<ConstantDataArray>(ConstantDataArray::get(TheContext, ArrayRef<uint8_t>((uint8_t *)value.c_str(), N)));
    GlobalVariable *str = M->getNamedGlobal(StringRef(name));
    str->setInitializer(constchararray);
    return str;
}

// Instruction *buildGlobalVariableInst(Module *M, GlobalVariable *var) {

// }

Instruction *addString(Module *M, string name, string value) {
    // Module *M = prev->getParent();
    GlobalVariable *var = addGlobalVariable(M, name, value);

    int N = value.size() + 1;
    ArrayType *arrayType = ArrayType::get(IntegerType::get(TheContext, 8), N);
    Value * indices[2];
    indices[0] = ConstantInt::getSigned(IntegerType::get(TheContext, 32), 0);
    indices[1] = ConstantInt::getSigned(IntegerType::get(TheContext, 32), 0);
    GetElementPtrInst *elem = GetElementPtrInst::CreateInBounds(arrayType, var, ArrayRef<Value *>(indices, 2));
    // elem->insertAfter(prev);
    return elem;
}

void patchFunction(Function *F) {
    BasicBlock *block = &F->getEntryBlock();
    for(auto it=block->begin(); it != block->end(); it++) {
        Instruction *inst = &*it;
        it->dump();
    }
    auto it = block->end();
    it--;
    it--;

    IntegerType *inttype = IntegerType::get(TheContext, 32);

    // it--;
    // Instruction *inst = &*it;
    // inst->dump();
    // Instruction *i2 = inst->clone();
    // i2->insertAfter(inst);
    // Instruction *i3 = inst->clone();
    // CallInst *call = cast<CallInst>(i3);
    // cout << "i3 operands " << i3->getNumOperands() << endl;
    // cout << "op0" << endl;
    // i3->getOperand(0)->dump();
    // cout << "op1" << endl;
    // i3->getOperand(1)->dump();

    // ConstantInt *constzero = ConstantInt::getSigned(inttype, 222);
    // i3->setOperand(0, ConstantInt::getSigned(inttype, 222));

    // i3->insertAfter(i2);

    // i2->setOperand(0, ConstantInt::getSigned(inttype, 333));

    Module *M = F->getParent();
    // Function *printInt = M->getFunction("_Z8printInti");
    Function *printInt = cast<Function>(M->getOrInsertFunction(
        "_Z8printInti",
        Type::getVoidTy(TheContext),
        IntegerType::get(TheContext, 32),
        NULL));
    ConstantInt *c4 = ConstantInt::getSigned(inttype, 555);
    CallInst *i4 = CallInst::Create(printInt, ArrayRef<Value *>(c4));
    i4->insertAfter(&*it);

    Function *printInt2 = cast<Function>(M->getOrInsertFunction(
        "_Z9printInt2i",
        Type::getVoidTy(TheContext),
        IntegerType::get(TheContext, 32),
        NULL));
    ConstantInt *c5 = ConstantInt::getSigned(inttype, 666);
    CallInst *i5 = CallInst::Create(printInt2, ArrayRef<Value *>(c5));
    i5->insertAfter(i4);

    Type *floatType = Type::getFloatTy(TheContext);
    Function *printFloat = cast<Function>(M->getOrInsertFunction(
        "_Z10printFloatf",
        Type::getVoidTy(TheContext),
        floatType,
        NULL));
    Constant *f6 = ConstantFP::get(floatType, 123.456f);
    CallInst *i6 = CallInst::Create(printFloat, ArrayRef<Value *>(f6));
    i6->insertAfter(i5);

    string name = ".mystr1";
    string desiredString = "this is the way the world ends";
    ArrayType *strtype = ArrayType::get(IntegerType::get(TheContext, 8), desiredString.size() + 1);
    Constant *charConst = M->getOrInsertGlobal(StringRef(name), strtype);
    ConstantDataSequential *charConstSeq = cast<ConstantDataSequential>(charConst);

    ConstantDataArray *constchararray = cast<ConstantDataArray>(ConstantDataArray::get(TheContext, ArrayRef<uint8_t>((uint8_t *)desiredString.c_str(), desiredString.size() + 1)));
    GlobalVariable *str = M->getNamedGlobal(StringRef(name));
    str->setInitializer(constchararray);

    ArrayType *arrayType = ArrayType::get(IntegerType::get(TheContext, 8), desiredString.size() + 1);
    Value * indices[2];
    indices[0] = ConstantInt::getSigned(inttype, 0);
    indices[1] = ConstantInt::getSigned(inttype, 0);
    GetElementPtrInst *elem = GetElementPtrInst::CreateInBounds(arrayType, str, ArrayRef<Value *>(indices, 2));
    elem->insertAfter(i6);

    Function *printChars = M->getFunction("_Z10printCharsPKc");
    CallInst *c7 = CallInst::Create(printChars, elem);
    c7->insertAfter(elem);

    string newString = "made easier";
    GlobalVariable *strfoobar = addGlobalVariable(M, ".str.foobar", newString);
    ArrayType *arrayType2 = ArrayType::get(IntegerType::get(TheContext, 8), newString.size() + 1);
    // Value * indices[2];
    // indices[0] = ConstantInt::getSigned(inttype, 0);
    // indices[1] = ConstantInt::getSigned(inttype, 0);
    GetElementPtrInst *elem2 = GetElementPtrInst::CreateInBounds(arrayType2, strfoobar, ArrayRef<Value *>(indices, 2));
    elem2->insertAfter(c7);

    // Function *printChars = M->getFunction("_Z10printCharsPKc");
    CallInst *c8 = CallInst::Create(printChars, elem2);
    c8->insertAfter(elem2);

    Instruction *l1 = addString(M, ".str.new2", "my new string");
    l1->insertAfter(c8);
    CallInst *c9 = CallInst::Create(printChars, l1);
    c9->insertAfter(l1);

    Function *printChars2 = cast<Function>(M->getOrInsertFunction(
        "_Z11printChars2PKc",
        Type::getVoidTy(TheContext),
        PointerType::get(IntegerType::get(TheContext, 8), 0),
        NULL));

    Instruction *l2 = addString(M, ".str.new3", "another new string");
    l2->insertAfter(c9);
    CallInst *c10 = CallInst::Create(printChars2, l2);
    c10->insertAfter(l2);

    // it++;
    // Instruction *insertpoint = &*it;
    // block->getInstList().insert(insertpoint, i2);
}

string getGlobalString(Module *M, string name) {
    GlobalVariable *str = M->getNamedGlobal(StringRef(name));
    // assert(str != 0);
    // if(str == 0) {
    //     cout << ".str is 0 " << endl;
    // }
    Constant *strconst = str->getInitializer();
    // assert(strconst != 0);
    ConstantDataSequential *strconstseq = cast<ConstantDataSequential>(strconst);
    // assert(strconstseq != 0);
    // string strconststring = string(strconstseq->getAsCString());
    return strconstseq->getAsCString();
}

void patchModule(Module *M) {
    for(auto it=M->global_begin(); it != M->global_end(); it++) {
        GlobalValue *glob = &*it;
        cout << string(glob->getName()) << endl;
        if(GlobalVariable *var = dyn_cast<GlobalVariable>(glob)) {
            cout << "   its a globalvaraible" << endl;
        }
    }
    string varName = ".str.1";
    GlobalVariable *str = M->getNamedGlobal(StringRef(varName));
    // assert(str != 0);
    if(str == 0) {
        cout << ".str is 0 " << endl;
    }
    Constant *strconst = str->getInitializer();
    assert(strconst != 0);
    ConstantDataSequential *strconstseq = cast<ConstantDataSequential>(strconst);
    assert(strconstseq != 0);
    string strconststring = string(strconstseq->getAsCString());
    cout << "str: " << strconststring << endl;

    cout << getGlobalString(M, ".str.2") << endl;

    Function *F = M->getFunction("main");
    patchFunction(F);
}

int main(int argc, char *argv[]) {
    string src = "caller.ll";
    string dest = "caller2.ll";

    SMDiagnostic Err;
    TheModule = parseIRFile(src, Err, TheContext);
    if(!TheModule) {
        Err.print(argv[0], errs());
        return 1;
    }

    patchModule(TheModule.get());

    AssemblyAnnotationWriter assemblyAnnotationWriter;
    ofstream ofile;
    ofile.open(dest);
    raw_os_ostream my_raw_os_ostream(ofile);
    verifyModule(*TheModule);
    TheModule->print(my_raw_os_ostream, &assemblyAnnotationWriter);
    ofile.close();

    return 0;
}
