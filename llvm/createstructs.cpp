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

#include "ir-to-opencl-common.h"

using namespace llvm;
using namespace std;

static llvm::LLVMContext TheContext;
static llvm::IRBuilder<> Builder(TheContext);
static std::unique_ptr<llvm::Module> TheModule;



int main(int argc, char *argv[]) {
    SMDiagnostic Err;
    if(argc != 4) {
        outs() << "Usage: " << argv[0] << " infile-rawhost.ll infile-device.cl outfile-patchedhost.ll" << "\n";
        return 1;
    }

    string rawhostfilename = "test/eigen/generated/test_cuda_nullary-hostraw.ll";
    string patchedhostfilename = "test/eigen/generated/test_cuda_nullary-host2.ll";

    // string rawhostfilename = argv[1];
    // string deviceclfilename = argv[2];
    // string patchedhostfilename = argv[3];
    outs() << "reading rawhost ll file " << rawhostfilename << "\n";
    // outs() << "reading device cl file " << deviceclfilename << "\n";
    outs() << "outputing to patchedhost file " << patchedhostfilename << "\n";

    TheModule = parseIRFile(rawhostfilename, Err, TheContext);
    if(!TheModule) {
        Err.print(argv[0], errs());
        return 1;
    }

    patchModule(deviceclfilename, TheModule.get());

    AssemblyAnnotationWriter assemblyAnnotationWriter;
    ofstream ofile;
    ofile.open(patchedhostfilename);
    raw_os_ostream my_raw_os_ostream(ofile);
    verifyModule(*TheModule);
    TheModule->print(my_raw_os_ostream, &assemblyAnnotationWriter);
    ofile.close();
    return 0;
}
