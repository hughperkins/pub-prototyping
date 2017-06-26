// Copyright Hugh Perkins 2016, 2017

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <fstream>
#include <stdexcept>

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/IR/Module.h"

void readSpirVText() {
    llvm::LLVMContext context;
    llvm::SMDiagnostic smDiagnostic;
    // std::string llFilename = "cl_kernel1.ll";
    // std::unique_ptr<llvm::Module> M = parseIRFile(llFilename, smDiagnostic, context);
    std::string llFilename = "cl_kernel1.spt";
    std::unique_ptr<llvm::Module> M = parseIRFile(llFilename, smDiagnostic, context);
    if(!M) {
        smDiagnostic.print("irtoopencl", llvm::errs());
        throw std::runtime_error("failed to parse IR");
    }
}

int main(int argc, char *argv[]) {
    readSpirVText();
    return 0;
}
