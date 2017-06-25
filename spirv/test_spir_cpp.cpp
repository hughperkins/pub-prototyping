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

#include "spirv-tools/libspirv.h"

// #include "llvm/IR/LLVMContext.h"
// // #include "llvm/IR/DebugInfoMetadata.h"
// #include "llvm/Support/raw_ostream.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/IRReader/IRReader.h"
// #include "llvm/IR/Module.h"

// // An instruction parsed from a binary SPIR-V module.
// typedef struct spv_parsed_instruction_t {
//   // An array of words for this instruction, in native endianness.
//   const uint32_t* words;
//   // The number of words in this instruction.
//   uint16_t num_words;
//   uint16_t opcode;
//   // The extended instruction type, if opcode is OpExtInst.  Otherwise
//   // this is the "none" value.
//   spv_ext_inst_type_t ext_inst_type;
//   // The type id, or 0 if this instruction doesn't have one.
//   uint32_t type_id;
//   // The result id, or 0 if this instruction doesn't have one.
//   uint32_t result_id;
//   // The array of parsed operands.
//   const spv_parsed_operand_t* operands;
//   uint16_t num_operands;
// } spv_parsed_instruction_t;

// // A pointer to a function that accepts a parsed SPIR-V instruction.
// // The parsed_instruction value is transient: it may be overwritten
// // or released immediately after the function has returned.  That also
// // applies to the words array member of the parsed instruction.  The
// // function should return SPV_SUCCESS if and only if parsing should
// // continue.
// typedef spv_result_t (*spv_parsed_instruction_fn_t)(
//     void* user_data, const spv_parsed_instruction_t* parsed_instruction);

spv_result_t on_instruction(
    void* user_data, const spv_parsed_instruction_t* inst) {
    std::cout << "on_instruction()" << std::endl;
    // return 0;
    std::cout << "  opcode=" << inst->opcode << " numwords=" << inst->num_words << " numoperands=" << inst->num_operands << std::endl;
    return SPV_SUCCESS;
}

int main(int argc, char *argv[]) {
    // llvm::LLVMContext context;
    // llvm::SMDiagnostic smDiagnostic;
    // // std::string llFilename = "cl_kernel1.ll";
    // // std::unique_ptr<llvm::Module> M = parseIRFile(llFilename, smDiagnostic, context);
    // std::string llFilename = "cl_kernel1.spv";
    // std::unique_ptr<llvm::Module> M = parseIRFile(llFilename, smDiagnostic, context);
    // if(!M) {
    //     smDiagnostic.print("irtoopencl", llvm::errs());
    //     throw std::runtime_error("failed to parse IR");
    // }

    std::string llFilename = "cl_kernel1.spv";

    int pos = 0;

    std::ifstream f(llFilename, std::ios::in | std::ios::binary | std::ios::ate);

    size_t sizeBytes = f.tellg();
    // memblock = new char [size];
    size_t sizeInts = (sizeBytes + 3) >> 2;
    std::cout << "size Ints " << sizeInts << std::endl;

    // create spirvData first, then reinterpret_cast that, so 4-aligned
    uint32_t *spirvData = new uint32_t[sizeInts + 1];
    char *spirvDataBytes = reinterpret_cast<char *>(spirvData);

    f.seekg (0, std::ios::beg);
    f.read(spirvDataBytes, sizeBytes);
    f.close();

    // see https://github.com/KhronosGroup/SPIRV-Tools/blob/master/include/spirv-tools/libspirv.h

    // Parses a SPIR-V binary, specified as counted sequence of 32-bit words.
    // Parsing feedback is provided via two callbacks provided as function
    // pointers.  Each callback function pointer can be a null pointer, in
    // which case it is never called.  Otherwise, in a valid parse the
    // parsed-header callback is called once, and then the parsed-instruction
    // callback once for each instruction in the stream.  The user_data parameter
    // is supplied as context to the callbacks.  Returns SPV_SUCCESS on successful
    // parse where the callbacks always return SPV_SUCCESS.  For an invalid parse,
    // returns a status code other than SPV_SUCCESS, and if diagnostic is non-null
    // also emits a diagnostic.  If a callback returns anything other than
    // SPV_SUCCESS, then that status code is returned, no further callbacks are
    // issued, and no additional diagnostics are emitted.
    spv_context context;
    spv_diagnostic diag;
    context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
    spv_result_t res = spvBinaryParse(context, 0,
                            spirvData, sizeInts,
                            0,
                            &on_instruction,
                            &diag);
    spvContextDestroy(context);

    return 0;
}
