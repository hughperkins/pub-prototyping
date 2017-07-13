cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set(space_separated "a series of tokens")
message("space_separated ${space_separated}")

set(sc_separated a series of tokens)
message("sc_separated ${sc_separated}")

add_custom_target(test_batch_spaces
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/example_batch.sh ${space_separated}
)

add_custom_target(test_batch_sc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/example_batch.sh ${sc_separated}
)

set(var3 ${space_separated})
set(var4 "${space_separated}")
set(var5 ${sc_separated})
set(var6 "${sc_separated}")

message("var3 ${var3}")
message("var4 ${var4}")
message("var5 ${var5}")
message("var6 ${var6}")

set(var7 ${space_separated} ${space_separated})
set(var8 "${space_separated} ${space_separated}")
set(var9 ${sc_separated} ${sc_separated})
set(var10 "${sc_separated} ${sc_separated}")

message("var7 ${var7}")
message("var8 ${var8}")
message("var9 ${var9}")
message("var10 ${var10}")

set(var11 "originally space separated")
set(var12 originally sc separated)
separate_arguments(var11)
separate_arguments(var12)
message("var11 ${var11}")
message("var12 ${var12}")

# set(after_separg ${space_separated})
# separate_arguments(space_separated)
# message("after_separg ${after_separg}")

set(var13 ${notexist} some sc strings)
set(var13b ${notexist} some;sc;strings)
set(var13c ${notexist} "some;sc;strings")
set(var14 "${notexist} some space strings")
set(var15 ${notexist} "some space strings")
set(var16 "${notexist} some;sc;strings")
message("var13 ${var13}")
message("var13b ${var13b}")
message("var13c ${var13c}")
message("var14 ${var14}")
message("var15 ${var15}")
message("var16 ${var16}")
