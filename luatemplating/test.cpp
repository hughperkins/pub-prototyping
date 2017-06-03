#include <iostream>
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
using namespace std;

static struct {
    char const*name;
} globals[] = {
    { "pairs"  },
    { "ipairs" },
    { "load" },
    { "type"   },
    { "table"  },
    { "string" },
    { "math"   },
    { NULL     }
};

void setValue( lua_State *L, string name, string value ) {
    lua_pushstring(L, value.c_str());
    lua_setfield(L, LUA_GLOBALSINDEX, name.c_str());
}

int main( int argc, char *argv[] ) {
    lua_State *L = lua_open();
    luaL_openlibs(L);
    luaL_dostring(L, "print 'hello'");
    if(luaL_dofile(L, "../templater.lua")) {
        printf("Could not load file: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return 0;
    }
    lua_getfield(L, -1, "compile_file");
    setValue(L, "color", "blue");
    lua_pushstring(L, "../test1.cl");
    if (lua_pcall(L, 1, 1, 0) != 0) {
        printf("Error: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return 0;
    }
    printf("result: %s\n", lua_tostring(L, -1));
    lua_pop(L, 1);

    lua_getfield(L, -1, "compile_file");
    lua_pushstring(L, "../test1.cl");
    if (lua_pcall(L, 1, LUA_MULTRET, 0) != 0) {
        printf("Error: %s\n", lua_tostring(L, -1));
        lua_close(L);
        return 0;
    }
    printf("%s\n", lua_tostring(L, -1));    
    lua_pop(L, 1);

    for( int j = 0; j < 10000; j++ ) {
        setValue(L, "MAX_CUTORCH_DIMS", "123");
        setValue(L, "operation", "val1 + val2");
        setValue(L, "adim", "3");
        setValue(L, "bdim", "3");
        setValue(L, "cdim", "7");
        lua_newtable(L);
        int i = 1;
        lua_pushstring(L, "3");
        lua_rawseti(L, -2, i++);
        lua_pushstring(L, "7");
        lua_rawseti(L, -2, i++);
        lua_setfield(L, LUA_GLOBALSINDEX, "dims");
        lua_getfield(L, -1, "compile_file");
        lua_pushstring(L, "../test_foo.cl");
        if (lua_pcall(L, 1, LUA_MULTRET, 0) != 0) {
            printf("Error: %s\n", lua_tostring(L, -1));
            lua_close(L);
            return 0;
        }
//        printf("%s\n", lua_tostring(L, -1));    
        lua_pop(L, 1);
    }

    lua_close(L);
    return 0;
}

