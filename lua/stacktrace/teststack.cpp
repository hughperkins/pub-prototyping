extern "C" {
  #include "lua.h"
  #include "lauxlib.h"
  #include "lualib.h"
}
#include <dlfcn.h>

#include <iostream>
using namespace std;

// from http://stackoverflow.com/questions/12256455/print-stacktrace-from-c-code-with-embedded-lua
static int traceback (lua_State *L) {
//  cout << "traceback()" << endl;
//  if (!lua_isstring(L, -1)) {  /* 'message' not a string? */
//    cout << "message not a string " << endl;
//    return 1;  /* keep it intact */
//  }
  lua_getglobal(L, "debug");
//  if (!lua_istable(L, -1)) {
//    cout << "no debug" << endl;
//    lua_pop(L, 1);
//    return 1;
//  }
  lua_getfield(L, -1, "traceback");
//  if (!lua_isfunction(L, -1)) {
//    cout << "no traceback" << endl;
//    lua_pop(L, 2);
//    return 1;
//  }
  lua_remove(L, -2);

  lua_pushthread(L);
  lua_pushvalue(L, -3);  /* pass error message */
  lua_pushinteger(L, 3);  /* skip this function and traceback */
  lua_call(L, 3, 1);  /* call debug.traceback */
//  cout << "traceback: " << lua_tostring(L, -1) << endl;
  return 1;
}

//void teststack() {
int main(int argc, char *argv[]) {
//  void *hdl = dlopen("/home/ubuntu/torch/install/lib/libluajit.so", RTLD_NOW | RTLD_GLOBAL);
//  if(hdl == 0) {
//    cout << dlerror() << endl;
//    return -1;
//  }
//  cout << "opened luajit" << endl;

  lua_State *L = luaL_newstate();
  luaL_openlibs(L);

  lua_getglobal(L, "require");
  lua_pushstring(L, "teststack");
  lua_call(L, 1, 0);
  cout << "got teststack" << endl;

//  lua_getglobal(L, "callfoo");
//  cout << "isfunction? " << lua_isfunction (L, -1) << endl;
//////   lua_pushstring(L, "hello from c");
//  lua_pushcfunction(L, traceback);
//  lua_pcall(L, 0, 0, -1);
//    cout << "pcall err " << lua_tostring(L, -1) << endl;
//  cout << endl;

//  lua_getglobal(L, "STP");
//  cout << "got stp" << endl;
//  lua_getfield(L, -1, "stacktrace");
//  cout << "got stp.stacktrace" << endl;
//  lua_remove(L, -2);
//  cout << "stp isfunction? " << lua_isfunction (L, -1) << endl;
//  lua_getglobal(L, "callfoo");
//  cout << "isfunction? " << lua_isfunction (L, -1) << endl;
//  lua_pcall(L, 0, 0, -1);
//   cout << "pcall err " << lua_tostring(L, -1) << endl;
//  cout << endl;

//  cout << "lua_gettop(L) " << lua_gettop(L) << endl;
//  lua_getglobal(L, "octopus");
//  lua_getglobal(L, "callfoo");
//  cout << "isfunction? " << lua_isfunction (L, 2) << endl;
//  int res = lua_pcall(L, 0, 0, 1);
//  cout << "pcall res " << res << endl;
//   cout << "pcall err " << lua_tostring(L, -1) << endl;
//  cout << endl;

//  cout << "lua_gettop(L) " << lua_gettop(L) << endl;
  lua_pushcfunction(L, traceback);
//  cout << "1 isfunction? " << lua_isfunction (L, 1) << endl;
  lua_getglobal(L, "callfoo");
//  cout << "2 isfunction? " << lua_isfunction (L, 2) << endl;
  lua_pushstring(L, "hello from c");
  int res = lua_pcall(L, 1, 0, 1);
  cout << "pcall res " << res << endl;
   cout << "pcall err " << lua_tostring(L, -1) << endl;
  cout << endl;

//  lua_getglobal(L, "callsub_anteater");
//   lua_pushstring(L, "hello from c");
//  int res = lua_pcall(L, 1, 2, 0);
//  cout << "res " << res << endl;
////  if(res != 0) {
////    cout << "error: " << lua_tostring(L, -1) << endl;
////  }
////  for(int i = 0; i < 3; i++) {
//    cout << "fn err " << lua_tonumber(L, -2) << endl;
//    const char *str = lua_tostring(L, -1);
//    cout << "fn res " << " " << str << endl;
////  }

  lua_close(L);

  return 0;
}

