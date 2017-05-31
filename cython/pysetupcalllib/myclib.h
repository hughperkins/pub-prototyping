#pragma once

#if defined(_WIN32) 
# if defined(myclib_EXPORTS)
#  define myclib_EXPORT __declspec(dllexport)
# else
#  define myclib_EXPORT __declspec(dllimport)
# endif // DeepCL_EXPORTS
#else // _WIN32
# define myclib_EXPORT
#endif

void myclib_EXPORT myclib_sayHello();

