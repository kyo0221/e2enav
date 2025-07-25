#ifndef E2ENAV__VISIBILITY_CONTROL_H_
#define E2ENAV__VISIBILITY_CONTROL_H_

#ifdef __cplusplus
extern "C"
{
#endif

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define E2ENAV_EXPORT __attribute__ ((dllexport))
    #define E2ENAV_IMPORT __attribute__ ((dllimport))
  #else
    #define E2ENAV_EXPORT __declspec(dllexport)
    #define E2ENAV_IMPORT __declspec(dllimport)
  #endif
  #ifdef E2ENAV_BUILDING_LIBRARY
    #define E2ENAV_PUBLIC E2ENAV_EXPORT
  #else
    #define E2ENAV_PUBLIC E2ENAV_IMPORT
  #endif
  #define E2ENAV_PUBLIC_TYPE E2ENAV_PUBLIC
  #define E2ENAV_LOCAL
#else
  #define E2ENAV_EXPORT __attribute__ ((visibility("default")))
  #define E2ENAV_IMPORT
  #if __GNUC__ >= 4
    #define E2ENAV_PUBLIC __attribute__ ((visibility("default")))
    #define E2ENAV_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define E2ENAV_PUBLIC
    #define E2ENAV_LOCAL
  #endif
  #define E2ENAV_PUBLIC_TYPE
#endif

#ifdef __cplusplus
}
#endif

#endif  // E2ENAV__VISIBILITY_CONTROL_H_