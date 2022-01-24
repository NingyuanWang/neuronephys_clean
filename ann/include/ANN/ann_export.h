
#ifndef ANN_EXPORT_H
#define ANN_EXPORT_H

#ifdef ANN_BUILT_AS_STATIC
#  define ANN_EXPORT
#  define ANN_NO_EXPORT
#else
#  ifndef ANN_EXPORT
#    ifdef ANN_EXPORTS
        /* We are building this library */
#      define ANN_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define ANN_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef ANN_NO_EXPORT
#    define ANN_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef ANN_DEPRECATED
#  define ANN_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef ANN_DEPRECATED_EXPORT
#  define ANN_DEPRECATED_EXPORT ANN_EXPORT ANN_DEPRECATED
#endif

#ifndef ANN_DEPRECATED_NO_EXPORT
#  define ANN_DEPRECATED_NO_EXPORT ANN_NO_EXPORT ANN_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef ANN_NO_DEPRECATED
#    define ANN_NO_DEPRECATED
#  endif
#endif

#endif /* ANN_EXPORT_H */
