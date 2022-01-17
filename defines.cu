#include "defines.cuh"
namespace Utility
{
    std::default_random_engine& UnseededRandomEngine()
    {
        // This random engine is seeded with the same value every time.
        // This means that randomness will be THE SAME between runs. 
        static std::default_random_engine gen;
        return gen;
    }

    std::default_random_engine& SeededRandomEngine()
    {
        // This random engine is seeded with the first time the function is called.
        // This means that randomness will be DIFFERENT between runs. 
        static std::default_random_engine gen(std::chrono::system_clock::now().time_since_epoch().count());
        return gen;
    }
};
