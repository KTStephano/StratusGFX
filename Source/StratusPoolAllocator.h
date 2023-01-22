#pragma once

namespace stratus {
    // Allocates memory of a pre-defined size provide optimal data locality
    // with zero fragmentation.
    //
    // This is not designed for fast allocation speed, so try to keep
    // that to a minimum per frame.
    template<typename E>
    struct PoolAllocator {

    };
}