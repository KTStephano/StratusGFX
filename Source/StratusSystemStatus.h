#pragma once

namespace stratus {
    // Allows core functions within the engine's main loop to signal
    // what should happen after the current update finishes
    enum class SystemStatus {
        // Main loop can continue for another frame
        SYSTEM_CONTINUE,
        // All systems should be restarted
        SYSTEM_RESTART,
        // All systems should gracefully shut down
        SYSTEM_SHUTDOWN,
        // Critical error happened which we can't recover from
        SYSTEM_PANIC
    };
}