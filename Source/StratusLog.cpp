#include "StratusLog.h"
#include "StratusThread.h"

namespace stratus {
    Log * Log::_instance = nullptr;

    static void EmbedLogData(const std::string & tagline, std::ostream & out) {
        auto& thread = Thread::Current();
        out << tagline << " Thread::(" << thread.Name() << ")";
    }

    Log::Log() {}

    std::ostream& Log::Inform() const {
        EmbedLogData("[Info]", std::cout);
        return std::cout;
    }

    std::ostream& Log::Warn() const {
        EmbedLogData("[Warn]", std::cout);
        return std::cout;
    }

    std::ostream& Log::Error() const {
        EmbedLogData("[Error]", std::cerr);
        return std::cerr;
    }
}