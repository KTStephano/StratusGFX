#include "StratusLog.h"
#include "StratusThread.h"
#include "StratusUtils.h"

namespace stratus {
    static void EmbedLogData(const std::string & tagline, const std::string& function, const int line, std::ostream & out) {
        auto& thread = Thread::Current();
        out << tagline << " Thread::(" << thread.Name() << ") " << function << ":" << line << " -> ";
    }

    Log::Log() {}

    std::ostream& Log::Inform(const std::string & function, const int line) const {
        EmbedLogData("[Info]", function, line, std::cout);
        return std::cout;
    }

    std::ostream& Log::Warn(const std::string & function, const int line) const {
        EmbedLogData("[Warn]", function, line, std::cout);
        return std::cout;
    }

    std::ostream& Log::Error(const std::string & function, const int line) const {
        EmbedLogData("[Error]", function, line, std::cerr);
        return std::cerr;
    }

    bool Log::Initialize() {
        return true;
    }

    SystemStatus Log::Update(const double) {
        return SystemStatus::SYSTEM_CONTINUE;
    }

    void Log::Shutdown() {
        
    }
}