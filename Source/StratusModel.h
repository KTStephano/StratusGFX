#pragma once

#include "StratusRenderEntity.h"
#include <string>

namespace stratus {
    class Renderer;
    
    class Model : public RenderEntity {
        friend class Renderer;

        /**
         * File storing the full model
         */
        std::string _filename;

        /**
         * Stores true if all loading was successful and false otherwise
         */
        bool _valid = false;

        // We only want the Renderer to create this class
        Model(Renderer &, const std::string filename);

    public:
        virtual ~Model();

        const std::string & getFile() const {
            return _filename;
        }

        // Checks if the model loaded properly
        bool isValid() const {
            return _valid;
        }
    };
}