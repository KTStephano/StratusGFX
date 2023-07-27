#!/usr/bin/env python3

from dependency_build import *

def build_examples():
    windows = True
    if os.name != "nt":
        windows = False
    
    configure = "cmake -Bbuild -S. -DBUILD_TESTS=OFF"
    build = "cmake --build build/ -j 8 --config RelWithDebInfo"
    if not windows:
        configure = "cmake -Bbuild -S. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=OFF"
        build = "cmake --build build/ -j 8"
    configure_build = configure + " && " + build

    os.system(configure_build)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Stratus Engine Dependency + Examples Build')
    parser.add_argument('-a', '--assimp',
                        action='store_true', 
                        default=False)
    args = parser.parse_args()

    build_dependencies(args.assimp)
    build_examples()