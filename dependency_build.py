#!/usr/bin/env python3

import os
import threading
from distutils.dir_util import copy_tree
import argparse

os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(os.cpu_count())

def build_dependencies(build_assimp):
    windows = ""
    if os.name == "nt":
        print("Windows")
        windows = "--config Release"
    else:
        print("Linux")

    print("Building assimp:", build_assimp)

    configure = "cmake -Bbuild -S. -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DASSIMP_BUILD_TESTS=OFF -DASSIMP_USE_STB_IMAGE_STATIC=1 -DASSIMP_BUILD_ASSIMP_TOOLS=OFF"
    build = "cmake --build build/ -j 8 {}".format(windows)
    install = "cmake --install build/ --prefix ../ThirdParty {}".format(windows)
    configure_build_install = configure + " && " + build + " && " + install

    # Update submodules to latest commit
    #os.system('git submodule update --remote --merge')

    # Catch2
    cmd = "cd Catch2 && " + configure_build_install
    catch2 = threading.Thread(target=lambda: os.system(cmd), args=())
    catch2.start()

    # SDL
    cmd = "cd SDL && " + configure_build_install
    sdl = threading.Thread(target=lambda: os.system(cmd), args=())
    sdl.start()

    # Assimp
    assimp = threading.Thread(target=lambda: [], args=())
    if build_assimp:
        cmd = "cd assimp && " + configure_build_install
        assimp = threading.Thread(target=lambda: os.system(cmd), args=())
    assimp.start()

    # Meshoptimizer
    cmd = "cd meshoptimizer && " + configure_build_install
    meshopt = threading.Thread(target=lambda: os.system(cmd), args=())
    meshopt.start()

    # GL3W
    cmd = "cd gl3w && python3 ./gl3w_gen.py --ext"
    gl3w = threading.Thread(target=lambda: os.system(cmd), args=())
    gl3w.start()

    # Bullet3

    configure = "cmake -Bbuild -S. -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release -DBULLET2_MULTITHREADING=ON -DINSTALL_LIBS=ON -DBUILD_UNIT_TESTS=OFF -DBUILD_OPENGL3_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF"
    build = "cmake --build build/ -j 8 {}".format(windows)
    install = "cmake --install build/ --prefix ../ThirdParty {}".format(windows)
    configure_build_install = configure + " && " + build + " && " + install

    cmd = "cd bullet3 && " + configure_build_install
    bullet3 = threading.Thread(target=lambda: os.system(cmd), args=())
    bullet3.start()

    catch2.join()
    sdl.join()
    assimp.join()
    meshopt.join()
    gl3w.join()
    bullet3.join()

    if build_assimp:
        copy_tree("./assimp/contrib", "./ThirdParty/contrib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Stratus Engine Dependency Build')
    parser.add_argument('-a', '--assimp',
                        action='store_true', 
                        default=False)
    args = parser.parse_args()

    build_dependencies(args.assimp)