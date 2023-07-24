#!/usr/bin/env python3

import os
import threading
from distutils.dir_util import copy_tree
import argparse

parser = argparse.ArgumentParser(prog='Stratus Engine Dependency Build')
parser.add_argument('-a', '--assimp',
                    action='store_true', 
                    default=False)
args = parser.parse_args()

os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(os.cpu_count())

windows = ""
build_assimp = args.assimp
if os.name == "nt":
    print("Windows")
    windows = "--config Release"
else:
    print("Linux")

print("Building assimp:", build_assimp)

configure = "cmake -Bbuild -S. -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DASSIMP_BUILD_TESTS=OFF"
build = "cmake --build build/ -j 8 {}".format(windows)
install = "cmake --install build/ --prefix ../ThirdParty {}".format(windows)
configure_build_install = configure + " && " + build + " && " + install

# Update submodules to latest commit
#os.system('git submodule update --remote --merge')

# Catch2
cmd = "cd Catch2 && " + configure_build_install
t = threading.Thread(target=lambda: os.system(cmd), args=())
t.start()

# SDL
cmd = "cd SDL && " + configure_build_install
t = threading.Thread(target=lambda: os.system(cmd), args=())
t.start()

# Assimp
if build_assimp:
    cmd = "cd assimp && " + configure_build_install
    t = threading.Thread(target=lambda: os.system(cmd), args=())
    t.start()

copy_tree("./assimp/contrib", "./ThirdParty/contrib")

# Meshoptimizer
cmd = "cd meshoptimizer && " + configure_build_install
t = threading.Thread(target=lambda: os.system(cmd), args=())
t.start()

# GL3W
cmd = "cd gl3w && python3 ./gl3w_gen.py --ext"
t = threading.Thread(target=lambda: os.system(cmd), args=())
t.start()