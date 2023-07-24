#!/usr/bin/env python3

import os
import threading
from distutils.dir_util import copy_tree

os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(os.cpu_count())

#configure_build_install = "cmake -Bbuild -S. -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF && cmake --build build/ --config Release && cmake --install build/ --prefix ../ThirdParty --config Release"
configure = "cmake -Bbuild -S. -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release"
build = "cmake --build build/"
install = "cmake --install build/ --prefix ../ThirdParty"
configure_build_install = configure + " && " + build + " && " + install

# Catch2
cmd = "cd Catch2 && " + configure_build_install
t = threading.Thread(target=lambda: os.system(cmd), args=())
t.start()

# SDL
cmd = "cd SDL && " + configure_build_install
t = threading.Thread(target=lambda: os.system(cmd), args=())
t.start()

# Assimp
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