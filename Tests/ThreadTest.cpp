#include <catch2/catch_all.hpp>
#include <iostream>
#include <memory>

#include "StratusThread.h"

struct S {};

TEST_CASE( "Stratus Thread Test", "[stratus_thread_test]" ) {
    std::cout << "Beginning stratus::Thread test" << std::endl;

    stratus::Thread thread(true);

    thread.Queue([&thread]() {
        REQUIRE(stratus::Thread::Current() == thread);
    });
    thread.Dispatch();
    thread.Synchronize();

    // Execution should happen in the order queued
    std::atomic<int> counter(0);
    int numFunctions = 100;
    for (int i = 0; i < numFunctions; ++i) {
        thread.Queue([i, &counter]() {
            REQUIRE(counter.fetch_add(1) == i);
        });
    }
    // Make sure no execution has happened yet
    REQUIRE(counter.load() == 0);
    thread.Dispatch();
    thread.Synchronize();
    REQUIRE(counter.load() == numFunctions);

    // Now test threads which share the current thread
    std::vector<std::unique_ptr<stratus::Thread>> threads;
    int numThreads = 50;
    counter.store(0);
    for (int i = 0; i < numThreads; ++i) {
        stratus::Thread * th = new stratus::Thread(false);
        th->Queue([&counter, i, th]() {
            REQUIRE(stratus::Thread::Current() == *th);
            REQUIRE(counter.fetch_add(1) == i);
        });
        threads.push_back(std::unique_ptr<stratus::Thread>(th));
    }
    // Make sure no execution has happened yet
    REQUIRE(counter.load() == 0);
    for (auto & th : threads) th->Dispatch();
    REQUIRE(counter.load() == numThreads);
}