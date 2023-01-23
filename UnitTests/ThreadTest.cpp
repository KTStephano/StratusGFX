#include <catch2/catch_all.hpp>
#include <iostream>
#include <memory>
#include <vector>

#include "StratusThread.h"
#include "StratusAsync.h"

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
    REQUIRE(thread.Idle() == true);
    thread.Dispatch();
    thread.Synchronize();
    REQUIRE(thread.Idle() == true);
    REQUIRE(counter.load() == numFunctions);

    // Now test threads which share the current thread
    std::vector<std::unique_ptr<stratus::Thread>> threads;
    int numThreads = 50;
    counter.store(0);
    for (int i = 0; i < numThreads; ++i) {
        stratus::Thread * th = new stratus::Thread(false);
        if (i > 0) {
            REQUIRE(*th != *threads[i - 1].get());
        }
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

    // Test the naming ability
    stratus::Thread named("ThreadNameTest", false);
    REQUIRE(named.Name() == "ThreadNameTest");
}

TEST_CASE( "Stratus Async Test", "[stratus_async_test]" ) {
    std::cout << "Beginning stratus::Async test" << std::endl;

    stratus::Thread thread(true);
    // First test a successful async operation
    stratus::Async<std::vector<int>> compute(thread, [](){
        std::vector<int> * vec = new std::vector<int>();
        for (int i = 0; i < 1000; ++i) {
            vec->push_back(i);
        }
        return vec;
    });

    REQUIRE(compute.Completed() == false);
    REQUIRE(compute.Failed() == false);

    thread.Dispatch();
    thread.Synchronize();
    REQUIRE(compute.Completed() == true);
    REQUIRE(compute.Failed() == false);
    REQUIRE(compute.Get().size() == 1000);
    REQUIRE(&compute.Get() == compute.GetPtr().get());

    // Now test a failed async operation
    compute = stratus::Async<std::vector<int>>(thread, [](){
        throw std::runtime_error("Unable to allocate memory");
        return (std::vector<int> *)nullptr;
    });

    REQUIRE(compute.Completed() == false);
    REQUIRE(compute.Failed() == false);

    thread.Dispatch();
    thread.Synchronize();
    REQUIRE(compute.Completed() == true);
    REQUIRE(compute.Failed() == true);

    // Now test callbacks
    stratus::Thread callbackThread(true);
    std::atomic<bool> called(false);
    stratus::Async<int>::AsyncCallback checkCallback = [&called, &callbackThread](stratus::Async<int> as) {
        // The callback will be registered on callbackThread, and so the callback should also take place
        // on callbackThread
        REQUIRE(stratus::Thread::Current() == callbackThread);
        REQUIRE(as.Completed() == true);
        REQUIRE(as.Failed() == false);
        REQUIRE(as.Get() == 10);
        called.store(true);
    };

    stratus::Async<int> computeint(thread, []() { return new int(10); });
    callbackThread.Queue([checkCallback, &computeint]() {
        computeint.AddCallback(checkCallback);
    });
    callbackThread.Dispatch();
    callbackThread.Synchronize();
    REQUIRE(called.load() == false); // should not have happened yet

    // Kick off the async job and wait for completion
    thread.Dispatch();
    thread.Synchronize();

    // At this point a job should have been added to callbackThread
    callbackThread.Dispatch();
    callbackThread.Synchronize();

    REQUIRE(called.load() == true);

    // Test auto-complete when result is given
    auto intcallback = stratus::Async<int>(std::make_shared<int>(10));
    REQUIRE(intcallback.Completed() == true);
    REQUIRE(intcallback.Failed() == false);
    REQUIRE(intcallback.Get() == 10);

    intcallback = stratus::Async<int>(nullptr);
    REQUIRE(intcallback.Completed() == true);
    REQUIRE(intcallback.Failed() == true);
}