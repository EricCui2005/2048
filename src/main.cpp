#include "game.h"
#include "mcts.h"
#include "gameRunner.h"
#include "benchmark.h"
#include <omp.h>
#include <chrono>
#include <string>

void generateData() {
    int numThreads, numTrials, runNumber;
    
    std::cout << "Threads: ";
    std::cin >> numThreads;
    std::cout << "Start run: ";
    std::cin >> runNumber;
    std::cout << "Trials per thread: ";
    std::cin >> numTrials;

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    
    for(int i = 0; i < numThreads; i++) {
        threads.emplace_back(GameRunner::runTrials, runNumber + i, numTrials);
    }

    for(auto& thread : threads) {
        thread.join();
    }
}

void benchmark() {
    const int numTrials = 300;
    const int numThreads = omp_get_max_threads();
    omp_set_num_threads(numThreads);
    
    const auto now = std::chrono::system_clock::now();
    const auto timestamp = std::chrono::system_clock::to_time_t(now);
    const std::string filename = "../../data/benchmark_" + std::to_string(timestamp) + ".csv";
    
    std::cout << "Starting benchmark with " << numThreads << " threads\n";
    std::cout << "Output file: " << filename << "\n";
    
    Benchmark benchmark(numTrials, std::move(filename));
    benchmark.runParallelBenchmarks();
}

int main() {
    #ifdef _GNU_SOURCE
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < omp_get_max_threads(); i++) {
        CPU_SET(i, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
    #endif

    benchmark();
    return 0;
}