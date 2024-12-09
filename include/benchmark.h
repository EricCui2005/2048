#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "game.h"
#include "mcts.h"
#include <vector>
#include <string>
#include <string_view>
#include <mutex>
#include <atomic>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>

struct alignas(64) BenchmarkConfig {
    int simulations;
    int rollouts;
    double c;
    double discount;
    std::atomic<double> winRate;
    
    BenchmarkConfig(int s, int r, double cc, double d) 
        : simulations(s), rollouts(r), c(cc), discount(d), winRate(0.0) {}
    
    BenchmarkConfig(const BenchmarkConfig& other)
        : simulations(other.simulations)
        , rollouts(other.rollouts)
        , c(other.c)
        , discount(other.discount)
        , winRate(other.winRate.load()) {}
};

class Benchmark {
public:
    using ConfigVector = std::vector<BenchmarkConfig>;
    
    Benchmark(int numTrials, std::string outputPath) noexcept;
    void runParallelBenchmarks();

private:
    void generateConfigs();
    double evaluateConfig(const BenchmarkConfig& config) const;
    void writeConfigToFile(const BenchmarkConfig& config);
    
    const int _numTrials;
    ConfigVector _configs;
    std::string _outputPath;
    std::mutex _fileMutex;
};

#endif // BENCHMARK_H