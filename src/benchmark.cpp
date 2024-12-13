#include "benchmark.h"
#include <cmath>
#include <chrono>
#include <thread>

Benchmark::Benchmark(int numTrials, std::string outputPath) noexcept
    : _numTrials(numTrials)
    , _outputPath(std::move(outputPath)) {
    
    generateConfigs();
    
    // Initialize CSV file with headers
    std::ofstream file(_outputPath);
    file << "simulations,rollouts,c,discount,winrate\n";
    file.close();
}

void Benchmark::generateConfigs() {
    _configs.reserve(91); // Pre-allocate space
    
    const int baseSimulations = 50;
    const int baseRollouts = 50;
    const double baseC = 50.0;
    const double baseDiscount = 0.75;

    // for (int sim = 50; sim <= 120; sim += 5) {
    //     _configs.emplace_back(sim, baseRollouts, baseC, baseDiscount);
    // }
    for (int roll = 10; roll <= 100; roll += 1) {
        _configs.emplace_back(baseSimulations, roll, baseC, baseDiscount);
    }
    // for (int c = 5; c <= 100; c += 5) {
    //     _configs.emplace_back(baseSimulations, baseRollouts, static_cast<double>(c), baseDiscount);
    // }
    // for (double disc = 0.1; disc <= 0.99; disc += 0.05) {
    //     _configs.emplace_back(baseSimulations, baseRollouts, baseC, disc);
    // }
}

void Benchmark::writeConfigToFile(const BenchmarkConfig& config) {
    std::lock_guard<std::mutex> lock(_fileMutex);
    std::ofstream file(_outputPath, std::ios::app);
    file << config.simulations << ','
         << config.rollouts << ','
         << config.c << ','
         << config.discount << ','
         << config.winRate << '\n';
    file.flush();
}

void Benchmark::runParallelBenchmarks() {
    const int totalConfigs = _configs.size();
    const int barWidth = 50;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "Using " << omp_get_max_threads() << " threads for game evaluation\n";
    std::cout << "Progress: [" << std::string(barWidth, ' ') << "] 0%\r";
    std::cout.flush();
    
    // Run configs sequentially
    for (size_t i = 0; i < totalConfigs; i++) {
        auto configStart = std::chrono::high_resolution_clock::now();
        
        _configs[i].winRate = evaluateConfig(_configs[i]);
        writeConfigToFile(_configs[i]);
        
        auto configEnd = std::chrono::high_resolution_clock::now();
        auto configDuration = std::chrono::duration_cast<std::chrono::seconds>(configEnd - configStart);
        
        // Update progress
        float progress = static_cast<float>(i + 1) / totalConfigs;
        int pos = static_cast<int>(barWidth * progress);
        
        std::cout << "Progress: [";
        std::cout << std::string(pos, '=');
        if (pos < barWidth) {
            std::cout << '>';
            std::cout << std::string(barWidth - pos - 1, ' ');
        }
        std::cout << "] " << static_cast<int>(progress * 100.0f) << "% ";
        std::cout << "(" << (i + 1) << "/" << totalConfigs << " configs) ";
        std::cout << "Last config: " << configDuration.count() << "s";
        std::cout << "\r";
        std::cout.flush();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    
    std::cout << "\nCompleted in " << duration.count() << " seconds\n";
}

double Benchmark::evaluateConfig(const BenchmarkConfig& config) const {
    int wins = 0;
    MCTSPlayer player(config.simulations, config.rollouts, config.c, config.discount);
    
    std::vector<Game> games(_numTrials);
    for (auto& game : games) {
        game.init();
    }
    
    static std::mutex printMutex;
    
    #pragma omp parallel for reduction(+:wins) schedule(dynamic, 1)
    for (int i = 0; i < _numTrials; i++) {
        Game& game = games[i];
        while (true) {
            const char move = player.move(&game)[0];
            
            switch(move) {
                case 'l': game.left(); break;
                case 'r': game.right(); break;
                case 'u': game.up(); break;
                case 'd': game.down(); break;
            }
            
            int gameState = game.isGameOver();
            if (gameState == 1) {
                wins++;
                {
                    std::lock_guard<std::mutex> lock(printMutex);
                    //std::cout << "Thread " << omp_get_thread_num() << ": Win  (Game " << i << ")\n";
                }
                break;
            } else if (gameState == -1) {
                {
                    std::lock_guard<std::mutex> lock(printMutex);
                    //std::cout << "Thread " << omp_get_thread_num() << ": Loss (Game " << i << ")\n";
                }
                break;
            }
        }
    }
    
    return static_cast<double>(wins) / _numTrials;
}