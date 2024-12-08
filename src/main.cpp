#include "game.h"
#include "mcts.h"
#include "gameRunner.h"

int main() {
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