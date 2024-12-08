#include "game.h"
#include "mcts.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <filesystem>

struct GameData {
    std::string move;
    std::vector<std::vector<int>> currentState;
};

std::string gridToString(std::vector<std::vector<int>> grid) {
    std::string res = "[";
    for(const auto& row : grid) {
        res += "[";
        for(const auto& cell : row) {
            res += std::to_string(cell) + ",";
        }
        res.pop_back();
        res += "],";
    }
    res += "]";
    return res;
}

std::mutex cout_mutex;
std::mutex file_mutex;

void runTrialsThread(int startRun, int numTrials) {
    MCTSPlayer player(100, 10, 100, 0.9);
    Game game;
    std::map<std::string, int> res = {{"Wins", 0}, {"Losses", 0}};
    
    std::string filepath = "../../data/run" + std::to_string(startRun) + ".csv";
    std::ofstream csvfile(filepath);
    
    {
        std::lock_guard<std::mutex> lock(file_mutex);
        if (!csvfile.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return;
        }
    }
    
    for(int i = 0; i < numTrials; ++i) {
        game.init();
        std::vector<GameData> data;

        while(true) {
            // Game logic here (unlocked)
            auto currentState = game.getGrid();
            std::string move = player.move(&game);
            
            // Process move (unlocked)
            switch(move[0]) {
                case 'l': game.left(); break;
                case 'r': game.right(); break;
                case 'u': game.up(); break;
                case 'd': game.down(); break;
            }

            // Collect data
            data.push_back({move, currentState});

            int gameState = game.isGameOver();
            if(gameState == 1) {
                res["Wins"]++;
                // Only lock when writing to file
                {
                    std::lock_guard<std::mutex> lock(file_mutex);
                    for(const auto& line : data) {
                        csvfile << line.move;
                        for(const auto& row : line.currentState) {
                            for (const auto& cell : row) {
                                csvfile << "," << cell;
                            }
                        }
                        csvfile << std::endl;
                    }
                }
                std::cout << "Win " << res["Wins"] << std::endl;
                break;
            } else if(gameState == -1) {
                res["Losses"]++;
                std::cout << "Loss " << res["Losses"] << std::endl;
                break;
            }
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Run " << startRun << " - Wins: " << res["Wins"] 
                 << ", Losses: " << res["Losses"] << std::endl;
    }
}

int main() {
    int numThreads, numTrials, runNumber;

    std::cout << "Enter number of threads: ";
    std::cin >> numThreads;
    
    std::cout << "Enter starting run number: ";
    std::cin >> runNumber;
    
    std::cout << "Enter number of trials to run per thread: ";
    std::cin >> numTrials;

    std::vector<std::thread> threads;
    
    // Create all threads
    for(int i = 0; i < numThreads; i++) {
        threads.emplace_back(runTrialsThread, runNumber + i, numTrials);
    }

    // Wait for all threads to complete
    for(auto& thread : threads) {
        thread.join();
    }

    return 0;
}