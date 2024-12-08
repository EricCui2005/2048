#include "mcts.h"
#include <algorithm>
#include <random>

MCTS::MCTS(Game* game, int simulations, int rollouts, double c, double discount) {
    _game = game;
    _simulations = simulations;
    _rollouts = rollouts;
    _c = c;
    _discount = discount;
    _totalVisits = 1;
    
    _moves = {
        {"left", {0, 0.0}},
        {"right", {0, 0.0}},
        {"up", {0, 0.0}},
        {"down", {0, 0.0}}
    };
}

std::string MCTS::move() {
    for(int i = 0; i < _simulations; ++i) {
        Game gameCopy = *_game;
        std::string move = choice();

        switch (move[0]) {
            case 'l':
                gameCopy.left();
                break;
            case 'r':
                gameCopy.right();
                break;
            case 'u':
                gameCopy.up();
                break;
            case 'd':
                gameCopy.down();
                break;
        }

        double utilityEstimate = 0.0;
        for(int j = 0; j < _rollouts; ++j) {
            utilityEstimate += std::pow(_discount, j) * gameCopy.getScore();
            std::vector<std::string> moves = {"left", "right", "up", "down"};

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 3);
            std::string simMove = moves[dis(gen)];

            switch (simMove[0]) {
                case 'l':
                    gameCopy.left();
                    break;
                case 'r':
                    gameCopy.right();
                    break;
                case 'u':
                    gameCopy.up();
                    break;
                case 'd':
                    gameCopy.down();
                    break;
            }
        }

        _totalVisits++;
        _moves[move].N++;
        _moves[move].Q = ((_moves[move].N - 1) * _moves[move].Q + utilityEstimate) / _moves[move].N;
    }

    return std::max_element(_moves.begin(), _moves.end(), [](const auto& a, const auto& b) {
        return a.second.Q < b.second.Q;
    })->first;
}

std::string MCTS::choice() {
    std::vector<std::pair<std::string, double>> ucbs;
    for(const auto& move : _moves) {
        double ucb = move.second.Q + _c * std::sqrt(div(std::log(_totalVisits), move.second.N));
        ucbs.emplace_back(move.first, ucb);
    }
    return std::max_element(ucbs.begin(), ucbs.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    })->first;
}

double MCTS::div(double dividend, double divisor) {
    if(divisor == 0) return dividend >= 0 ? INFINITY : -INFINITY;
    return static_cast<double>(dividend) / divisor;
}

MCTSPlayer::MCTSPlayer(int simulations, int rollouts, double c, double discount) {
    _simulations = simulations;
    _rollouts = rollouts;
    _c = c;
    _discount = discount;
}

std::string MCTSPlayer::move(Game* game) {
    MCTS mcts(game, _simulations, _rollouts, _c, _discount);
    return mcts.move();
}