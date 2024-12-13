#include "mcts.h"
#include <algorithm>
#include <random>

MCTS::MCTS(Game* game, int simulations, int rollouts, double c, double discount)
    : _game(game)
    , _simulations(simulations)
    , _rollouts(rollouts)
    , _c(c)
    , _discount(discount)
    , _totalVisits(1) {
}

std::string MCTS::move() {
    for (auto& move : _moves) {
        move.N = 0;
        move.Q = 0.0;
    }

    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::uniform_int_distribution<> dis(0, NUM_MOVES - 1);

    for(int i = 0; i < _simulations; i++) {
        Game gameCopy = *_game;
        
        int moveIdx = choice();

        switch (moveIdx) {
            case 0: gameCopy.left(); break;
            case 1: gameCopy.right(); break;
            case 2: gameCopy.up(); break;
            case 3: gameCopy.down(); break;
        }

        double utilityEstimate = 0.0;
        double discountFactor = 1.0;

        for(int j = 0; j < _rollouts; j++) {
            utilityEstimate += discountFactor * gameCopy.getScore();
            discountFactor *= _discount;

            int simMoveIdx = dis(gen);
            switch (simMoveIdx) {
                case 0: gameCopy.left(); break;
                case 1: gameCopy.right(); break;
                case 2: gameCopy.up(); break;
                case 3: gameCopy.down(); break;
            }
        }

        _totalVisits++;
        auto& move = _moves[moveIdx];
        move.N++;
        move.Q = ((move.N - 1) * move.Q + utilityEstimate) / move.N;
    }

    int bestMoveIdx = 0;
    double bestQ = _moves[0].Q;
    for(int i = 1; i < NUM_MOVES; i++) {
        if(_moves[i].Q > bestQ) {
            bestMoveIdx = i;
            bestQ = _moves[i].Q;
        }
    }

    return MOVE_STRINGS[bestMoveIdx];
}

int MCTS::choice() {
    int bestMoveIdx = 0;
    double bestUCB = -std::numeric_limits<double>::infinity();

    for(int i = 0; i < NUM_MOVES; i++) {
        const auto& move = _moves[i];
        double ucb = move.Q + _c * std::sqrt(std::log(_totalVisits) / (move.N + 1e-10));
        if(ucb > bestUCB) {
            bestMoveIdx = i;
            bestUCB = ucb;
        }
    }

    return bestMoveIdx;
}

MCTSPlayer::MCTSPlayer(int simulations, int rollouts, double c, double discount) 
    : _simulations(simulations)
    , _rollouts(rollouts)
    , _c(c)
    , _discount(discount) {
}

std::string MCTSPlayer::move(Game* game) {
    MCTS mcts(game, _simulations, _rollouts, _c, _discount);
    return mcts.move();
}