#pragma once

#include <vector>
#include <tuple>
#include <torch/torch.h>
#include "hanabi-learning-environment/hanabi_lib/hanabi_card.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_game.h"
#include "hanabi-learning-environment/hanabi_lib/hanabi_hand.h"

namespace hanabi_learning_env {
    class HanabiGame;
    class HanabiHand;
}

// Helper functions shared between R2D2Actor and R2D2ActorSimple
std::vector<hle::HanabiCardValue> sampleCards(
    const std::vector<float>& v0,
    const std::vector<int>& privCardCount,
    const std::vector<int>& invColorPermute,
    const hle::HanabiGame& game,
    const hle::HanabiHand& hand,
    std::mt19937& rng);

std::tuple<std::vector<hle::HanabiCardValue>, bool> filterSample(
    const torch::Tensor& samples,
    const std::vector<int>& privCardCount,
    const std::vector<int>& invColorPermute,
    const hle::HanabiGame& game,
    const hle::HanabiHand& hand);

std::tuple<bool, bool> analyzeCardBelief(const std::vector<float>& b); 