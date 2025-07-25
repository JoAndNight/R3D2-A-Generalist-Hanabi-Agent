#pragma once

#include "cpp/hanabi_env.h"
#include <vector>
#include <memory>
#include <iostream>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>

namespace py = pybind11;
using nlohmann::json;

class HumanActorCallback {
 public:
  HumanActorCallback(int numPlayer, int playerIdx)
      : numPlayer_(numPlayer)
      , playerIdx_(playerIdx)
      , stage_(Stage::ObserveBeforeAct) {
  }

  void setPartners(std::vector<std::shared_ptr<HumanActorCallback>> partners) {
    partners_ = std::move(partners);
    assert((int)partners_.size() == numPlayer_);
    assert(partners_[playerIdx_] == nullptr);
  }

  void setActionCallback(py::function callback) {
    actionCallback_ = callback;
  }

  void reset(const HanabiEnv& env);

  bool ready() const {
    return true;  // Human is always ready
  }

  bool stepDone() const {
    return stage_ == Stage::ObserveBeforeAct;
  }

  std::unique_ptr<hle::HanabiMove> next(const HanabiEnv& env);

  // Compatibility methods for R2D2ActorSimple
  int getPlayerIdx() const { return playerIdx_; }
  bool getShuffleColor() const { return false; }
  bool getHideAction() const { return false; }
  int getAux() const { return 0; } // AuxType::Null
  bool getSad() const { return false; }
  std::vector<float> getPlayerEps() const { return {0.0f}; }
  std::vector<float> getPlayerTemp() const { return {}; }
  rela::TensorDict getPrevHidden() const { return {}; }
  void setActReply(const rela::TensorDict& reply) { /* Human doesn't use actReply */ }

  enum class Stage {
    ObserveBeforeAct,
    DecideMove,
    ObserveAfterAct
  };

 private:
  void observeBeforeAct(const HanabiEnv& env);
  std::unique_ptr<hle::HanabiMove> decideMove(const HanabiEnv& env);
  void observeAfterAct(const HanabiEnv& env);

  // Helper functions for preparing data for callback
  std::string getGameStateString(const HanabiEnv& env);
  json getSingleMove(const hle::HanabiMove& move);
  std::vector<json> getLegalMoves(const std::vector<hle::HanabiMove>& legalMoves);
  std::string to_json_string(const HanabiEnv& env);

  const int numPlayer_;
  const int playerIdx_;
  std::vector<std::shared_ptr<HumanActorCallback>> partners_;
  py::function actionCallback_;
  
  // to control stages
  Stage stage_ = Stage::ObserveBeforeAct;
}; 