#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "cpp/hanabi_env.h"
#include "cpp/utils.h"
#include "cpp/r2d2_actor_utils.h"

namespace py = pybind11;

class R2D2ActorSimple {
 public:
  R2D2ActorSimple(
      py::object agent,           // Direct PyTorch model
      int numPlayer,              // total number of players
      int playerIdx,              // player idx for this player
      bool vdn,
      bool sad,
      bool hideAction)
      : agent_(agent)
      , jitModel_(agent.attr("_c").cast<torch::jit::script::Module*>())
      , rng_(1)  // not used in eval mode
      , numPlayer_(numPlayer)
      , playerIdx_(playerIdx)
      , vdn_(vdn)
      , sad_(sad)
      , shuffleColor_(false)
      , hideAction_(hideAction)
      , aux_(AuxType::Null)
      , playerEps_(1)
      , playerTemp_(1) {
    initHidden_ = getH0(1);
  }

  void setPartners(std::vector<std::shared_ptr<R2D2ActorSimple>> partners) {
    partners_ = std::move(partners);
    assert((int)partners_.size() == numPlayer_);
    assert(partners_[playerIdx_] == nullptr);
  }

  void setExploreEps(std::vector<float> eps) {
    epsList_ = std::move(eps);
  }

  void setBoltzmannT(std::vector<float> t) {
    tempList_ = std::move(t);
  }

  void setLLMPrior(
      const std::unordered_map<std::string, std::vector<float>>& llmPrior,
      std::vector<float> piklLambdas,
      float piklBeta) {
    assert(llmPrior_.size() == 0); // has not been previously set
    for (const auto& kv : llmPrior) {
      llmPrior_[kv.first] = torch::tensor(kv.second, torch::kFloat32);
    }
    piklLambdas_ = std::move(piklLambdas);
    piklBeta_ = piklBeta;
  }

  void updateLLMLambda(std::vector<float> piklLambdas) {
    piklLambdas_ = std::move(piklLambdas);
  }

  void reset(const HanabiEnv& env);

  bool ready() const {
    return true;  // Always ready since we don't use async calls
  }

  bool stepDone() const {
    return stage_ == Stage::ObserveBeforeAct;
  }

  std::unique_ptr<hle::HanabiMove> next(const HanabiEnv& env);

  float getSuccessFictRate() {
    float rate = -1;
    if (totalFict_) {
      rate = (float)successFict_ / totalFict_;
    }
    successFict_ = 0;
    totalFict_ = 0;
    return rate;
  }

  std::tuple<int, int, int, int> getPlayedCardInfo() const {
    return {noneKnown_, colorKnown_, rankKnown_, bothKnown_};
  }

  enum class Stage {
    ObserveBeforeAct,
    DecideMove,
    FictAct,
    ObserveAfterAct,
    StoreTrajectory
  };

 private:
  // Direct model call without rela
  rela::TensorDict callModel(const std::string& method, const rela::TensorDict& input) {
    std::cout << "    R2D2ActorSimple::callModel() - Method: " << method << std::endl;
    std::cout << "    Input TensorDict size: " << input.size() << std::endl;
    for (const auto& kv : input) {
      std::cout << "      Key: " << kv.first << ", Shape: ";
      if (kv.second.dim() == 0) {
        std::cout << "scalar";
      } else {
        std::cout << "[";
        for (int i = 0; i < kv.second.dim(); ++i) {
          if (i > 0) std::cout << ", ";
          std::cout << kv.second.size(i);
        }
        std::cout << "]";
      }
      std::cout << std::endl;
    }
    
    torch::NoGradGuard ng;
    
    // Convert to batch format like BatchRunner does
    rela::TensorDict batchInput;
    for (const auto& kv : input) {
      // Add batch dimension (unsqueeze at dim 0)
      batchInput[kv.first] = kv.second.unsqueeze(0);
    }
    
    std::cout << "    Converting batch input to IValue..." << std::endl;
    std::vector<torch::jit::IValue> ivalues;
    ivalues.push_back(rela::tensor_dict::toIValue(batchInput, torch::kCPU));
    std::cout << "    IValue vector size: " << ivalues.size() << std::endl;
    std::cout << "    Calling JIT model method: " << method << std::endl;
    
    try {
      auto output = jitModel_->get_method(method)(ivalues);
      std::cout << "    Converting output from IValue..." << std::endl;
      auto batchResult = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
      
      // Remove batch dimension (squeeze at dim 0) to get single sample result
      rela::TensorDict result;
      for (const auto& kv : batchResult) {
        result[kv.first] = kv.second.squeeze(0);
      }
      
      std::cout << "    callModel completed successfully" << std::endl;
      return result;
    } catch (const std::exception& e) {
      std::cout << "    ERROR in callModel: " << e.what() << std::endl;
      std::cout << "    Trying to get model method signature..." << std::endl;
      
      // Try to get method signature for debugging
      try {
        auto method_obj = jitModel_->get_method(method);
        std::cout << "    Method object obtained successfully" << std::endl;
      } catch (const std::exception& e2) {
        std::cout << "    ERROR getting method: " << e2.what() << std::endl;
      }
      
      throw;
    } catch (...) {
      std::cout << "    ERROR in callModel: Unknown exception" << std::endl;
      throw;
    }
  }

  rela::TensorDict getH0(int numPlayer) {
    std::cout << "    R2D2ActorSimple::getH0() - numPlayer: " << numPlayer << std::endl;
    std::vector<torch::jit::IValue> input{numPlayer};
    std::cout << "    Calling JIT model get_h0 method..." << std::endl;
    auto output = jitModel_->get_method("get_h0")(input);
    std::cout << "    Converting get_h0 output from IValue..." << std::endl;
    auto result = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
    std::cout << "    getH0 result size: " << result.size() << std::endl;
    for (const auto& kv : result) {
      std::cout << "      Key: " << kv.first << ", Shape: ";
      if (kv.second.dim() == 0) {
        std::cout << "scalar";
      } else {
        std::cout << "[";
        for (int i = 0; i < kv.second.dim(); ++i) {
          if (i > 0) std::cout << ", ";
          std::cout << kv.second.size(i);
        }
        std::cout << "]";
      }
      std::cout << std::endl;
    }
    std::cout << "    getH0 completed successfully" << std::endl;
    return result;
  }

  void observeBeforeAct(const HanabiEnv& env);
  std::unique_ptr<hle::HanabiMove> decideMove(const HanabiEnv& env);
  void fictAct(const HanabiEnv& env);
  void observeAfterAct(const HanabiEnv& env);
  void storeTrajectory(const HanabiEnv& env);

  py::object agent_;                                    // Python agent object
  torch::jit::script::Module* const jitModel_;          // JIT model pointer
  std::mt19937 rng_;
  const int numPlayer_;
  const int playerIdx_;
  const bool vdn_;
  const bool sad_;
  const bool shuffleColor_;
  const bool hideAction_;
  const AuxType aux_;

  // optional, e.g. ppo does not use it
  std::vector<float> epsList_;
  std::vector<float> tempList_;

  std::vector<float> playerEps_;  // vector for easy conversion to tensor, size==1
  std::vector<float> playerTemp_;
  std::vector<int> colorPermute_;
  std::vector<int> invColorPermute_;

  rela::TensorDict initHidden_;
  rela::TensorDict prevHidden_;
  rela::TensorDict hidden_;

  bool offBelief_ = false;
  rela::TensorDict beliefHidden_;

  std::vector<int> privCardCount_;
  std::vector<hle::HanabiCardValue> sampledCards_;

  int totalFict_ = 0;
  int successFict_ = 0;
  bool validFict_ = false;
  std::unique_ptr<hle::HanabiState> fictState_ = nullptr;
  std::vector<std::shared_ptr<R2D2ActorSimple>> partners_;

  // to control stages
  Stage stage_ = Stage::ObserveBeforeAct;

  // Store replies for synchronous calls
  rela::TensorDict actReply_;
  rela::TensorDict targetReply_;

  // information on cards played
  // only computed during eval mode
  std::vector<std::vector<float>> perCardPrivV0_;
  int noneKnown_ = 0;
  int colorKnown_ = 0;
  int rankKnown_ = 0;
  int bothKnown_ = 0;

  // llm stuff
  std::unordered_map<std::string, torch::Tensor> llmPrior_;
  std::vector<float> piklLambdas_;
  float piklLambda_ = 0;
  float piklBeta_ = 1;
}; 