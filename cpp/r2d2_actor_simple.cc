#include "cpp/r2d2_actor_simple.h"
#include "cpp/utils.h"
#include "cpp/r2d2_actor_utils.h"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

// Remove duplicate global variables and helper functions that are already in r2d2_actor.cc

/////////////////////////// R2D2ActorSimple ///////////////////////////////

void R2D2ActorSimple::reset(const HanabiEnv& env) {
  // some asserts to keep us sane
  if (offBelief_) {
    assert(!vdn_);
  }

  hidden_ = initHidden_;

  if (epsList_.size()) {
    assert(playerEps_.size() == 1);
    playerEps_[0] = epsList_[rng_() % epsList_.size()];
  }

  if (tempList_.size() > 0) {
    assert(playerTemp_.size() == 0);
    playerTemp_[0] = tempList_[rng_() % tempList_.size()];
  }

  if (piklLambdas_.size() > 0) {
    piklLambda_ = piklLambdas_[rng_() % piklLambdas_.size()];
  }

  // other-play
  if (shuffleColor_) {
    const auto& game = env.getHleGame();
    colorPermute_.clear();
    invColorPermute_.clear();
    for (int i = 0; i < game.NumColors(); ++i) {
      colorPermute_.push_back(i);
      invColorPermute_.push_back(i);
    }
    std::shuffle(colorPermute_.begin(), colorPermute_.end(), rng_);
    std::sort(invColorPermute_.begin(), invColorPermute_.end(), [&](int i, int j) {
      return colorPermute_[i] < colorPermute_[j];
    });
  }
}

// Returns optional move and whether we're done acting
std::unique_ptr<hle::HanabiMove> R2D2ActorSimple::next(const HanabiEnv& env) {
  std::cout << "R2D2ActorSimple::next() - Player " << playerIdx_ << ", Stage: ";
  
  if (stage_ == Stage::ObserveBeforeAct) {
    std::cout << "ObserveBeforeAct" << std::endl;
    observeBeforeAct(env);
    stage_ = Stage::DecideMove;
    return nullptr;
  }

  if (stage_ == Stage::DecideMove) {
    std::cout << "DecideMove" << std::endl;
    auto move = decideMove(env);
    if (offBelief_) {
      stage_ = Stage::FictAct;
    } else {
      stage_ = Stage::ObserveBeforeAct;
    }
    return move;
  }

  if (stage_ == Stage::FictAct) {
    std::cout << "FictAct" << std::endl;
    fictAct(env);
    stage_ = Stage::ObserveAfterAct;
    return nullptr;
  }

  if (stage_ == Stage::ObserveAfterAct) {
    std::cout << "ObserveAfterAct" << std::endl;
    observeAfterAct(env);
    if (env.terminated()) {
      stage_ = Stage::StoreTrajectory;
    } else {
      stage_ = Stage::ObserveBeforeAct;
    }
    return nullptr;
  }

  if (stage_ == Stage::StoreTrajectory) {
    std::cout << "StoreTrajectory" << std::endl;
    storeTrajectory(env);
    stage_ = Stage::ObserveBeforeAct;
    return nullptr;
  }

  std::cout << "Unknown stage!" << std::endl;
  assert(false);
  return nullptr;
}

void R2D2ActorSimple::observeBeforeAct(const HanabiEnv& env) {
  std::cout << "R2D2ActorSimple::observeBeforeAct() - Player " << playerIdx_ << std::endl;
  
  torch::NoGradGuard ng;
  prevHidden_ = hidden_;
  std::vector<int> token_ids;
  std::string result;
  const auto& state = env.getHleState();
  result = state.ToText();
  token_ids = state.ToTokenize();

  std::cout << "  Getting observation..." << std::endl;
  auto input = observe(
      state,
      playerIdx_,
      shuffleColor_,
      colorPermute_,
      invColorPermute_,
      hideAction_,
      aux_,
      sad_);
  std::cout << "  Observation obtained" << std::endl;
  
  input["priv_s_text"] = torch::tensor(token_ids);

  // add features such as eps and temperature
  if (epsList_.size()) {
    input["eps"] = torch::tensor(playerEps_);
  }
  if (playerTemp_.size() > 0) {
    input["temperature"] = torch::tensor(playerTemp_);
  }

  if (llmPrior_.size()) {
    auto obs = hle::HanabiObservation(state, playerIdx_, true);
    auto lastMove = getLastNonDealMove(obs.LastMoves());

    float piklLambda = 0;
    auto foundMove = llmPrior_.end();
    if (env.getCurrentPlayer() == playerIdx_) {
      piklLambda = piklLambda_;
      if (lastMove == nullptr) {
        foundMove = llmPrior_.find("[null]");
      } else {
        foundMove = llmPrior_.find(lastMove->ToLangKey());
      }
    } else {
      piklLambda = 0;
      foundMove = llmPrior_.find("[null]");
    }

    assert(foundMove != llmPrior_.end());
    input["pikl_lambda"] = torch::tensor(piklLambda, torch::kFloat32);
    input["llm_prior"] = foundMove->second * piklBeta_;
  }

  addHid(input, hidden_);
  
  std::cout << "  After addHid, input TensorDict size: " << input.size() << std::endl;
  for (const auto& kv : input) {
    std::cout << "    Key: " << kv.first << ", Shape: ";
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
  
  std::cout << "  Calling model with 'act' method..." << std::endl;
  // Direct synchronous call to neural network
  auto reply = callModel("act", input);
  std::cout << "  Model call completed" << std::endl;

  // Store reply for decideMove
  actReply_ = reply;

  // collect stats for eval mode
  const auto& game = env.getHleGame();
  auto obs = hle::HanabiObservation(state, state.CurPlayer(), true);
  auto encoder = hle::CanonicalObservationEncoder(&game);
  auto [privV0, cardCount] = encoder.EncodeV0Belief(
    obs, std::vector<int>(), false, std::vector<int>(), false);
  perCardPrivV0_ = extractPerCardBelief(
    privV0, env.getHleGame(), obs.Hands()[0].Cards().size());

  if (!offBelief_) {
    std::cout << "  observeBeforeAct completed (no off-belief)" << std::endl;
    return;
  }

  // forward belief model
  assert(!shuffleColor_ && !hideAction_);
  auto [beliefInput, privCardCount, v0] = spartaObserve(
      state, playerIdx_);
  privCardCount_ = privCardCount;

  // For simplicity, we'll use analytical belief here
  // Note: We need to declare these functions as extern or move them to a shared header
  // For now, we'll skip the belief sampling functionality
  // sampledCards_ = sampleCards(
  //     v0,
  //     privCardCount_,
  //     invColorPermute_,
  //     env.getHleGame(),
  //     state.Hands()[playerIdx_],
  //     rng_);

  fictState_ = std::make_unique<hle::HanabiState>(state);
  std::cout << "  observeBeforeAct completed (with off-belief)" << std::endl;
}

std::unique_ptr<hle::HanabiMove> R2D2ActorSimple::decideMove(const HanabiEnv& env) {
  torch::NoGradGuard ng;

  // get act results, update hid, and record action if needed
  auto reply = actReply_;
  int action = reply.at("a").item<int64_t>();
  moveHid(reply, hidden_);

  if (env.getCurrentPlayer() == playerIdx_ && reply.count("bp_logits") > 0) {
    auto bp_logits = reply.at("bp_logits");
    auto adv = reply.at("adv");
    auto pikl_lambda = reply.at("pikl_lambda");
    auto legal_adv = reply.at("legal_adv");
    auto legal_move = reply.at("legal_move");

    std::cout << "@decideMove, step: " << env.numStep() << std::endl;
    std::cout << "@decideMove last move: " << env.getMove(env.getLastAction()).ToString() << std::endl;
    for (int action = 0; action < 20; ++action) {
      if (legal_move[action].item<int>() == 0) {
        continue;
      }
      if (pikl_lambda.item<float>() == 0) {
        continue;
      }
      std::cout << std::fixed;
      std::cout << std::setprecision(4)
                << "@decideMove action: " << env.getMove(action).ToString()
                << ", adv: " << adv[action].item<float>()
                << ", bp_logits: " << bp_logits[action].item<float>()
                << ", final_adv: " << legal_adv[action].item<float>() << std::endl;
    }
    std::cout << "---------------------------------" << std::endl;
  } else if (env.getCurrentPlayer() == playerIdx_ && reply.count("adv") > 0) {
    std::cout << "@decideMove, step: " << env.numStep() << std::endl;
    auto adv = reply.at("adv");
    auto legal_move = reply.at("legal_move");

    std::cout << "@decideMove last move: " << env.getMove(env.getLastAction()).ToString() << std::endl;
    for (int action = 0; action < 20; ++action) {
      if (legal_move[action].item<int>() == 0) {
        continue;
      }
      std::cout << std::fixed;
      std::cout << std::setprecision(4)
                << "@decideMove action: " << env.getMove(action).ToString()
                << ", adv: " << adv[action].item<float>() << std::endl;
    }
    std::cout << "---------------------------------" << std::endl;
  }

  // get the real action
  auto curPlayer = env.getCurrentPlayer();
  std::unique_ptr<hle::HanabiMove> move;

  auto& state = env.getHleState();
  move = std::make_unique<hle::HanabiMove>(state.ParentGame()->GetMove(action));
  if (shuffleColor_ && move->MoveType() == hle::HanabiMove::Type::kRevealColor) {
    int realColor = invColorPermute_[move->Color()];
    move->SetColor(realColor);
  }

  // collect stats
  if (move->MoveType() == hle::HanabiMove::kPlay) {
    auto cardBelief = perCardPrivV0_[move->CardIndex()];
    auto [colorKnown, rankKnown] = analyzeCardBelief(cardBelief);

    if (colorKnown && rankKnown) {
      ++bothKnown_;
    } else if (colorKnown) {
      ++colorKnown_;
    } else if (rankKnown) {
      ++rankKnown_;
    } else {
      ++noneKnown_;
    }
  }

  if (offBelief_) {
    const auto& hand = fictState_->Hands()[playerIdx_];
    bool success = true;

    if (success) {
      auto& deck = fictState_->Deck();
      deck.PutCardsBack(hand.Cards());
      // For now, skip belief sampling to avoid compilation issues
      // deck.DealCards(sampledCards_);
      // fictState_->Hands()[playerIdx_].SetCards(sampledCards_);
      ++successFict_;
    }
    validFict_ = success;
    ++totalFict_;

    if (curPlayer != playerIdx_) {
      auto partner = partners_[curPlayer];
      assert(partner != nullptr);
      // it is not my turn, I need to re-evaluate my partner on
      // the fictitious transition
      auto partnerInput = observe(
          *fictState_,
          partner->playerIdx_,
          partner->shuffleColor_,
          partner->colorPermute_,
          partner->invColorPermute_,
          partner->hideAction_,
          partner->aux_,
          partner->sad_);
      // add features such as eps and temperature
      partnerInput["eps"] = torch::tensor(partner->playerEps_);
      if (partner->playerTemp_.size() > 0) {
        partnerInput["temperature"] = torch::tensor(partner->playerTemp_);
      }
      addHid(partnerInput, partner->prevHidden_);
      partner->actReply_ = partner->callModel("act", partnerInput);
    }
  }
  return move;
}

void R2D2ActorSimple::fictAct(const HanabiEnv& env) {
  if (!offBelief_) {
    return;
  }

  auto fictMove = env.lastMove();
  if (env.getCurrentPlayer() != playerIdx_) {
    auto fictReply = actReply_;
    auto action = fictReply["a"].item<int64_t>();
    fictMove = env.getMove(action);
  }

  auto [fictR, fictTerm] = applyMove(*fictState_, fictMove, false);

  // submit network call to compute target value
  auto fictInput = observe(
      *fictState_,
      playerIdx_,
      shuffleColor_,
      colorPermute_,
      invColorPermute_,
      hideAction_,
      aux_,
      sad_);
  addHid(fictInput, hidden_);

  // the hidden is new, so we are good
  fictInput["reward"] = torch::tensor(fictR);
  fictInput["terminal"] = torch::tensor(float(fictTerm));
  if (playerTemp_.size() > 0) {
    fictInput["temperature"] = torch::tensor(playerTemp_);
  }

  targetReply_ = callModel("compute_target", fictInput);
}

void R2D2ActorSimple::observeAfterAct(const HanabiEnv& env) {
  // Nothing to do in simple version
  return;
}

void R2D2ActorSimple::storeTrajectory(const HanabiEnv& env) {
  // Nothing to do in simple version
  return;
} 