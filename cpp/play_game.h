#pragma once

#include "cpp/hanabi_env.h"
#include "cpp/r2d2_actor_simple.h"
#include <vector>
#include <memory>

class PlayGame {
 public:
  PlayGame(
      std::shared_ptr<HanabiEnv> env,
      std::vector<std::shared_ptr<R2D2ActorSimple>> actors)
      : env_(std::move(env))
      , actors_(std::move(actors)) {
    assert(env_ != nullptr);
    assert(actors_.size() > 0);
    assert(actors_.size() == env_->getNumPlayers());
  }

  // Run a single game to completion
  void runGame();

  // Get the final score of the game
  int getScore() const {
    return env_->lastEpisodeScore();
  }

  // Get the final life tokens
  int getLife() const {
    return env_->getLife();
  }

  // Get the final information tokens
  int getInfo() const {
    return env_->getInfo();
  }

  // Get the final fireworks
  std::vector<int> getFireworks() const {
    return env_->getFireworks();
  }

  // Check if the game is terminated
  bool isTerminated() const {
    return env_->terminated();
  }

  // Get the number of steps taken
  int getNumSteps() const {
    return env_->numStep();
  }

  // Reset the game and actors
  void reset();

 private:
  std::shared_ptr<HanabiEnv> env_;
  std::vector<std::shared_ptr<R2D2ActorSimple>> actors_;
}; 