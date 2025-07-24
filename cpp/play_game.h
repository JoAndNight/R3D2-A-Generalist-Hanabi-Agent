#pragma once

#include "cpp/hanabi_env.h"
#include "cpp/r2d2_actor_simple.h"
#include "cpp/human_actor.h"
#include <vector>
#include <memory>
#include <variant>



// Common interface for all actors
class ActorInterface {
 public:
  virtual ~ActorInterface() = default;
  virtual void reset(const HanabiEnv& env) = 0;
  virtual bool ready() const = 0;
  virtual bool stepDone() const = 0;
  virtual std::unique_ptr<hle::HanabiMove> next(const HanabiEnv& env) = 0;
};

// Wrapper for R2D2ActorSimple
class R2D2ActorWrapper : public ActorInterface {
 public:
  R2D2ActorWrapper(std::shared_ptr<R2D2ActorSimple> actor) : actor_(actor) {}
  
  void reset(const HanabiEnv& env) override { actor_->reset(env); }
  bool ready() const override { return actor_->ready(); }
  bool stepDone() const override { return actor_->stepDone(); }
  std::unique_ptr<hle::HanabiMove> next(const HanabiEnv& env) override { return actor_->next(env); }
  
 private:
  std::shared_ptr<R2D2ActorSimple> actor_;
};

// Wrapper for HumanActor
class HumanActorWrapper : public ActorInterface {
 public:
  HumanActorWrapper(std::shared_ptr<HumanActor> actor) : actor_(actor) {}
  
  void reset(const HanabiEnv& env) override { actor_->reset(env); }
  bool ready() const override { return actor_->ready(); }
  bool stepDone() const override { return actor_->stepDone(); }
  std::unique_ptr<hle::HanabiMove> next(const HanabiEnv& env) override { return actor_->next(env); }
  
 private:
  std::shared_ptr<HumanActor> actor_;
};


class PlayGame {
 public:
  PlayGame(
      std::shared_ptr<HanabiEnv> env,
      std::vector<std::shared_ptr<ActorInterface>> actors)
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
  std::vector<std::shared_ptr<ActorInterface>> actors_;
}; 