#include "cpp/play_game.h"
#include <iostream>

void PlayGame::runGame() {
  std::cout << "=== PlayGame::runGame() started ===" << std::endl;
  
  // Reset the game and all actors
  std::cout << "Resetting game and actors..." << std::endl;
  reset();
  std::cout << "Reset completed" << std::endl;
  
  int step_count = 0;
  
  // Main game loop
  std::cout << "Entering main game loop..." << std::endl;
  while (!env_->terminated()) {
    step_count++;
    std::cout << "=== Step " << step_count << " ===" << std::endl;
    std::cout << "Current player: " << env_->getCurrentPlayer() << std::endl;
    std::cout << "Game terminated: " << (env_->terminated() ? "true" : "false") << std::endl;
    
    // Check if all actors are ready
    std::cout << "Checking if all actors are ready..." << std::endl;
    bool allActorReady = true;
    for (size_t i = 0; i < actors_.size(); ++i) {
      std::cout << "  Actor " << i << " ready: " << (actors_[i]->ready() ? "true" : "false") << std::endl;
      if (!actors_[i]->ready()) {
        allActorReady = false;
        break;
      }
    }
    
    if (!allActorReady) {
      std::cout << "Not all actors ready, continuing..." << std::endl;
      // Wait for all actors to be ready
      continue;
    }
    
    std::cout << "All actors ready, getting moves..." << std::endl;
    
    // Get moves from all actors
    std::vector<std::unique_ptr<hle::HanabiMove>> moves;
    for (size_t i = 0; i < actors_.size(); ++i) {
      std::cout << "  Getting move from actor " << i << "..." << std::endl;
      try {
        auto move = actors_[i]->next(*env_);
        std::cout << "  Actor " << i << " returned move: " << (move ? move->ToString() : "nullptr") << std::endl;
        moves.push_back(std::move(move));
      } catch (const std::exception& e) {
        std::cout << "  ERROR: Actor " << i << " threw exception: " << e.what() << std::endl;
        throw;
      } catch (...) {
        std::cout << "  ERROR: Actor " << i << " threw unknown exception" << std::endl;
        throw;
      }
    }
    
    std::cout << "Got moves from all actors, executing move..." << std::endl;
    
    // Execute the move for the current player
    if (!env_->terminated()) {
      auto current_player = env_->getCurrentPlayer();
      std::cout << "Current player: " << current_player << std::endl;
      std::cout << "Moves vector size: " << moves.size() << std::endl;
      
      if (current_player < moves.size()) {
        auto& move = moves[current_player];
        if (move != nullptr) {
          std::cout << "Executing move: " << move->ToString() << std::endl;
          try {
            env_->step(*move);
            std::cout << "Move executed successfully" << std::endl;
          } catch (const std::exception& e) {
            std::cout << "ERROR: Failed to execute move: " << e.what() << std::endl;
            throw;
          } catch (...) {
            std::cout << "ERROR: Failed to execute move with unknown exception" << std::endl;
            throw;
          }
        } else {
          std::cout << "Move is nullptr, skipping" << std::endl;
        }
      } else {
        std::cout << "ERROR: Current player " << current_player << " >= moves size " << moves.size() << std::endl;
      }
    } else {
      std::cout << "Game already terminated, skipping move execution" << std::endl;
    }
    
    std::cout << "Step " << step_count << " completed" << std::endl;
  }
  
  // Game completed
  std::cout << "=== Game completed! ===" << std::endl;
  std::cout << "Final score: " << env_->lastEpisodeScore() << std::endl;
  std::cout << "Total steps: " << env_->numStep() << std::endl;
  std::cout << "=== PlayGame::runGame() finished ===" << std::endl;
}

void PlayGame::reset() {
  std::cout << "PlayGame::reset() started" << std::endl;
  
  // Reset the environment
  std::cout << "Resetting environment..." << std::endl;
  try {
    env_->reset();
    std::cout << "Environment reset successful" << std::endl;
  } catch (const std::exception& e) {
    std::cout << "ERROR: Failed to reset environment: " << e.what() << std::endl;
    throw;
  } catch (...) {
    std::cout << "ERROR: Failed to reset environment with unknown exception" << std::endl;
    throw;
  }
  
  // Reset all actors
  std::cout << "Resetting actors..." << std::endl;
  for (size_t i = 0; i < actors_.size(); ++i) {
    std::cout << "  Resetting actor " << i << "..." << std::endl;
    try {
      actors_[i]->reset(*env_);
      std::cout << "  Actor " << i << " reset successful" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "  ERROR: Failed to reset actor " << i << ": " << e.what() << std::endl;
      throw;
    } catch (...) {
      std::cout << "  ERROR: Failed to reset actor " << i << " with unknown exception" << std::endl;
      throw;
    }
  }
  
  std::cout << "PlayGame::reset() completed" << std::endl;
} 