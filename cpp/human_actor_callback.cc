#include "cpp/human_actor_callback.h"
#include <iostream>
#include <string>
#include <limits>
#include <sstream>

void HumanActorCallback::reset(const HanabiEnv& env) {
    stage_ = Stage::ObserveBeforeAct;
}

std::unique_ptr<hle::HanabiMove> HumanActorCallback::next(const HanabiEnv& env) {
  if (stage_ == Stage::ObserveBeforeAct) {
    std::cout << "HumanActorCallback::next() - Player " << playerIdx_ << ", Stage: ObserveBeforeAct" << std::endl;
    observeBeforeAct(env);
    stage_ = Stage::DecideMove;
    return nullptr;
  }

  if (stage_ == Stage::DecideMove) {
    std::cout << "HumanActorCallback::next() - Player " << playerIdx_ << ", Stage: DecideMove" << std::endl;
    auto move = decideMove(env);
    stage_ = Stage::ObserveBeforeAct;
    return move;
  }

  assert(false);
  return nullptr;
}

void HumanActorCallback::observeBeforeAct(const HanabiEnv& env) {
  // Basic observation - similar to what R2D2ActorSimple does
  // This ensures that the environment state is properly observed
  const auto& state = env.getHleState();
  
  // Create an observation to ensure state consistency
  auto obs = hle::HanabiObservation(state, playerIdx_, true);
  
  // Get legal moves to ensure state is properly initialized
  const auto& legalMoves = obs.LegalMoves();
  
  // This is a minimal observation that ensures the environment state
  // is properly observed, similar to what R2D2ActorSimple does
  // but without the complex neural network calls
}

std::unique_ptr<hle::HanabiMove> HumanActorCallback::decideMove(const HanabiEnv& env) {
  // If it's not this player's turn, return a Deal move like R2D2ActorSimple does
  if (env.getCurrentPlayer() != playerIdx_) {
    // Create a Deal move - this is what R2D2ActorSimple returns when it's not the current player
    const auto& state = env.getHleState();
    auto dealMove = std::make_unique<hle::HanabiMove>(hle::HanabiMove::Type::kDeal, -1, -1, -1, -1);
    return dealMove;
  }

  // If callback is not set, fall back to console input
  if (!actionCallback_) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "YOUR TURN (Player " << playerIdx_ << ")" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    // Print current game state
    std::cout << getGameStateString(env);

    // Get legal moves from current obs
    const auto& state = env.getHleState();
    auto obs = hle::HanabiObservation(state, playerIdx_, true);
    const auto& legalMoves = obs.LegalMoves();
    
    std::cout << "\n=== LEGAL MOVES ===" << std::endl;
    for (size_t i = 0; i < legalMoves.size(); ++i) {
      std::cout << i << ": " << legalMoves[i].ToString() << std::endl;
    }

    // Get user choice
    int choice = -1;
    while (choice < 0 || choice >= (int)legalMoves.size()) {
      std::cout << "\nEnter your choice (0-" << legalMoves.size() - 1 << "): ";
      std::string input;
      std::getline(std::cin, input);
      
      try {
        choice = std::stoi(input);
      } catch (const std::exception& e) {
        std::cout << "Invalid input. Please enter a number." << std::endl;
        choice = -1;
        continue;
      }
      
      if (choice < 0 || choice >= (int)legalMoves.size()) {
        std::cout << "Invalid choice. Please enter a number between 0 and " << legalMoves.size() - 1 << "." << std::endl;
        if(choice == -1) {
          exit(0);
        }
      }
    }

    // Create the move
    auto move = std::make_unique<hle::HanabiMove>(legalMoves[choice]);
    std::cout << "You chose: " << move->ToString() << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    return move;
  }

  // Use Python callback
  try {
    // Get game state and legal moves as strings
    std::string gameStateStr = getGameStateString(env);
    const auto& state = env.getHleState();
    auto obs = hle::HanabiObservation(state, playerIdx_, true);
    const auto& legalMoves = obs.LegalMoves();
    std::vector<std::string> legalMovesStr = getLegalMovesStrings(legalMoves);

    // Call Python callback
    py::gil_scoped_acquire gil;
    py::object result = actionCallback_(gameStateStr, legalMovesStr);
    
    // Extract the chosen move index from Python
    int choice = result.cast<int>();
    
    if (choice < 0 || choice >= (int)legalMoves.size()) {
      throw std::runtime_error("Invalid choice returned from Python callback");
    }

    // Create the move
    auto move = std::make_unique<hle::HanabiMove>(legalMoves[choice]);
    return move;
  } catch (const std::exception& e) {
    std::cerr << "Error in Python callback: " << e.what() << std::endl;
    // Fall back to console input
    return decideMove(env);
  }
}

void HumanActorCallback::observeAfterAct(const HanabiEnv& env) {
  // Not needed for human actors
}

std::string HumanActorCallback::getGameStateString(const HanabiEnv& env) {
  const auto& state = env.getHleState();
  std::ostringstream oss;
  
  oss << "\n=== GAME STATE ===" << std::endl;
  oss << "Score: " << state.Score() << "/25" << std::endl;
  oss << "Life tokens: " << state.LifeTokens() << "/3" << std::endl;
  oss << "Information tokens: " << state.InformationTokens() << "/8" << std::endl;
  
  // Print fireworks
  oss << "Fireworks: ";
  for (int color = 0; color < 5; ++color) {
    oss << state.Fireworks()[color] << " ";
  }
  oss << std::endl;

  // Print hands
  oss << "\n=== HANDS ===" << std::endl;
  auto obs = hle::HanabiObservation(state, playerIdx_, true);
  const auto& hands = obs.Hands();
  for (int player = 0; player < numPlayer_; ++player) {
    oss << "Player " << player << " hand: ";
    const auto& hand = hands[player];
    if (player != playerIdx_) {
      // Show other players' cards
      for (size_t i = 0; i < hand.Cards().size(); ++i) {
        const auto& card = hand.Cards()[i];
        oss << card.ToString() << " ";
      }
      oss << std::endl;
    } 

    oss << hand.Cards().size() << " cards" << std::endl;
    // Show hand knowledge for self
    oss << "  Knowledge:" << std::endl;
    const auto& knowledge = hand.Knowledge();
    for (size_t i = 0; i < knowledge.size(); ++i) {
        oss << "    " << knowledge[i].ToString() << std::endl;
    }
    
  }

  // Print discard pile
  oss << "\n=== DISCARD PILE ===" << std::endl;
  const auto& discardPile = state.DiscardPile();
  for (const auto& card : discardPile) {
    oss << card.ToString() << " ";
  }
  oss << std::endl;

  // Print last moves
  oss << "\n=== LAST MOVES ===" << std::endl;
  auto obs2 = hle::HanabiObservation(state, playerIdx_, true);
  const auto& lastMoves = obs2.LastMoves();
  for (const auto& move : lastMoves) {
    if(move.player == -1) continue;
    int absolutePlayer = (playerIdx_ + move.player) % numPlayer_;
    oss << "Player " << absolutePlayer << ": " << move.move.ToString() << std::endl;
  }

  return oss.str();
}

std::vector<std::string> HumanActorCallback::getLegalMovesStrings(const std::vector<hle::HanabiMove>& legalMoves) {
  std::vector<std::string> result;
  
  for (size_t i = 0; i < legalMoves.size(); ++i) {
    std::ostringstream oss;
    oss << i << ": " << legalMoves[i].ToString();
    
    // Add more descriptive information
    switch (legalMoves[i].MoveType()) {
      case hle::HanabiMove::Type::kPlay:
        oss << " (Play card " << legalMoves[i].CardIndex() << ")";
        break;
      case hle::HanabiMove::Type::kDiscard:
        oss << " (Discard card " << legalMoves[i].CardIndex() << ")";
        break;
      case hle::HanabiMove::Type::kRevealColor:
        oss << " (Hint color " << legalMoves[i].Color() << " to player " << legalMoves[i].TargetOffset() << ")";
        break;
      case hle::HanabiMove::Type::kRevealRank:
        oss << " (Hint rank " << legalMoves[i].Rank() << " to player " << legalMoves[i].TargetOffset() << ")";
        break;
      default:
        break;
    }
    
    result.push_back(oss.str());
  }
  
  return result;
} 