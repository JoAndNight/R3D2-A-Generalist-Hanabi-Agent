#include "cpp/human_actor.h"
#include <iostream>
#include <string>
#include <limits>

void HumanActor::reset(const HanabiEnv& env) {
    stage_ = Stage::ObserveBeforeAct;
}

std::unique_ptr<hle::HanabiMove> HumanActor::next(const HanabiEnv& env) {
  if (stage_ == Stage::ObserveBeforeAct) {
    std::cout << "HumanActor::next() - Player " << playerIdx_ << ", Stage: ObserveBeforeAct" << std::endl;
    observeBeforeAct(env);
    stage_ = Stage::DecideMove;
    return nullptr;
  }

  if (stage_ == Stage::DecideMove) {
    std::cout << "HumanActor::next() - Player " << playerIdx_ << ", Stage: DecideMove" << std::endl;
    auto move = decideMove(env);
    stage_ = Stage::ObserveBeforeAct;
    return move;
  }


  assert(false);
  return nullptr;
}

void HumanActor::observeBeforeAct(const HanabiEnv& env) {
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

std::unique_ptr<hle::HanabiMove> HumanActor::decideMove(const HanabiEnv& env) {
  // If it's not this player's turn, return a Deal move like R2D2ActorSimple does
  if (env.getCurrentPlayer() != playerIdx_) {
    // Create a Deal move - this is what R2D2ActorSimple returns when it's not the current player
    const auto& state = env.getHleState();
    auto dealMove = std::make_unique<hle::HanabiMove>(hle::HanabiMove::Type::kDeal, -1, -1, -1, -1);
    return dealMove;
  }

  std::cout << "\n" << std::string(50, '=') << std::endl;
  std::cout << "YOUR TURN (Player " << playerIdx_ << ")" << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  // Print current game state
  printGameState(env);

  // Get legal moves from current obs
  const auto& state = env.getHleState();
  auto obs = hle::HanabiObservation(state, playerIdx_, true);
  const auto& legalMoves = obs.LegalMoves();
  printLegalMoves(legalMoves);

  // Get user choice
  int choice = getUserActionChoice(legalMoves);

  // Create the move
  auto move = std::make_unique<hle::HanabiMove>(legalMoves[choice]);
  
  std::cout << "You chose: " << move->ToString() << std::endl;
  std::cout << std::string(50, '=') << std::endl;

  return move;
}

void HumanActor::printGameState(const HanabiEnv& env) {
  const auto& state = env.getHleState();

  
  std::cout << "\n=== GAME STATE ===" << std::endl;
  std::cout << "Score: " << state.Score() << "/25" << std::endl;
  std::cout << "Life tokens: " << state.LifeTokens() << "/3" << std::endl;
  std::cout << "Information tokens: " << state.InformationTokens() << "/8" << std::endl;
  
  // Print fireworks
  std::cout << "Fireworks: ";
  for (int color = 0; color < 5; ++color) {
    std::cout << state.Fireworks()[color] << " ";
  }
  std::cout << std::endl;

  // Print hands
  std::cout << "\n=== HANDS ===" << std::endl;
  for (int player = 0; player < numPlayer_; ++player) {
    std::cout << "Player " << player << " hand: ";
    if (player != playerIdx_) {
      // Show own cards
      const auto& hand = state.Hands()[player];
      for (size_t i = 0; i < hand.Cards().size(); ++i) {
        const auto& card = hand.Cards()[i];
        std::cout << card.ToString() << " ";
      }
    } else {
      // Show number of cards for other players
      std::cout << state.Hands()[player].Cards().size() << " cards";
    }
    std::cout << std::endl;
  }

  // Print discard pile
  std::cout << "\n=== DISCARD PILE ===" << std::endl;
  const auto& discardPile = state.DiscardPile();
  for (const auto& card : discardPile) {
    std::cout << card.ToString() << " ";
  }
  std::cout << std::endl;

  // Print last moves
  std::cout << "\n=== LAST MOVES ===" << std::endl;
  auto obs = hle::HanabiObservation(state, playerIdx_, true);
  const auto& lastMoves = obs.LastMoves();
  for (const auto& move : lastMoves) {
    if(move.player == -1) continue;
    int absolutePlayer = (playerIdx_ + move.player) % numPlayer_;
    std::cout << "Player " << absolutePlayer << ": " << move.move.ToString() << std::endl;
  }
}

void HumanActor::printLegalMoves(const std::vector<hle::HanabiMove>& legalMoves) {
  std::cout << "\n=== LEGAL MOVES ===" << std::endl;
  for (size_t i = 0; i < legalMoves.size(); ++i) {
    std::cout << i << ": " << legalMoves[i].ToString();
    
    // Add more descriptive information
    switch (legalMoves[i].MoveType()) {
      case hle::HanabiMove::Type::kPlay:
        std::cout << " (Play card " << legalMoves[i].CardIndex() << ")";
        break;
      case hle::HanabiMove::Type::kDiscard:
        std::cout << " (Discard card " << legalMoves[i].CardIndex() << ")";
        break;
      case hle::HanabiMove::Type::kRevealColor:
        std::cout << " (Hint color " << legalMoves[i].Color() << " to player " << legalMoves[i].TargetOffset() << ")";
        break;
      case hle::HanabiMove::Type::kRevealRank:
        std::cout << " (Hint rank " << legalMoves[i].Rank() << " to player " << legalMoves[i].TargetOffset() << ")";
        break;
      default:
        break;
    }
    std::cout << std::endl;
  }
}

int HumanActor::getUserActionChoice(const std::vector<hle::HanabiMove>& legalMoves) {
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
  
  return choice;
}