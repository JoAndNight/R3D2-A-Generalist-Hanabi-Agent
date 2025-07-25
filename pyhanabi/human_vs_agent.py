#!/usr/bin/env python3
"""
Human vs R3D2 Agent Hanabi Game

This script allows a human player to play against an R3D2 agent.
The human player is Player 0, and the R3D2 agent is Player 1.
"""

import os
import sys
import numpy as np
import torch

lib_path = os.path.dirname(os.path.abspath(__file__))
print(lib_path)
sys.path.append(lib_path)
root_dir = os.path.dirname(lib_path)  
build_path = os.path.join(root_dir, "build")

sys.path.append(build_path)
print(sys.path)
import rela
import hanalearn
import utils
import r2d2


def create_human_action_callback():
    """
    Create a callback function for human player actions.
    This function will be called by the C++ HumanActorCallback when it's the human's turn.
    
    Returns:
        A function that takes a json string, displays the game state and legal moves,
        and returns the chosen action index.
    """
    import json
    def human_action_callback(json_str):
        """
        Callback function for human player actions.
        Args:
            json_str: JSON string containing game state and legal moves
        Returns:
            The index of the chosen move from legal_moves
        """
        data = json.loads(json_str)
        print("\n" + "="*60)
        print("YOUR TURN!")
        print("="*60)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("\nLegal moves:")
        for i, move in enumerate(data["legal_moves"]):
            print(f"{i}: {move}")
        while True:
            try:
                choice = int(input(f"\nEnter your choice (0-{len(data['legal_moves'])-1}): "))
                if 0 <= choice < len(data["legal_moves"]):
                    print(f"You chose: {data['legal_moves'][choice]}")
                    return choice
                elif choice == -1:
                    return choice
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(data['legal_moves'])-1}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                sys.exit(0)
    return human_action_callback


def create_game_env():
    """
    Create the Hanabi game environment.
    
    Returns:
        A HanabiEnv instance configured for 2-player game
    """
    env_config = {
        "players": "2",
        "max_life_tokens": "3",
        "max_information_tokens": "8",
        "colors": "5",
        "ranks": "5",
        "hand_size": "5",
        "max_deck_size": "50",
        "observation_type": "1",
        "random_start_player": "0",
        "seed": "1",
    }
    
    return hanalearn.HanabiEnv(env_config, 80, False)


def load_r3d2_agent(weight_path):
    """
    Load the R3D2 agent from the specified weight file.
    
    Args:
        weight_path: Path to the R3D2 model weights
        
    Returns:
        The loaded R3D2 agent
    """
    print(f"Loading R3D2 agent from: {weight_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, cfg = utils.load_agent(weight_path, {"device": device})
    agent.eval()
    print("R3D2 agent loaded successfully!")
    return agent


def create_actors(env, r3d2_agent, human_callback=None):
    """
    Create the actors for the game.
    
    Args:
        env: The Hanabi environment
        r3d2_agent: The R3D2 agent
        human_callback: Optional callback function for human actions
        
    Returns:
        List of actors wrapped for PlayGame
    """
    num_player = 2
    
    # Create human actor with callback (Player 0) - using HumanActorCallback
    human_actor = hanalearn.HumanActorCallback(num_player, 0)
    
    # Set up partners (simplified for 2-player game)
    human_actor.set_partners([None, human_actor])
    
    # Register callback if provided
    if human_callback:
        human_actor.set_action_callback(human_callback)
    
    # Create R3D2 actor (Player 1) - using positional arguments like in play.py
    r3d2_actor = hanalearn.R2D2ActorSimple(
        r3d2_agent, 
        num_player, 
        1, 
        False, 
        False, 
        False
    )
    
    # Set up partners for R3D2 actor
    r3d2_actor.set_partners([r3d2_actor, None])
    
    # Create actor wrappers for PlayGame
    human_wrapper = hanalearn.HumanActorCallbackWrapper(human_actor)
    r3d2_wrapper = hanalearn.R2D2ActorWrapper(r3d2_actor)
    
    return [human_wrapper, r3d2_wrapper]


def play_human_vs_agent():
    """
    Main function to play a game between human and R3D2 agent.
    """
    print("="*60)
    print("HANABI: Human vs R3D2 Agent")
    print("="*60)
    print("You are Player 0, R3D2 agent is Player 1")
    print("Good luck!")
    print()
    
    # Default weight path
    weight_path = "final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw"
    
    # Check if weight file exists
    if not os.path.exists(weight_path):
        print(f"Error: Weight file not found at {weight_path}")
        print("Please make sure the R3D2 model weights are available.")
        return
    
    try:
        # Load R3D2 agent
        r3d2_agent = load_r3d2_agent(weight_path)
        
        # Create game environment
        env = create_game_env()
        
        # Create human action callback
        human_callback = create_human_action_callback()
        
        # Create actors
        actors = create_actors(env, r3d2_agent, human_callback)
        
        # Create PlayGame instance
        play_game = hanalearn.PlayGame(env, actors)
        
        # Start the game
        print("\nStarting the game...")
        play_game.run_game()
        
        # Display final results
        print("\n" + "="*60)
        print("GAME COMPLETED!")
        print("="*60)
        print(f"Final Score: {play_game.get_score()}/25")
        print(f"Life Tokens Remaining: {play_game.get_life()}/3")
        print(f"Information Tokens: {play_game.get_info()}/8")
        print(f"Fireworks: {play_game.get_fireworks()}")
        
        if play_game.get_score() == 25:
            print("🎉 PERFECT GAME! Congratulations!")
        elif play_game.get_life() == 0:
            print("�� Game Over - Out of life tokens!")
        else:
            print("Game completed!")
            
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError during game: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    play_human_vs_agent()