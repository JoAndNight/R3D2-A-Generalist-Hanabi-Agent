#!/usr/bin/env python3
"""
Human vs R3D2 Agent Hanabi Game with WebSocket Support

This script allows a human player to play against an R3D2 agent through a web interface.
The human player interacts via WebSocket, and the R3D2 agent plays automatically.
"""

import os
import sys
import numpy as np
import torch
import asyncio
import json
import websockets
from websockets.server import serve
import logging
import threading
from queue import Queue
import time

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebGameManager:
    """Manages the WebSocket connections and game state for web-based Hanabi games"""
    
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.connections = {}  # room_id -> websocket
        self.game_queues = {}  # room_id -> Queue for move responses
        self.message_queues = {}  # room_id -> Queue for outgoing messages
        self.server = None
        
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        logger.info("Available endpoints:")
        logger.info("  - ws://localhost:8080/ws/game/default-room")
        logger.info("  - ws://localhost:8080/ws/game/{any-room-id}")
        
        self.server = await serve(self.handle_websocket, self.host, self.port)
        logger.info(f"Server is running on ws://{self.host}:{self.port}")
        
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        room_id = None
        try:
            # Extract room ID from path
            # Path format: /ws/game/{roomId}
            path_parts = path.split('/')
            room_id = path_parts[-1] if len(path_parts) >= 4 else 'default-room'
            
            logger.info(f"New WebSocket connection to room: {room_id}")
            
            # Store connection
            self.connections[room_id] = websocket
            self.game_queues[room_id] = Queue()
            self.message_queues[room_id] = Queue()
            
            # Start game in a separate thread for this room
            game_thread = threading.Thread(
                target=play_game_in_thread,
                args=(self, room_id, "final_r3d2_ckpts/R3D2-2p/a/epoch3000.pthw"),
                daemon=True
            )
            game_thread.start()
            
            # Start message sender task
            asyncio.create_task(self._message_sender(room_id))
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message from room {room_id}: {data}")
                    
                    if data.get("type") == "move":
                        # Handle move from client
                        move_index = data.get("move_index")
                        if move_index is not None:
                            # Put the move in the queue for the callback to pick up
                            self.game_queues[room_id].put(move_index)
                        else:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": "Missing move_index"
                            }))
                    else:
                        # Echo back for testing
                        response = {
                            "type": "echo",
                            "message": data
                        }
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received from room {room_id}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for room {room_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        finally:
            # Clean up
            if room_id and room_id in self.connections:
                del self.connections[room_id]
            if room_id and room_id in self.game_queues:
                del self.game_queues[room_id]
            if room_id and room_id in self.message_queues:
                del self.message_queues[room_id]
    
    async def _message_sender(self, room_id):
        """Async task to send messages from queue"""
        while room_id in self.connections:
            try:
                # Use asyncio.to_thread to avoid blocking the event loop
                message = await asyncio.to_thread(self.message_queues[room_id].get)
                if room_id in self.connections:
                    await self.connections[room_id].send(json.dumps(message))
                    logger.info(f"Sent message to room {room_id}")
            except Exception as e:
                # Log error with full details
                import traceback
                logger.error(f"Error sending message to room {room_id}: {str(e)}")
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # If connection is broken, break the loop
                if room_id not in self.connections:
                    break
                # If it's a websocket connection error, break
                if "ConnectionClosed" in str(e) or "WebSocket" in str(e):
                    logger.info(f"WebSocket connection closed for room {room_id}, stopping message sender")
                    break
    
    def send_game_state(self, room_id, game_state):
        """Send game state to a specific room"""
        if room_id in self.message_queues:
            message = {
                "type": "game_state",
                "state": game_state
            }
            self.message_queues[room_id].put(message)
    
    def get_move_from_client(self, room_id):
        """Get move from client with timeout"""
        if room_id not in self.game_queues:
            logger.warning(f"Room {room_id} not found in game_queues")
            return -1  # Exit signal
        
        try:
            logger.info(f"Waiting for move from client in room {room_id}")
            return self.game_queues[room_id].get()
        except Exception as e:
            logger.warning(f"Timeout or error waiting for move from client in room {room_id}: {str(e)}")
            return -1  # Timeout or error


def create_web_human_action_callback(web_manager, room_id):
    """
    Create a callback function for human player actions via WebSocket.
    
    Args:
        web_manager: WebGameManager instance
        room_id: Room ID for the game
        
    Returns:
        A function that takes a json string, sends it to the web client,
        and returns the chosen action index from the client.
    """
    import json
    
    def web_human_action_callback(json_str):
        """
        Callback function for human player actions via WebSocket.
        Args:
            json_str: JSON string containing game state and legal moves
        Returns:
            The index of the chosen move from legal_moves
        """
        try:
            # Parse the JSON to get the game state
            data = json.loads(json_str)
            logger.info(f"Parsed game state for room {room_id}")
            
            # Send game state to web client
            web_manager.send_game_state(room_id, data)
            logger.info(f"Sent game state to room {room_id}")
            
            # Wait for move from client
            choice = web_manager.get_move_from_client(room_id)
            
            if choice == -1:
                logger.info("No move received from client, exiting...")
                return -1
            
            logger.info(f"Received move from client: {choice}")
            return choice
            
        except Exception as e:
            import traceback
            logger.error(f"Error in web callback for room {room_id}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return -1
    
    return web_human_action_callback


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


def play_game_in_thread(web_manager, room_id, weight_path):
    """
    Play a game in a separate thread to avoid blocking the WebSocket server.
    
    Args:
        web_manager: WebGameManager instance
        room_id: Room ID for the game
        weight_path: Path to R3D2 model weights
    """
    try:
        logger.info(f"Starting game for room {room_id}")
        
        # Load R3D2 agent
        r3d2_agent = load_r3d2_agent(weight_path)
        
        # Create game environment
        env = create_game_env()
        
        # Create web human action callback
        human_callback = create_web_human_action_callback(web_manager, room_id)
        
        # Create actors
        actors = create_actors(env, r3d2_agent, human_callback)
        
        # Create PlayGame instance
        play_game = hanalearn.PlayGame(env, actors)
        
        # Start the game
        logger.info(f"Starting the game for room {room_id}...")
        play_game.run_game()
        
        # Display final results
        final_score = play_game.get_score()
        final_life = play_game.get_life()
        final_info = play_game.get_info()
        final_fireworks = play_game.get_fireworks()
        
        logger.info(f"Game completed for room {room_id}")
        logger.info(f"Final Score: {final_score}/25")
        logger.info(f"Life Tokens Remaining: {final_life}/3")
        logger.info(f"Information Tokens: {final_info}/8")
        logger.info(f"Fireworks: {final_fireworks}")
        
        # Send final results to client
        final_message = {
            "type": "game_end",
            "score": final_score,
            "life": final_life,
            "info": final_info,
            "fireworks": final_fireworks,
            "perfect": final_score == 25,
            "game_over": final_life == 0
        }
        web_manager.send_game_state(room_id, final_message)
        
    except Exception as e:
        logger.error(f"Error during game for room {room_id}: {e}")
        import traceback
        traceback.print_exc()
        
        # Send error message to client
        error_message = {
            "type": "error",
            "message": f"Game error: {str(e)}"
        }
        web_manager.send_game_state(room_id, error_message)


async def main():
    """Main function to run the WebSocket server and handle games"""
    # Create web game manager
    web_manager = WebGameManager()
    
    # Start the WebSocket server
    await web_manager.start_server()
    
    # Keep the server running
    await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}") 