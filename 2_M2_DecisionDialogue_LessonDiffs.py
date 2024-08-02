import json
import random

class GameConfig:
    def __init__(self):
        self.load_game_config()
        
    def load_game_config(self):
        with open('game_config.json', 'r') as f:
            config = json.load(f)
        self.player_health = config['initial_player_health']
        self.player_friendly = config['initial_player_friendly']
        self.player_has_item = config['initial_player_has_item']
        self.time_options = config['time_options']
        self.location_options = config['location_options']
        self.time_of_day = random.choice(self.time_options)
        self.location = random.choice(self.location_options)

class Game:
    def __init__(self):
        self.config = GameConfig()

    def start(self):
        print(f"Welcome to 'Decisions n Dialogue'!")
        print(f"You find yourself in the {self.config.location} during the {self.config.time_of_day}.")
        print(f"Your health: {self.config.player_health}")
        print(f"You are {'friendly' if self.config.player_friendly else 'unfriendly'}")
        print(f"You {'have' if self.config.player_has_item else 'do not have'} an item")

# Main execution
if __name__ == "__main__":
    game = Game()
    game.start()