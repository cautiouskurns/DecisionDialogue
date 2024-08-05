import random
import json

class GameConfig:
    def __init__(self):
        self.load_game_config()

    def load_game_config(self):
        with open('game_config_test.json', 'r') as f:
            config = json.load(f)
        self.player_health = config['initial_player_health']
        self.player_friendly = config['initial_player_friendly']
        self.player_has_item = config['initial_player_has_item']
        self.time_of_day = random.choice(config['time_options'])
        self.location = random.choice(config['location_options'])

class NPC:
    def __init__(self, name):
        self.name = name
        self.load_npc_config()

    def load_npc_config(self):
        with open('npc_config_test.json', 'r') as f:
            config = json.load(f)
        self.health = config['initial_health']
        self.friendly = random.choice(config['friendly_options'])
        self.has_item = random.choice(config['has_item_options'])
        self.mood = random.choice(config['mood_options'])

    def interact(self, player_action):
        # For now, just return a random response
        responses = [
            f"{self.name} looks at you curiously.",
            f"{self.name} nods slowly.",
            f"{self.name} seems unsure how to respond.",
            f"{self.name} smiles warmly.",
            f"{self.name} frowns slightly."
        ]
        return random.choice(responses)

class Game:
    def __init__(self):
        self.config = GameConfig()
        self.npc = NPC("Guardian")

    def start(self):
        print("Welcome to 'Decisions n Dialogue'!")
        print(f"You find yourself in the {self.config.location}. It's currently {self.config.time_of_day}.")
        print(f"You encounter {self.npc.name}.")

        while True:
            action = input("What would you like to do? (talk/leave): ").lower()
            if action == 'talk':
                response = self.npc.interact(action)
                print(response)
            elif action == 'leave':
                print("You decide to leave. Game over.")
                break
            else:
                print("Invalid action. Please choose 'talk' or 'leave'.")

if __name__ == "__main__":
    game = Game()
    game.start()