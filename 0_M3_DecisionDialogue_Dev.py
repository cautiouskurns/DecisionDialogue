import random
import json
import csv  # New import for Lesson 3
from datetime import datetime  # New import for Lesson 3

# Existing GameConfig class (unchanged)
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

# Existing DecisionTree class (unchanged)
class DecisionTree:
    def __init__(self):
        self.tree = {
            'friendly': {
                True: {'has_item': {True: 'talk', False: 'give_item'}},
                False: {'player_has_item': {True: 'trade', False: 'ignore'}}
            }
        }

    def make_decision(self, npc_friendly, npc_has_item, player_has_item):
        decision = self.tree['friendly'][npc_friendly]
        if npc_friendly:
            decision = decision['has_item'][npc_has_item]
        else:
            decision = decision['player_has_item'][player_has_item]
        return decision

# Modified NPC class
class NPC:
    def __init__(self, name):
        self.name = name
        self.load_npc_config()
        self.decision_tree = DecisionTree()

    def load_npc_config(self):
        with open('npc_config_test.json', 'r') as f:
            config = json.load(f)
        self.health = config['initial_health']
        self.friendly = random.choice(config['friendly_options'])
        self.has_item = random.choice(config['has_item_options'])
        self.mood = random.choice(config['mood_options'])

    def interact(self, player_action, player_has_item):
        decision = self.decision_tree.make_decision(self.friendly, self.has_item, player_has_item)
        
        responses = {
            'talk': f"{self.name} engages in friendly conversation.",
            'give_item': f"{self.name} offers you an item.",
            'trade': f"{self.name} proposes a trade.",
            'ignore': f"{self.name} ignores you."
        }
        
        return decision, responses[decision]  # Modified to return both decision and response

# New class for Lesson 3
class DataCollector:
    def __init__(self):
        self.data = []

    def collect_data(self, player_action, npc_decision, npc_response, game_state):
        """
        Collect data from each player-NPC interaction.
        """
        interaction_data = {
            'timestamp': datetime.now().isoformat(),
            'player_action': player_action,
            'npc_decision': npc_decision,
            'npc_response': npc_response,
            'player_health': game_state.player_health,
            'player_friendly': game_state.player_friendly,
            'player_has_item': game_state.player_has_item,
            'time_of_day': game_state.time_of_day,
            'location': game_state.location
        }
        self.data.append(interaction_data)

    def save_data(self, filename='game_data.csv'):
        """
        Save collected data to a CSV file.
        """
        if not self.data:
            print("No data to save.")
            return

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = self.data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
        print(f"Data saved to {filename}")

    def load_data(self, filename='game_data.csv'):
        """
        Load data from a CSV file.
        """
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                self.data = list(reader)
            print(f"Data loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. No data loaded.")

# Modified Game class
class Game:
    def __init__(self):
        self.config = GameConfig()
        self.npc = NPC("Guardian")
        self.data_collector = DataCollector()  # New for Lesson 3

    def start(self):
        print("Welcome to 'Decisions n Dialogue'!")
        print(f"You find yourself in the {self.config.location}. It's currently {self.config.time_of_day}.")
        print(f"You encounter {self.npc.name}.")

        while True:
            action = input("What would you like to do? (talk/leave): ").lower()
            if action == 'talk':
                npc_decision, npc_response = self.npc.interact(action, self.config.player_has_item)
                print(npc_response)
                # New for Lesson 3: Collect data after each interaction
                self.data_collector.collect_data(action, npc_decision, npc_response, self.config)
            elif action == 'leave':
                print("You decide to leave. Game over.")
                break
            else:
                print("Invalid action. Please choose 'talk' or 'leave'.")

        # New for Lesson 3: Save collected data at the end of the game
        self.data_collector.save_data()

if __name__ == "__main__":
    game = Game()
    game.start()