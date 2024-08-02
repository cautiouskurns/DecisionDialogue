import json
import random
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Existing code from Lesson 1
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

# New code for Lesson 2
class NPCDecisionTree:
    def __init__(self):
        self.clf = self.train_decision_tree()

    def train_decision_tree(self):
        # Simple training data
        X = np.array([
            [1, 1, 100, 0, 0, 0],  # friendly, has_item, full_health, morning, forest
            [0, 0, 50, 1, 1, 1],   # unfriendly, no_item, half_health, night, castle
            [1, 0, 100, 2, 2, 2],  # friendly, no_item, full_health, afternoon, village
            [0, 1, 20, 3, 3, 3],   # unfriendly, has_item, low_health, evening, cave
        ])
        y = np.array([1, 0, 2, 3])  # 0: attack, 1: greet, 2: trade, 3: flee

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        return clf

    def decide_action(self, player_friendly, player_has_item, npc_health, time_of_day, location):
        features = [
            int(player_friendly),
            int(player_has_item),
            npc_health,
            ['morning', 'afternoon', 'evening', 'night'].index(time_of_day),
            ['forest', 'village', 'castle', 'cave'].index(location),
            random.randint(0, 3)  # Random mood as an additional feature
        ]
        action = self.clf.predict([features])[0]
        return ['attack', 'greet', 'trade', 'flee'][action]

class NPC:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.decision_tree = NPCDecisionTree()

    def interact(self, player_friendly, player_has_item, time_of_day, location):
        action = self.decision_tree.decide_action(player_friendly, player_has_item, self.health, time_of_day, location)
        return f"{self.name} decides to {action}."

class Game:
    def __init__(self):
        self.config = GameConfig()
        self.npc = NPC("Guardian")

    def start(self):
        print(f"Welcome to 'Decisions n Dialogue'!")
        print(f"You find yourself in the {self.config.location} during the {self.config.time_of_day}.")
        print(f"Your health: {self.config.player_health}")
        print(f"You are {'friendly' if self.config.player_friendly else 'unfriendly'}")
        print(f"You {'have' if self.config.player_has_item else 'do not have'} an item")
        
        # New code to demonstrate NPC interaction
        npc_response = self.npc.interact(
            self.config.player_friendly,
            self.config.player_has_item,
            self.config.time_of_day,
            self.config.location
        )
        print(f"\nYou encounter an NPC named {self.npc.name}.")
        print(npc_response)

# Main execution
if __name__ == "__main__":
    game = Game()
    game.start()