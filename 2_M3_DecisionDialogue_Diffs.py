import random
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class GameConfig:
    def __init__(self):
        self.player_health = 100
        self.player_friendly = True
        self.player_has_item = False
        self.time_of_day = random.choice(["morning", "afternoon", "evening", "night"])
        self.location = random.choice(["forest", "village", "castle", "cave"])

class NPCTrainingData:
    def __init__(self):
        self.X = np.array([
            [1, 1, 1, 0, 0],  # player_friendly, player_has_item, morning, afternoon, forest
            [1, 0, 0, 1, 1],  # player_friendly, no_item, afternoon, village
            [0, 1, 0, 0, 1],  # not_friendly, has_item, evening, village
            [0, 0, 1, 0, 0],  # not_friendly, no_item, night, forest
        ])
        self.y = np.array(['talk', 'give_item', 'trade', 'ignore'])

class NPCDecisionTree:
    def __init__(self, training_data):
        self.training_data = training_data
        self.clf = self.train_decision_tree()

    def train_decision_tree(self):
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(self.training_data.X, self.training_data.y)
        return clf

    def decide_action(self, player_friendly, player_has_item, time_of_day, location):
        features = [
            int(player_friendly),
            int(player_has_item),
            1 if time_of_day in ['morning', 'night'] else 0,
            1 if time_of_day in ['afternoon', 'evening'] else 0,
            1 if location in ['forest', 'village'] else 0
        ]
        return self.clf.predict([features])[0]

class NPC:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.friendly = random.choice([True, False])
        self.training_data = NPCTrainingData()
        self.decision_tree = NPCDecisionTree(self.training_data)

    def interact(self, player_friendly, player_has_item, time_of_day, location):
        action = self.decision_tree.decide_action(player_friendly, player_has_item, time_of_day, location)
        
        responses = {
            'talk': f"{self.name} engages in friendly conversation.",
            'give_item': f"{self.name} offers you an item.",
            'trade': f"{self.name} proposes a trade.",
            'ignore': f"{self.name} ignores you."
        }
        
        return responses[action]

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
                response = self.npc.interact(
                    self.config.player_friendly,
                    self.config.player_has_item,
                    self.config.time_of_day,
                    self.config.location
                )
                print(response)
            elif action == 'leave':
                print("You decide to leave. Game over.")
                break
            else:
                print("Invalid action. Please choose 'talk' or 'leave'.")

if __name__ == "__main__":
    game = Game()
    game.start()