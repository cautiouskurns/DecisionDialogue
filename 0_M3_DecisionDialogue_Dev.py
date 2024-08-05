import random
import json
import csv
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

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

# Modified DecisionTree class (now using sklearn)
class DecisionTree:
    def __init__(self):
        self.clf = DecisionTreeClassifier(random_state=42)
        self.label_encoder = LabelEncoder()
        self.feature_names = ['npc_friendly', 'npc_has_item', 'player_has_item', 'time_of_day', 'location']
        self.trained = False

    def train(self, X, y):
        # Encode categorical variables
        X_encoded = np.array([self.label_encoder.fit_transform(x) for x in X.T]).T
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train the decision tree
        self.clf.fit(X_encoded, y_encoded)
        self.trained = True

    def make_decision(self, npc_friendly, npc_has_item, player_has_item, time_of_day, location):
        if not self.trained:
            # Fallback to the original simple decision tree if not trained
            if npc_friendly:
                return 'talk' if npc_has_item else 'give_item'
            else:
                return 'trade' if player_has_item else 'ignore'
        
        # Prepare the input for prediction
        X = np.array([[npc_friendly, npc_has_item, player_has_item, time_of_day, location]])
        X_encoded = np.array([self.label_encoder.transform(x) for x in X.T]).T
        
        # Make prediction
        decision_encoded = self.clf.predict(X_encoded)[0]
        return self.label_encoder.inverse_transform([decision_encoded])[0]

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

    def interact(self, player_action, player_has_item, time_of_day, location):
        decision = self.decision_tree.make_decision(self.friendly, self.has_item, player_has_item, time_of_day, location)
        
        responses = {
            'talk': f"{self.name} engages in friendly conversation.",
            'give_item': f"{self.name} offers you an item.",
            'trade': f"{self.name} proposes a trade.",
            'ignore': f"{self.name} ignores you."
        }
        
        return decision, responses[decision]

    def update_decision_tree(self, X, y):
        self.decision_tree.train(X, y)
        print(f"{self.name}'s decision tree has been updated!")

# Existing DataCollector class (unchanged)
class DataCollector:
    def __init__(self):
        self.data = []

    def collect_data(self, player_action, npc_decision, npc_response, game_state):
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
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                self.data = list(reader)
            print(f"Data loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found. No data loaded.")

    # New method for Lesson 4
    def prepare_training_data(self):
        X = []
        y = []
        for entry in self.data:
            X.append([
                entry['player_friendly'] == 'True',
                entry['player_has_item'] == 'True',
                entry['time_of_day'],
                entry['location']
            ])
            y.append(entry['npc_decision'])
        return np.array(X), np.array(y)

# Modified Game class
class Game:
    def __init__(self):
        self.config = GameConfig()
        self.npc = NPC("Guardian")
        self.data_collector = DataCollector()
        self.turn_count = 0

    def start(self):
        print("Welcome to 'Decisions n Dialogue'!")
        print(f"You find yourself in the {self.config.location}. It's currently {self.config.time_of_day}.")
        print(f"You encounter {self.npc.name}.")

        while True:
            action = input("What would you like to do? (talk/leave): ").lower()
            if action == 'talk':
                npc_decision, npc_response = self.npc.interact(action, self.config.player_has_item, self.config.time_of_day, self.config.location)
                print(npc_response)
                self.data_collector.collect_data(action, npc_decision, npc_response, self.config)
                
                # Update NPC's decision tree every 10 turns
                self.turn_count += 1
                if self.turn_count % 10 == 0:
                    X, y = self.data_collector.prepare_training_data()
                    self.npc.update_decision_tree(X, y)
                
            elif action == 'leave':
                print("You decide to leave. Game over.")
                break
            else:
                print("Invalid action. Please choose 'talk' or 'leave'.")

        self.data_collector.save_data()

if __name__ == "__main__":
    game = Game()
    game.start()