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

# Existing DecisionTree class (unchanged)
class DecisionTree:
    def __init__(self):
        self.clf = DecisionTreeClassifier(random_state=42)
        self.label_encoder = LabelEncoder()
        self.feature_names = ['npc_friendly', 'npc_has_item', 'player_has_item', 'time_of_day', 'location']
        self.trained = False

    def train(self, X, y):
        X_encoded = np.array([self.label_encoder.fit_transform(x) for x in X.T]).T
        y_encoded = self.label_encoder.fit_transform(y)
        self.clf.fit(X_encoded, y_encoded)
        self.trained = True

    def make_decision(self, npc_friendly, npc_has_item, player_has_item, time_of_day, location):
        if not self.trained:
            if npc_friendly:
                return 'talk' if npc_has_item else 'give_item'
            else:
                return 'trade' if player_has_item else 'ignore'
        
        X = np.array([[npc_friendly, npc_has_item, player_has_item, time_of_day, location]])
        X_encoded = np.array([self.label_encoder.transform(x) for x in X.T]).T
        decision_encoded = self.clf.predict(X_encoded)[0]
        return self.label_encoder.inverse_transform([decision_encoded])[0]

# New class for Lesson 5: NPC Response Templates
class NPCResponseTemplates:
    def __init__(self):
        self.load_templates()

    def load_templates(self):
        # Load response templates from a JSON file
        with open('npc_response_templates.json', 'r') as f:
            self.templates = json.load(f)

    def get_response(self, action, context):
        """
        Get a context-aware response based on the NPC's action and game context.
        
        :param action: The NPC's decided action
        :param context: A dictionary containing relevant game state information
        :return: A string containing the NPC's response
        """
        if action not in self.templates:
            return f"The NPC performs the action: {action}"

        possible_responses = self.templates[action]
        
        # Filter responses based on context
        filtered_responses = [
            response for response in possible_responses
            if all(context.get(key) == value for key, value in response.get('conditions', {}).items())
        ]

        if filtered_responses:
            chosen_response = random.choice(filtered_responses)
            return chosen_response['text'].format(**context)
        else:
            return random.choice(possible_responses)['text'].format(**context)

# Modified NPC class
class NPC:
    def __init__(self, name):
        self.name = name
        self.load_npc_config()
        self.decision_tree = DecisionTree()
        self.response_templates = NPCResponseTemplates()  # New for Lesson 5

    def load_npc_config(self):
        with open('npc_config_test.json', 'r') as f:
            config = json.load(f)
        self.health = config['initial_health']
        self.friendly = random.choice(config['friendly_options'])
        self.has_item = random.choice(config['has_item_options'])
        self.mood = random.choice(config['mood_options'])

    def interact(self, player_action, player_has_item, time_of_day, location):
        decision = self.decision_tree.make_decision(self.friendly, self.has_item, player_has_item, time_of_day, location)
        
        # Create context for response generation (New for Lesson 5)
        context = {
            'npc_name': self.name,
            'npc_mood': self.mood,
            'player_has_item': player_has_item,
            'time_of_day': time_of_day,
            'location': location
        }
        
        response = self.response_templates.get_response(decision, context)
        
        return decision, response

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
                
                # Display more detailed NPC response (New for Lesson 5)
                print(f"\n{self.npc.name}'s Response:")
                print(f"Action: {npc_decision}")
                print(f"Response: {npc_response}\n")
                
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