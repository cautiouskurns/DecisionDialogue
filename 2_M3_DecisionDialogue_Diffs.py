import random
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from ipywidgets import widgets, Layout, VBox, HBox
from IPython.display import display, clear_output
import json

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
    
class NPCResponseTemplates:
    def __init__(self):
        self.load_response_templates()

    def load_response_templates(self):
        with open('responses_templates.json', 'r') as f:
            self.response_templates = json.load(f)

    def get_response(self, response_type, player_action, npc_name):
        templates = self.response_templates.get(response_type, [])
        if not templates:
            return f"{npc_name} doesn't know how to respond."
        
        chosen_template = random.choice(templates)
        response = chosen_template.format(
            player_action=player_action,
            npc_name=npc_name
        )
        return response

class NPC:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.friendly = random.choice([True, False])
        self.training_data = NPCTrainingData()
        self.decision_tree = NPCDecisionTree(self.training_data)
        self.response_templates = NPCResponseTemplates()
        self.interaction_history = []

    def interact(self, player_friendly, player_has_item, time_of_day, location, player_action):
        action = self.decision_tree.decide_action(player_friendly, player_has_item, time_of_day, location)
        response = self.response_templates.get_response(action, player_action, self.name)
        
        self.interaction_history.append({
            'player_action': player_action,
            'npc_action': action,
            'context': [player_friendly, player_has_item, time_of_day, location]
        })
        
        return response

    def get_response_type(self, npc_action, player_action):
        if npc_action == 'talk':
            return "greet" if player_action == "Approach Friendly" else "talk"
        return npc_action

class GameInterface:
    def __init__(self, game):
        self.game = game
        self.setup_interface()

    def setup_interface(self):
        self.output = widgets.HTML()
        self.action_buttons = [
            widgets.Button(description="Talk"),
            widgets.Button(description="Leave")
        ]
        for button in self.action_buttons:
            button.on_click(self.on_button_clicked)
        
        self.layout = VBox([
            widgets.HTML("<h1>Decisions n Dialogue</h1>"),
            self.output,
            HBox(self.action_buttons)
        ])
        display(self.layout)

    def on_button_clicked(self, button):
        if button.description.lower() == "talk":
            response = self.game.logic.interact("talk")
            self.update_display(response)
        elif button.description.lower() == "leave":
            self.update_display("You decide to leave. Game over.")
            self.game.running = False

    def update_display(self, message):
        self.output.value += f"<p>{message}</p>"

class GameLogic:
    def __init__(self, game):
        self.game = game

    def interact(self, action):
        if action == "talk":
            return self.game.npc.interact(
                self.game.config.player_friendly,
                self.game.config.player_has_item,
                self.game.config.time_of_day,
                self.game.config.location,
                action
            )
        return "Invalid action"

class Game:
    def __init__(self):
        self.config = GameConfig()
        self.npc = NPC("Guardian")
        self.logic = GameLogic(self)
        self.interface = GameInterface(self)
        self.running = True

    def start(self):
        self.interface.update_display("Welcome to 'Decisions n Dialogue'!")
        self.interface.update_display(f"You find yourself in the {self.config.location}. It's currently {self.config.time_of_day}.")
        self.interface.update_display(f"You encounter {self.npc.name}.")

    def run(self):
        self.start()
        while self.running:
            pass  # The game now runs based on button clicks in the interface

if __name__ == "__main__":
    game = Game()
    game.start()