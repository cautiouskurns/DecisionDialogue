import random
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from ipywidgets import widgets, Layout, VBox, HBox
from IPython.display import display, clear_output

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
                self.game.config.location
            )
        return "Invalid action"

class Game:
    def __init__(self):
        self.config = GameConfig()
        self.npc = NPC("Guardian")
        self.logic = GameLogic(self)
        self.running = True
        self.interface = GameInterface(self)

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