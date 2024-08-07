import random
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from ipywidgets import widgets, Layout, VBox, HBox
from IPython.display import display, clear_output
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

class NPCDecisionTree:
    def __init__(self, training_data):
        self.training_data = training_data
        self.clf = self.train_decision_tree()
        self.accuracy_history = []
        self.tree_depth_history = []

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

    def update_metrics(self):
        y_pred = self.clf.predict(self.training_data.X)
        accuracy = np.mean(y_pred == self.training_data.y)
        self.accuracy_history.append(accuracy)
        self.tree_depth_history.append(self.clf.get_depth())

class NPCVisualizer:
    def __init__(self, decision_tree):
        self.decision_tree = decision_tree

    def visualize_decision_tree(self):
        tree = self.decision_tree.clf.tree_
        feature_names = ['Player Friendly', 'Player Has Item', 'Morning/Night', 'Afternoon/Evening', 'Forest/Village']
        class_names = self.decision_tree.clf.classes_

        def tree_to_graph(node, x, y, dx, dy):
            if tree.feature[node] != -2:  # not a leaf node
                threshold = tree.threshold[node]
                feature = feature_names[tree.feature[node]]
                left_child = tree.children_left[node]
                right_child = tree.children_right[node]

                # Node
                nodes.append(go.Scatter(x=[x], y=[y], mode='markers+text', 
                                        marker=dict(size=30, color='lightblue'),
                                        text=[f"{feature}<br>{threshold:.2f}"], textposition='middle center',
                                        hoverinfo='text', name=''))

                # Edges
                edges.append(go.Scatter(x=[x, x-dx, x, x+dx], y=[y, y-dy, y, y-dy], mode='lines',
                                        line=dict(color='gray'), hoverinfo='none', name=''))

                tree_to_graph(left_child, x-dx, y-dy, dx/2, dy)
                tree_to_graph(right_child, x+dx, y-dy, dx/2, dy)
            else:  # leaf node
                value = tree.value[node]
                class_idx = np.argmax(value)
                class_name = class_names[class_idx]
                nodes.append(go.Scatter(x=[x], y=[y], mode='markers+text',
                                        marker=dict(size=25, color='lightgreen'),
                                        text=[class_name], textposition='middle center',
                                        hoverinfo='text', name=''))

        nodes, edges = [], []
        tree_to_graph(0, 0, 1, 0.5, 0.1)

        layout = go.Layout(
            title="NPC's Decision Tree",
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        fig = go.Figure(data=edges + nodes, layout=layout)
        return fig

    def plot_performance_metrics(self):
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Model Accuracy Over Time", "Decision Tree Depth Over Time"))

        fig.add_trace(go.Scatter(y=self.decision_tree.accuracy_history, mode='lines+markers', name='Accuracy'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(y=self.decision_tree.tree_depth_history, mode='lines+markers', name='Tree Depth'),
                      row=2, col=1)

        fig.update_layout(height=600, width=800, title_text="NPC Performance Metrics")
        fig.update_xaxes(title_text="Update Iterations", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Tree Depth", row=2, col=1)

        return fig

class NPC:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.friendly = random.choice([True, False])
        self.training_data = NPCTrainingData()
        self.decision_tree = NPCDecisionTree(self.training_data)
        self.response_templates = NPCResponseTemplates()
        self.interaction_history = []
        self.visualizer = NPCVisualizer(self.decision_tree)

    def interact(self, player_friendly, player_has_item, time_of_day, location, player_action):
        action = self.decision_tree.decide_action(player_friendly, player_has_item, time_of_day, location)
        response = self.response_templates.get_response(action, player_action, self.name)
        
        self.interaction_history.append({
            'player_action': player_action,
            'npc_action': action,
            'context': [player_friendly, player_has_item, time_of_day, location]
        })
        
        self.decision_tree.update_metrics()
        
        return response

    def get_response_type(self, npc_action, player_action):
        if npc_action == 'talk':
            return "greet" if player_action == "Approach Friendly" else "talk"
        return npc_action

    def visualize_decision_tree(self):
        return self.visualizer.visualize_decision_tree()

    def plot_performance_metrics(self):
        return self.visualizer.plot_performance_metrics()

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
        self.viz_buttons = [
            widgets.Button(description="Show Decision Tree"),
            widgets.Button(description="Show Performance Metrics")
        ]
        for button in self.action_buttons + self.viz_buttons:
            button.on_click(self.on_button_clicked)
        
        self.layout = VBox([
            widgets.HTML("<h1>Decisions n Dialogue</h1>"),
            self.output,
            HBox(self.action_buttons),
            HBox(self.viz_buttons)
        ])
        display(self.layout)

    def on_button_clicked(self, button):
        if button.description == "Talk":
            response = self.game.logic.interact("talk")
            self.update_display(response)
        elif button.description == "Leave":
            self.update_display("You decide to leave. Game over.")
            self.game.running = False
        elif button.description == "Show Decision Tree":
            fig = self.game.npc.visualize_decision_tree()
            fig.show()
        elif button.description == "Show Performance Metrics":
            fig = self.game.npc.plot_performance_metrics()
            fig.show()

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