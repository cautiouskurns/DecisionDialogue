import random
import json
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from ipywidgets import widgets, Layout, HTML, VBox, HBox, GridspecLayout
from IPython.display import display, clear_output
import base64

# Define a color scheme
COLOR_SCHEME = {
    'background': '#2b2b2b',
    'text': '#FFFFFF',
    'primary': '#3498DB',
    'secondary': '#E74C3C',
    'accent': '#2ECC71',
    'hover': '#9b9b9b',
    'primary_button_color': '#f6f8d3',
    'secondary_button_color': '#fccb62',
}

class NPCConfig:
    def __init__(self):
        self.load_npc_config()

    def load_npc_config(self):
        with open('npc_config.json', 'r') as f:
            config = json.load(f)
        self.health = config['initial_health']
        self.friendly = random.choice(config['friendly_options'])
        self.has_item = random.choice(config['has_item_options'])
        self.mood = random.choice(config['mood_options'])

class GameConfig:
    def __init__(self):
        self.load_game_config()
        self.time_options = ["morning", "afternoon", "evening", "night"]
        self.location_options = ["forest", "village", "castle", "dungeon"]
        self.environment_change_turns = 10
        self.turn_count = 0
        self.time_of_day = "morning"
        self.location = "village"
        
    def load_game_config(self):
        with open('game_config.json', 'r') as f:
            config = json.load(f)
        self.player_health = config['initial_player_health']
        self.player_friendly = config['initial_player_friendly']
        self.player_has_item = config['initial_player_has_item']
        self.time_of_day = random.choice(config['time_options'])
        self.location = random.choice(config['location_options'])
        self.turn_count = 0
        self.game_log = []
        self.COLOR_SCHEME = config['color_scheme']
        self.attack_keywords = config['attack_keywords']
        self.give_keywords = config['give_keywords']
        self.npc_attack_damage = config['npc_attack_damage']
        self.npc_evolution_turns = config['npc_evolution_turns']
        self.environment_change_turns = config['environment_change_turns']

class NPCTrainingData:
    def __init__(self):
        self.load_initial_training_data()

    def load_initial_training_data(self):
        with open('npc_training_data.json', 'r') as f:
            training_data = json.load(f)
        self.X = np.array(training_data['features'])
        self.y = np.array(training_data['labels'])
        
        # Convert boolean values to integers
        self.X = np.array([[int(val) if isinstance(val, bool) else val for val in row] for row in self.X])

class NPCResponseTemplates:
    def __init__(self):
        self.load_response_templates()

    def load_response_templates(self):
        with open('npc_responses.json', 'r') as f:
            self.response_templates = json.load(f)

    def get_response(self, response_type, player_action, npc_name):
        templates = self.response_templates[response_type]
        chosen_template = random.choice(templates)
        
        response = chosen_template.format(
            player_action=player_action,
            npc_name=npc_name,
        )
        
        return response

class NPCDecisionTree:
    def __init__(self, training_data):
        self.training_data = training_data
        self.clf = self.train_decision_tree()
        self.interaction_history = []
        self.accuracy_history = []
        self.tree_depth_history = []

    def train_decision_tree(self):
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(self.training_data.X, self.training_data.y)
        return clf

    def decide_action(self, player_friendly, player_has_item, time_of_day, location, health, mood):
        features = [
            int(player_friendly),
            int(player_has_item),
            health,
            ['happy', 'neutral', 'angry'].index(mood),
            ['morning', 'afternoon', 'evening', 'night'].index(time_of_day),
            ['forest', 'village', 'castle', 'cave'].index(location)
        ]
        action = self.clf.predict([features])[0]
        return action

    def update_decision_tree(self):
        new_X = []
        new_y = []
        for interaction in self.interaction_history[-10:]:  # Consider last 10 interactions
            context = interaction['context']
            new_X.append([
                int(context[0]),  # player_friendly
                int(context[1]),  # player_has_item
                int(context[2]),  # health
                ['happy', 'neutral', 'angry'].index(context[3]),  # mood
                ['morning', 'afternoon', 'evening', 'night'].index(context[4]),  # time_of_day
                ['forest', 'village', 'castle', 'cave'].index(context[5])  # location
            ])
            new_y.append(interaction['npc_action'])

        # Add new data to existing training data
        self.training_data.X = np.vstack([self.training_data.X, new_X])
        self.training_data.y = np.hstack([self.training_data.y, new_y])

        # Retrain the classifier
        self.clf = self.train_decision_tree()

        # Calculate and store performance metrics
        y_pred = self.clf.predict(self.training_data.X)
        accuracy = np.mean(y_pred == self.training_data.y)
        self.accuracy_history.append(accuracy)
        self.tree_depth_history.append(self.clf.get_depth())

    def get_action_distribution(self):
        # Calculate the distribution of NPC actions
        action_counts = np.bincount(self.training_data.y, minlength=6)
        action_names = ['Attack', 'Talk', 'Flee', 'Give Item', 'Trade', 'Ignore']
        return dict(zip(action_names, action_counts / len(self.training_data.y)))

class NPCVisualizer:
    def __init__(self, decision_tree):
        self.decision_tree = decision_tree

    def visualize_decision_tree(self, width=800, height=600):
        tree = self.decision_tree.clf.tree_
        feature_names = ['Player Friendly', 'Player Has Item', 'NPC Health', 'NPC Mood', 'Time of Day', 'Location']
        class_names = ['Attack', 'Talk', 'Flee', 'Give Item', 'Trade', 'Ignore']

        def tree_to_plotly(node, x, y, dx, dy):
            if tree.feature[node] != -2:  # not a leaf node
                threshold = tree.threshold[node]
                feature = feature_names[tree.feature[node]]
                left_child = tree.children_left[node]
                right_child = tree.children_right[node]

                # Node
                nodes.append(go.Scatter(x=[x], y=[y], mode='markers+text', 
                                        marker=dict(size=30, color=COLOR_SCHEME['primary']),
                                        text=[f"{feature}<br>{threshold:.2f}"], textposition='middle center',
                                        hoverinfo='text', name=''))

                # Edges
                edges.append(go.Scatter(x=[x, x-dx, x, x+dx], y=[y, y-dy, y, y-dy], mode='lines',
                                        line=dict(color=COLOR_SCHEME['secondary']), hoverinfo='none', name=''))

                tree_to_plotly(left_child, x-dx, y-dy, dx/2, dy)
                tree_to_plotly(right_child, x+dx, y-dy, dx/2, dy)
            else:  # leaf node
                value = tree.value[node]
                class_idx = np.argmax(value)
                class_name = class_names[class_idx]
                nodes.append(go.Scatter(x=[x], y=[y], mode='markers+text',
                                        marker=dict(size=25, color=COLOR_SCHEME['accent']),
                                        text=[class_name], textposition='middle center',
                                        hoverinfo='text', name=''))

        nodes, edges = [], []
        tree_to_plotly(0, 0, 1, 0.5, 0.1)

        layout = go.Layout(
            title=dict(text=f"NPC's Decision Tree", font=dict(size=24, color=COLOR_SCHEME['text'])),
            hovermode='closest', showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor=COLOR_SCHEME['background'],
            plot_bgcolor=COLOR_SCHEME['background'],
            font=dict(family="Arial, sans-serif", size=14, color=COLOR_SCHEME['text']),
            width=width,
            height=height
        )

        fig = go.Figure(data=edges + nodes, layout=layout)
        return fig

    def plot_performance_metrics(self):
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Model Accuracy Over Time", "Decision Tree Depth Over Time"))

        fig.add_trace(go.Scatter(y=self.decision_tree.accuracy_history, mode='lines+markers', name='Accuracy',
                                 line=dict(color=COLOR_SCHEME['primary'])), row=1, col=1)
        fig.add_trace(go.Scatter(y=self.decision_tree.tree_depth_history, mode='lines+markers', name='Tree Depth',
                                 line=dict(color=COLOR_SCHEME['secondary'])), row=2, col=1)

        fig.update_layout(
            height=600, width=800,
            title=dict(text="NPC Performance Metrics", font=dict(size=24, color=COLOR_SCHEME['text'])),
            paper_bgcolor=COLOR_SCHEME['background'],
            plot_bgcolor=COLOR_SCHEME['background'],
            font=dict(family="Arial, sans-serif", size=14, color=COLOR_SCHEME['text'])
        )
        fig.update_xaxes(title_text="Update Iterations", row=2, col=1, gridcolor='lightgrey')
        fig.update_yaxes(title_text="Accuracy", row=1, col=1, gridcolor='lightgrey')
        fig.update_yaxes(title_text="Tree Depth", row=2, col=1, gridcolor='lightgrey')

        return fig

    def plot_feature_importance(self):
        feature_importance = self.decision_tree.clf.feature_importances_
        feature_names = ['Player Friendly', 'Player Has Item', 'NPC Health', 'NPC Mood', 'Time of Day', 'Location']
        
        fig = px.bar(x=feature_importance, y=feature_names, orientation='h',
                     labels={'x': 'Importance', 'y': 'Feature'},
                     color=feature_importance, color_continuous_scale=px.colors.sequential.Viridis)
        
        fig.update_layout(
            height=600, width=800,
            title=dict(text="Feature Importance in NPC Decision Making", font=dict(size=24, color=COLOR_SCHEME['text'])),
            paper_bgcolor=COLOR_SCHEME['background'],
            plot_bgcolor=COLOR_SCHEME['background'],
            font=dict(family="Arial, sans-serif", size=14, color=COLOR_SCHEME['text']),
            yaxis={'categoryorder': 'total ascending'}
        )
        fig.update_xaxes(gridcolor='lightgrey')
        fig.update_yaxes(gridcolor='lightgrey')

        return fig

class NPC:
    def __init__(self, name):
        self.name = name
        self.config = NPCConfig()
        self.training_data = NPCTrainingData()
        self.response_templates = NPCResponseTemplates()
        self.decision_tree = NPCDecisionTree(self.training_data)
        self.visualizer = NPCVisualizer(self.decision_tree)
        self.health = self.config.health
        self.mood = self.config.mood
        self.has_item = self.config.has_item

    def interact(self, player_action, player_friendly, player_has_item, time_of_day, location):
        action = self.decision_tree.decide_action(player_friendly, player_has_item, time_of_day, location, self.health, self.mood)
        self.decision_tree.interaction_history.append({
            'player_action': player_action,
            'npc_action': action,
            'context': [player_friendly, player_has_item, self.health, self.mood, time_of_day, location]
        })
        
        response_type = self.get_response_type(action, player_action)
        response = self.response_templates.get_response(response_type, player_action, self.name)
        
        return response

    def get_response_type(self, npc_action, player_action):
        if npc_action == 0:  # Attack
            return "attack"
        elif npc_action == 1:  # Friendly greeting
            return "greet" if player_action in ["Approach Friendly", "Approach Cautiously"] else "talk"
        elif npc_action == 2:  # Retreat
            return "retreat"
        elif npc_action == 3:  # Offer item
            return "offer_item"
        elif npc_action == 4:  # Propose trade
            return "propose_trade"
        else:  # Ignore
            return "ignore"

    def update_decision_tree(self):
        self.decision_tree.update_decision_tree()

    def get_action_distribution(self):
        return self.decision_tree.get_action_distribution()

    def visualize_decision_tree(self, width=800, height=600):
        return self.visualizer.visualize_decision_tree(width, height)

    def plot_performance_metrics(self):
        return self.visualizer.plot_performance_metrics()

    def plot_feature_importance(self):
        return self.visualizer.plot_feature_importance()

class GameInterface:
    def __init__(self, game):
        self.game = game
        self.player_animating = False
        self.npc_animating = False
        self.player_health = 100
        self.game_log = []
        self.setup_interface()

    def setup_interface(self):
        custom_css = f"""
        <style>
        .widget-box {{
            background-color: {COLOR_SCHEME['background']} !important;
        }}
        .widget-label {{
            color: {COLOR_SCHEME['text']} !important;
        }}
        .jp-OutputArea-output {{
            color: {COLOR_SCHEME['text']} !important;
        }}
        .custom-button {{
            width: 100% !important;
            height: 40px !important;
            margin: 5px 0 !important;
            transition: background-color 0.3s;
        }}
        .custom-button:hover {{
            background-color: {COLOR_SCHEME['hover']} !important;
        }}
        .character-image {{
            width: 150px;
            height: 150px;
            object-fit: contain;
            position: absolute;
            bottom: 10px;
            z-index: 2;
            transition: transform 0.3s ease-in-out;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            background: transparent !important;
        }}
        .background-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            position: absolute;
            top: 0;
            left: 0;
            z-index: 1;
        }}
        .image-container {{
            position: relative;
            width: 100%;
            height: 200px;
            overflow: hidden;
            background-color: transparent !important;
        }}
        .status-box {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }}
        @keyframes player-move {{
            0% {{ transform: translateX(0); }}
            50% {{ transform: translateX(10px); }}
            100% {{ transform: translateX(0); }}
        }}
        @keyframes npc-move {{
            0% {{ transform: translateX(0); }}
            50% {{ transform: translateX(-10px); }}
            100% {{ transform: translateX(0); }}
        }}
        .player-animate {{
            animation: player-move 0.3s ease-in-out;
        }}
        .npc-animate {{
            animation: npc-move 0.3s ease-in-out;
        }}
        .game-log {{
            height: 300px;
            overflow-y: auto;
            border: 2px solid {COLOR_SCHEME['accent']};
            border-radius: 10px;
            padding: 15px;
            background-color: rgba(0, 0, 0, 0.8);
            font-family: 'Courier New', monospace;
            font-size: 16px;
            line-height: 1.5;
        }}
        .player-action {{
            color: #4CAF50;
            font-weight: bold;
            margin-bottom: 10px;
            padding: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .npc-action {{
            color: #FF9800;
            font-style: italic;
            margin-bottom: 10px;
            padding: 5px;
            border-left: 4px solid #FF9800;
        }}
        .system-message {{
            color: #2196F3;
            margin-bottom: 10px;
            padding: 5px;
            border-left: 4px solid #2196F3;
        }}
        .timestamp {{
            font-size: 12px;
            color: #999;
            margin-right: 10px;
        }}
        </style>
        """
        display(HTML(custom_css))

        self.show_evolution_button = widgets.Button(
            description="Show NPC Evolution",
            layout=Layout(width='100%', height='40px', margin='5px 0'),
            style=dict(button_color=COLOR_SCHEME['primary_button_color'], font_weight='bold')
        )
        self.show_evolution_button.add_class('custom-button')
        self.show_evolution_button.on_click(self.on_show_evolution)

        with open('game_config.json', 'r') as f:
            config = json.load(f)
        self.action_buttons = []
        displayed_actions = random.sample(config['action_options'], 5)
        for action in displayed_actions:
            button = widgets.Button(
                description=f"{action['text']} ({action['intent']})",
                layout=Layout(width='100%', height='40px', margin='5px 0'),
                style=dict(button_color=COLOR_SCHEME['primary_button_color'], font_weight='bold')
            )
            button.add_class('custom-button')
            button.on_click(self.on_action)
            self.action_buttons.append(button)

        self.quit_button = widgets.Button(
            description='Quit',
            style=dict(button_color=COLOR_SCHEME['secondary_button_color'], font_weight='bold'),
            layout=Layout(width='100%', height='40px', margin='5px 0')
        )
        self.quit_button.add_class('custom-button')
        self.quit_button.on_click(self.on_quit)

        viz_button_layout = Layout(width='32%', height='40px', margin='5px')
        self.show_tree_button = widgets.Button(description="Show Decision Tree", layout=viz_button_layout, style=dict(button_color=COLOR_SCHEME['primary_button_color'], font_weight='bold'))
        self.show_metrics_button = widgets.Button(description="Show Performance Metrics", layout=viz_button_layout, style=dict(button_color=COLOR_SCHEME['primary_button_color'], font_weight='bold'))
        self.show_importance_button = widgets.Button(description="Show Feature Importance", layout=viz_button_layout, style=dict(button_color=COLOR_SCHEME['primary_button_color'], font_weight='bold'))

        for button in [self.show_tree_button, self.show_metrics_button, self.show_importance_button]:
            button.add_class('custom-button')

        self.show_tree_button.on_click(self.on_show_tree)
        self.show_metrics_button.on_click(self.on_show_metrics)
        self.show_importance_button.on_click(self.on_show_importance)

        self.log_output = widgets.HTML(
            value='<div id="game-log" class="game-log"></div>',
            layout=Layout(width='100%', height='320px')
        )

        self.status_output = widgets.Output(layout=Layout(width='100%'))

        self.viz_output = widgets.Output(layout=Layout(width='100%', height='500px', border=f'1px solid {COLOR_SCHEME["text"]}'))

        player_image_path = r'C:\Users\diarm\Downloads\Player_2.png'
        npc_image_path = r'C:\Users\diarm\Downloads\Wizard_2.png'
        bg_image_path = r'C:\Users\diarm\Downloads\BG_2.png'

        with open(player_image_path, 'rb') as image_file:
            self.encoded_player_image = base64.b64encode(image_file.read()).decode('utf-8')

        with open(npc_image_path, 'rb') as image_file:
            self.encoded_npc_image = base64.b64encode(image_file.read()).decode('utf-8')

        with open(bg_image_path, 'rb') as image_file:
            encoded_bg_image = base64.b64encode(image_file.read()).decode('utf-8')

        self.bg_image = f'<img src="data:image/png;base64,{encoded_bg_image}" alt="Background" class="background-image" />'

        self.image_box = widgets.HTML(self.create_image_box())

        title = widgets.HTML(value=f"<h1 style='color: {COLOR_SCHEME['text']}; text-align: center; text-shadow: 2px 2px 4px #000000;'>Decisions n Dialogue</h1>")
        
        status_box = widgets.Box([self.status_output], layout=Layout(width='100%'), 
                                 style={'background-color': 'rgba(0, 0, 0, 0.7)', 'padding': '10px', 'border-radius': '5px'})
        
        action_box = VBox(self.action_buttons, layout=Layout(width='100%', align_items='stretch'))
        
        viz_controls = HBox([self.show_tree_button, self.show_metrics_button, self.show_importance_button],
                            layout=Layout(width='100%', justify_content='space-between'))
        
        game_log_title = widgets.HTML(value=f"<h3 style='color: {COLOR_SCHEME['text']};'>Dialogue</h3>")
        
        layout = VBox([
            title,
            self.image_box,
            status_box,
            action_box,
            self.quit_button,
            game_log_title,
            self.log_output,
            HBox([self.show_tree_button, self.show_metrics_button, self.show_importance_button, self.show_evolution_button])
        ], layout=Layout(width='800px', padding='20px'))
        
        display(layout)

        self.update_status()
        self.log("Welcome to 'Decisions n Dialogue'! You encounter the Guardian in the forest.")

    def create_image_box(self, player_animate=False, npc_animate=False):
        player_class = 'character-image player-animate' if player_animate else 'character-image'
        npc_class = 'character-image npc-animate' if npc_animate else 'character-image'
        
        return f"""
        <div class="image-container">
            {self.bg_image}
            <div class="character-box">
                <img src="data:image/png;base64,{self.encoded_player_image}" alt="Player" class="{player_class}" style="left: 10px;" />
                <div class="health-bar" id="player-health">Player Health: {self.player_health}</div>
            </div>
            <div class="character-box">
                <img src="data:image/png;base64,{self.encoded_npc_image}" alt="NPC" class="{npc_class}" style="right: 10px;" />
                <div class="health-bar" id="npc-health">NPC Health: {self.game.npc.health}</div>
            </div>
        </div>
        """

    def on_action(self, action):
        if isinstance(action, str):
            action_text = action
        else:
            action_text = action.description.split(' (')[0]
        self.animate_character("player")
        
        with open('player_actions.json', 'r') as f:
            actions = json.load(f)
        
        if action_text in actions:
            self.log(actions[action_text]['message'], 'player')
            for effect in actions[action_text]['effects']:
                setattr(self.game.config, effect['attribute'], effect['value'])
        else:
            self.log(f"Unknown action: {action_text}", 'system')

        self.game.logic.interact(action_text)

    def on_quit(self, b):
        self.log("Thanks for playing!")
        self.game.running = False

    def log(self, message, message_type='system'):
        if message_type == 'player':
            formatted_message = f'<p class="player-action"><strong style="color: #4CAF50;">You:</strong> <span style="color: white;">{message}</span></p>'
        elif message_type == 'npc':
            formatted_message = f'<p class="npc-action"><strong style="color: #FF9800;">{self.game.npc.name}:</strong> <span style="color: white;">{message}</span></p>'
        else:
            formatted_message = f'<p class="system-message"><strong style="color: #2196F3;">Narrator:</strong> <span style="color: white;">{message}</span></p>'

        current_log = self.log_output.value
        updated_log = current_log.replace('</div>', f'{formatted_message}</div>')
        self.log_output.value = updated_log

        self.game_log.append(message)

    def update_status(self):
        status_html = f"""
        <div style="display: flex; justify-content: space-around; align-items: center; background-color: rgba(0, 0, 0, 0.7); padding: 10px; border-radius: 5px; color: {COLOR_SCHEME['text']};">
            <div style="width: 200px; margin-right: 20px;">
                <div style="font-size: 14px;">Player Health</div>
                <div style="background-color: #ddd; border-radius: 10px; overflow: hidden;">
                    <div style="width: {self.game.config.player_health}%; height: 20px; background-color: #4CAF50; border-radius: 10px;"></div>
                </div>
            </div>
            <div style="margin-right: 20px;">Time of Day: {self.game.config.time_of_day}</div>
            <div style="margin-right: 20px;">Location: {self.game.config.location}</div>
            <div style="margin-right: 20px;">Player has item: {'Yes' if self.game.config.player_has_item else 'No'}</div>
            <div style="width: 200px;">
                <div style="font-size: 14px;">NPC Health</div>
                <div style="background-color: #ddd; border-radius: 10px; overflow: hidden;">
                    <div style="width: {self.game.npc.health}%; height: 20px; background-color: #FF9800; border-radius: 10px;"></div>
                </div>
            </div>
        </div>
        """
        self.status_output.clear_output(wait=True)
        with self.status_output:
            display(HTML(status_html))

    def on_show_evolution(self, b):
        self.game.visualization.visualize_npc_evolution()

    def on_show_tree(self, b):
        with self.viz_output:
            clear_output(wait=True)
            fig = self.game.npc.visualize_decision_tree()
            fig.show()

    def on_show_metrics(self, b):
        with self.viz_output:
            clear_output(wait=True)
            fig = self.game.npc.plot_performance_metrics()
            fig.show()

    def on_show_importance(self, b):
        with self.viz_output:
            clear_output(wait=True)
            fig = self.game.npc.plot_feature_importance()
            fig.show()

    def animate_character(self, character):
        if character == "player" and not self.player_animating:
            self.player_animating = True
            self.image_box.value = self.create_image_box(player_animate=True)
            
            def reset_player_animation():
                self.player_animating = False
                self.image_box.value = self.create_image_box()
            
            import threading
            threading.Timer(0.3, reset_player_animation).start()
        
        elif character == "npc" and not self.npc_animating:
            self.npc_animating = True
            self.image_box.value = self.create_image_box(npc_animate=True)
            
            def reset_npc_animation():
                self.npc_animating = False
                self.image_box.value = self.create_image_box()
                import threading
                threading.Timer(0.3, reset_npc_animation).start()

class GameLogic:
    def __init__(self, game):
        self.game = game

    def interact(self, player_action):
        npc_response = self.game.npc.interact(player_action, self.game.config.player_friendly, self.game.config.player_has_item,
                                               self.game.config.time_of_day, self.game.config.location)
        self.game.interface.log(npc_response, 'npc')
        self.game.interface.animate_character("npc")
        self.handle_npc_response(npc_response)
        self.update_game_state()

    def handle_npc_response(self, response):
        if any(keyword in response.lower() for keyword in self.game.config.attack_keywords):
            self.game.config.player_health -= self.game.config.npc_attack_damage
            self.game.interface.log(f"Your health decreased. Current health: {self.game.config.player_health}")
        elif any(keyword in response.lower() for keyword in self.game.config.give_keywords):
            self.game.config.player_has_item = True
            self.game.interface.log("The NPC gave you an item.")

    def update_game_state(self):
        self.game.config.turn_count += 1
        if self.game.config.turn_count % self.game.config.npc_evolution_turns == 0:
            self.game.npc.update_decision_tree()
            self.game.interface.log("The NPC's behavior has evolved!")
            self.game.visualization.show_npc_evolution()
        if self.game.config.turn_count % self.game.config.environment_change_turns == 0:
            self.game.config.time_of_day = random.choice(self.game.config.time_options)
            self.game.config.location = random.choice(self.game.config.location_options)
            self.game.interface.log(f"You've moved to the {self.game.config.location} and time has passed. It's now {self.game.config.time_of_day}.")
        if self.game.config.player_health <= 0:
            self.game.interface.log("Game Over! You have been defeated.")
            self.game.running = False
        if self.game.npc.health <= 0:
            self.game.interface.log(f"{self.game.npc.name} has been defeated!")
            self.game.running = False
        self.game.interface.update_status()

class GameVisualization:
    def __init__(self, game):
        self.game = game
        self.viz_output = widgets.Output(layout=Layout(width='100%', height='500px', border=f'1px solid {self.game.config.COLOR_SCHEME["text"]}'))

    def show_npc_evolution(self):
        action_dist = self.game.npc.get_action_distribution()
        evolution_message = "NPC Behavior Change:\n"
        for action, prob in action_dist.items():
            evolution_message += f"{action}: {prob:.2f}\n"
        self.game.interface.log(evolution_message, 'system')
        
        # Visualize the evolution
        self.visualize_npc_evolution()

    def visualize_npc_evolution(self, width=800, height=600):
        with self.viz_output:
            clear_output(wait=True)
            action_dist = self.game.npc.get_action_distribution()
            fig = go.Figure(data=[go.Bar(x=list(action_dist.keys()), y=list(action_dist.values()))])
            fig.update_layout(
                title="NPC Action Distribution",
                xaxis_title="Actions",
                yaxis_title="Probability",
                paper_bgcolor=self.game.config.COLOR_SCHEME['background'],
                plot_bgcolor=self.game.config.COLOR_SCHEME['background'],
                font=dict(color=self.game.config.COLOR_SCHEME['text']),
                width=width,
                height=height
            )
            fig.show()

class Game:
    def __init__(self):
        self.config = GameConfig()
        self.npc = NPC("Guardian")
        self.interface = GameInterface(self)
        self.logic = GameLogic(self)
        self.visualization = GameVisualization(self)
        self.running = True

    def start(self):
        self.interface.log("Welcome to 'Decisions n Dialogue'! You encounter the Guardian in the forest.")

    def run(self):
        self.start()
        while self.running:
            pass  # The game now runs based on button clicks in the interface

if __name__ == "__main__":
    game = Game()
    game.run()