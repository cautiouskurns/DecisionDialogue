import random

class GameConfig:
    def __init__(self):
        self.player_health = 100
        self.player_friendly = True
        self.player_has_item = False
        self.time_of_day = random.choice(["morning", "afternoon", "evening", "night"])
        self.location = random.choice(["forest", "village", "castle", "cave"])

class NPC:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.friendly = random.choice([True, False])

    def interact(self):
        if self.friendly:
            return f"{self.name} greets you warmly."
        else:
            return f"{self.name} looks at you suspiciously."

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
                print(self.npc.interact())
            elif action == 'leave':
                print("You decide to leave. Game over.")
                break
            else:
                print("Invalid action. Please choose 'talk' or 'leave'.")

if __name__ == "__main__":
    game = Game()
    game.start()