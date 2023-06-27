import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import os
import time
# Function to clear the screen



# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# Compute the score difference between before_source and after_source
def get_score_difference(before_source, after_source):
    # Compute the score difference by summing the differences between corresponding elements of before_source and after_source
    score_diff = sum(sum(after_source[i][j] - before_source[i][j] for j in range(4)) for i in range(4))
    return score_diff

# Function to update game state by moving tiles left
def left(source):
    after_source = copy.deepcopy(source)
    for i in range(4):
        after_source[i] = move_tiles_left(after_source[i])
    return after_source

# Helper function to move tiles left in a single row
def move_tiles_left(row):
    new_row = [0, 0, 0, 0]
    merged = [False, False, False, False]
    index = 0
    for value in row:
        if value != 0:
            # Check if there is a merge with the previous tile
            if index > 0 and value == new_row[index - 1] and not merged[index - 1]:
                new_row[index - 1] *= 2
                merged[index - 1] = True
            else:
                new_row[index] = value
                index += 1
    return new_row

# Function to update game state by moving tiles right
def right(source):
    after_source = copy.deepcopy(source)
    for i in range(4):
        after_source[i] = move_tiles_right(after_source[i])
    return after_source

# Helper function to move tiles right in a single row
def move_tiles_right(row):
    reversed_row = list(reversed(row))
    new_row = list(reversed(move_tiles_left(reversed_row)))
    return list(reversed(new_row))

# Function to update game state by moving tiles up
def up(source):
    after_source = copy.deepcopy(source)
    for j in range(4):
        column = [after_source[i][j] for i in range(4)]
        updated_column = move_tiles_left(column)
        for i in range(4):
            after_source[i][j] = updated_column[i]
    return after_source

# Function to update game state by moving tiles down
def down(source):
    after_source = copy.deepcopy(source)
    for j in range(4):
        column = [after_source[i][j] for i in range(4)]
        updated_column = move_tiles_right(column)
        for i in range(4):
            after_source[i][j] = updated_column[i]
    return after_source

# Check if the game is over
def game_over(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                return False
            if i > 0 and board[i][j] == board[i-1][j]:
                return False
            if i < len(board)-1 and board[i][j] == board[i+1][j]:
                return False
            if j > 0 and board[i][j] == board[i][j-1]:
                return False
            if j < len(board[i])-1 and board[i][j] == board[i][j+1]:
                return False
    return True

# Game visualization function
def display_game(game_state):
    with open("train.log",'a') as f:
        num=1
        for row in game_state:
            if num == 1 :
                f.write('[game state] \n'+str(row) + '\n')
            else :
                f.write(str(row) + '\n')
            num+=1

# Training result display function
def display_training_result(score):
    print("Training finished.")
    print(f"Max Score: {score}")

# Train function integrating game and model
def train(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    max_score = 0
    num_iterations = 0

    while True:
        score = 0
        game_state = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]

        for _ in range(2):
            i = random.randint(0, 3)
            j = random.randint(0, 3)
            game_state[i][j] = random.choice([0, 1])

        while True:
            display_game(game_state)

            action = random.choice(["left", "right", "up", "down"])
            if action == "left":
                game_state = left(game_state)
            if action == "right":
                game_state = right(game_state)
            if action == "up":
                game_state = up(game_state)
            if action == "down":
                game_state = down(game_state)

            score += get_score_difference(game_state, copy.deepcopy(game_state))

            input_data = torch.tensor(sum(game_state, []), dtype=torch.float).unsqueeze(0)

            with torch.no_grad():
                output = model(input_data)
                _, predicted = torch.max(output.data, 1)
                predicted_action = ["left", "right", "up", "down"][predicted.item()]

            if game_over(game_state):
                with open('train.log','a') as f:
                    f.write('[game] game over')
                if score > max_score:
                    max_score = score
                break

            action = predicted_action
            if action == "left":
                game_state = left(game_state)
            if action == "right":
                game_state = right(game_state)
            if action == "up":
                game_state = up(game_state)
            if action == "down":
                game_state = down(game_state)

            # Check if there is an empty tile
            empty_tile = False
            for i in range(4):
                for j in range(4):
                    if game_state[i][j] == 0:
                        empty_tile = True
                        break
                if empty_tile:
                    break

            # If there is an empty tile, randomly generate a new number (0 or 1)
            if empty_tile:
                while True:
                    i = random.randint(0, 3)
                    j = random.randint(0, 3)
                    if game_state[i][j] == 0:
                        game_state[i][j] = random.choice([0, 1])
                        break

        num_iterations += 1
        if num_iterations == 100:
            torch.save(model.state_dict(), "model.pt")

        print(f"Iterations: {num_iterations}, Max Score: {max_score}")

        if num_iterations >= 1000:
            torch.save(model.state_dict(), "model.pt")
            print("Training completed.")
            break

    display_training_result(max_score)


# Instantiate the neural network model
model = Net()

# Train the model
train(model)
