from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
#from keras.layers.advanced_activations import PReLU
from keras.layers import ELU, PReLU, LeakyReLU
import matplotlib.pyplot as plt
# %matplotlib inline

route_mark = 0.8  # routed cell will be painted by gray 0.8
state_mark = 0.5  # current cell will be painteg by gray 0.5
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1

# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# rat = (row, col) initial rat position (defaults to (0,0))

class Enviroment(object):
    def __init__(self, shape):
        self.generate_random_board(shape)
        # nrows, ncols = self._board.shape
        # self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._board[r,c] == 1.0 and np.array([r,c]) != self.end]
        # self.free_cells.remove(self.target)
        # if self._board[self.end] == 0.0:
        #     raise Exception("Invalid maze: target cell cannot be blocked!")
        # if not start in self.free_cells:
            # raise Exception("Invalid start Location: must sit on a free cell")
        self.reset(self.start)

    def reset(self, start):
        self.start = start
        self.board = np.copy(self._board)
        row, col = start[0], start[1]
        self.board[row, col] = state_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.5 * self.board.size
        self.total_reward = 0
        self.routed = set()

    def update_state(self, action):
        nrow, ncol, nmode = state_row, state_col, mode = self.state

        if self.board[state_row, state_col] > 0.0:
            self.routed.add((state_row, state_col))  # mark routed cell

        valid_actions = self.valid_actions()
                
        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in rat position
            mode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)
        self.start = np.array([nrow, ncol])

    def get_reward(self):
        state_row, state_col, mode = self.state
        if state_row == self.nrows-1 and state_col == self.ncols-1:
            return 1.0
        if mode == 'blocked':
            return self.min_reward - 1
        if (state_row, state_col) in self.routed:
            return -0.25
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        board_state = self.observe()
        return board_state, reward, status

    def observe(self):
        canvas = self.draw_env()
        board_state = canvas.reshape((1, -1))
        return board_state

    def draw_env(self):
        canvas = np.copy(self.board)
        # clear all visual marks
        for r in range(self.nrows):
            for c in range(self.ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = state_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        state_row, state_col, mode = self.state
        if state_row == self.nrows-1 and state_col == self.ncols-1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        if row == 0:
            actions.remove(1)
        elif row == self.nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == self.ncols-1:
            actions.remove(2)

        if row>0 and self.board[row-1,col] == 0.0:
            actions.remove(1)
        if row<self.nrows-1 and self.board[row+1,col] == 0.0:
            actions.remove(3)

        if col>0 and self.board[row,col-1] == 0.0:
            actions.remove(0)
        if col<self.ncols-1 and self.board[row,col+1] == 0.0:
            actions.remove(2)

        return actions
    
    def generate_random_board(self, shape):
        self.nrows, self.ncols = shape
        
        self._board = np.ones((self.nrows, self.ncols))
        for i in range(1, self.nrows - 1):
            for j in range(1, self.ncols - 1):
                if random.random() < 0.3:
                    self._board[i, j] = 0

        free_space = np.argwhere(self._board == 1)
        while True:
            start = free_space[random.randint(0, len(free_space) - 1)]
            if start[0] == 0 or start[0] == self.nrows - 1 or start[1] == 0 or start[1] == self.ncols - 1:
                break
            elif start + [1, 0] in free_space or start + [-1, 0] in free_space or start + [0, 1] in free_space or start + [0, -1] in free_space:
                break
            else :
                continue
        self.start = start

        while True:
            end = free_space[random.randint(0, len(free_space) - 1)]
            if end[0] == 0 or end[0] == self.nrows - 1 or end[1] == 0 or end[1] == self.ncols - 1:
                break
            elif end + [1, 0] in free_space or end + [-1, 0] in free_space or end + [0, 1] in free_space or end + [0, -1] in free_space:
                break
            else:
                continue
        self.end = end
        self.free_cells = [(r,c) for r in range(self.nrows) for c in range(self.ncols) if self._board[r,c] == 1 and np.array([r,c]) not in [self.start, self.end]]
        
    def set_routing_problem(self, board, start, end):
        self._board = board
        self.nrows, self.ncols = board.shape
        self.start = start
        self.end = end
        self.free_cells = [(r,c) for r in range(self.nrows) for c in range(self.ncols) if self._board[r,c] == 1 and np.array([r,c]) not in [self.start, self.end]]
        self.reset(self.start)
    
def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.routed:
        canvas[row,col] = 0.6
    state_row, state_col, _ = qmaze.state
    canvas[state_row, state_col] = 0.3   # state cell
    canvas[nrows-1, ncols-1] = 0.9 # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img

def play_game(model, qmaze, state):
    qmaze.reset(state)
    board_state = qmaze.observe()
    while True:
        prev_board_state = board_state
        # get next action
        q = model.predict(prev_board_state)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        board_state, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False
        
def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True

class ReplayMemory(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def push(self, episode):
        # episode = [board_state, action, reward, board_state_next, game_over]
        # memory[i] = episode
        # board_state == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, board_state):
        return self.model.predict(board_state, verbose=0)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # board_state 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            board_state, action, reward, board_state_next, game_over = self.memory[j]
            inputs[i] = board_state
            # There should be no target values for actions not taken.
            targets[i] = self.predict(board_state)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(board_state_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

def qtrain(model, shape, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 1500)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    routing_board = Enviroment(shape)

    # Initialize memory replay object
    memory = ReplayMemory(model, max_memory=max_memory)

    win_history = []   # history of win/lose game
    # n_free_cells = len(routing_board.free_cells)
    hsize = routing_board.board.size//2   # history window size
    win_rate = 0.0
    imctr = 1

    for epoch in range(n_epoch):
        loss = 0.0
        routing_board.generate_random_board()
        state_cell = routing_board.start
        routing_board.reset(state_cell)
        # state_cell = random.choice(routing_board.free_cells)
        game_over = False

        # get initial board_state (1d flattened canvas)
        board_state = routing_board.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = routing_board.valid_actions()
            if not valid_actions: break
            prev_board_state = board_state
            # Get next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(memory.predict(prev_board_state))

            # Apply action, get reward and new board_state
            board_state, reward, game_status = routing_board.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False

            # Store episode (memory)
            episode = [prev_board_state, action, reward, board_state, game_over]
            memory.push(episode)
            n_episodes += 1
            # Train neural network model
            inputs, targets = memory.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
    
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
        # we simply check if training has exhausted all free cells and if in all
        # cases the agent won
        if win_rate > 0.9 : epsilon = 0.05
        if sum(win_history[-hsize:]) == hsize and completion_check(model, routing_board):
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break

    # Save trained model weights and architecture, this will be used by the visualization code
    h5file = name + ".h5"
    json_file = name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)
    
def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer='adam', loss='mse')
    return model


def solve_routing(model, board, start, end):
    """Solve routing problem using trained model"""
    env = Enviroment(board.shape)
    env.set_routing_problem(board, start, end)
    
    state = env.observe()
    game_over = False
    path = []
    path.append(env.start)
    while not game_over:
        q = model.predict(state)
        action = np.argmax(q[0])
        state, reward, game_status = env.act(action)
        if game_status == 'win':
            game_over = True
        elif game_status == 'lose':
            game_over = True
        else:
            game_over = False
        path.append(env.start)
    return path

def update_board(board, path):
    for i in range(len(path)):
        board[path[i]] = 0


model = build_model(maze)
qtrain(model, maze, n_epoch=100, max_memory=8*maze.size, data_size=32)

routing1 =  np.array([
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
])

routing1_path = solve_routing(model, routing1, (0, 0), (7, 7))

routing2 = np.ones((10, 8))

start_dest_pair = [[[0, 4], [6, 3]], [[0, 6], [6, 5]], [[2, 2], [8, 1]], [[2, 5], [8, 4]]]
path_list = []

for start, dest in start_dest_pair:
    path = solve_routing(model, routing2, np.array(start), np.array(dest))
    update_board(routing2, path)
    path_list.append(path)


