# import requirements
from __future__ import print_function
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt

### main constants

# markers
free_cell_mark = 1.0
visited_cell_mark = 0.8 
agent_cell_mark = 0.5  
obstacle_cell_mark = 0.0
destination_cell_mark = 0.3

# rewards
GOAL = 1.0
REVISIT = -1.0
VALID = -0.25
INVALID = -5.0
# BLOCKED = -0.75


# Actions 
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3
VIA = 4
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
    VIA: 'via'
}

ACTION_NUMS = len(actions_dict)

# Exploration factor
epsilon = 0.1

# Grid Environment
class Environment(object):
    def __init__(self, shape, start, dest, obstacle_cells=None):
        self.height = shape[0]
        self.width = shape[1]
        self.grid = np.ones([self.height, self.width, 2])
        self.free_cells = [(r,c,0) for r in range(self.grid.shape[0]) for c in range(self.grid.shape[1]) if self.grid[r,c,0] == 1.0]
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                self.free_cells.append((r,c,1)) 
        self.obstacle_cells = []
        
        self.reset(start, dest, obstacle_cells)

    def reset(self, start, dest, obstacle_cells=None):
        self.grid = np.ones([self.height, self.width, 2])
        
        self.start = start
        self.agent = self.start
        if self.start in self.free_cells:
            self.free_cells.remove(self.start)
            self.free_cells.remove((self.start[0], self.start[1], 1))
        self.grid[self.agent[0], self.agent[1], self.agent[2]] = agent_cell_mark

        self.destination = dest
        while np.array_equal(self.start, self.destination):
            self.destination = (np.random.randint(self.grid.shape[0]), np.random.randint(self.grid.shape[1]), 0)
        self.grid[self.destination[0], self.destination[1], self.destination[2]] = destination_cell_mark

        self.free_cells = [(r,c,0) for r in range(self.grid.shape[0]) for c in range(self.grid.shape[1]) if self.grid[r,c,0] == 1.0]
        for r in range(self.grid.shape[0]):
            for c in range(self.grid.shape[1]):
                self.free_cells.append((r,c,1)) 

        if obstacle_cells != None:
            for x, y, pcb in obstacle_cells:
                if (x,y,0) != self.start and (x,y,) != (self.destination[0], self.destination[1]):
                    self.grid[x, y, pcb] = obstacle_cell_mark
                if (x,y,pcb) in self.free_cells:
                    self.free_cells.remove((x,y,pcb))
                self.obstacle_cells.append((x,y,pcb))  

        self.min_reward = -0.5 * self.grid.size
        self.agent_state = (self.agent[0], self.agent[1], self.agent[2], 'start')
        
        self.total_reward = 0.0
        
        self.visited = set()

    def update_state(self, action):
        nrow, ncol, pcb, nmode = self.agent_state
        agent_row = nrow
        agent_col = ncol

        if self.grid[agent_row, agent_col, pcb] > 0.0:
            self.visited.add((agent_row, agent_col, pcb))  # mark visited cell

        valid_actions = self.get_valid_actions()
                
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
            if action == VIA:
                pcb = 1
        else:                  # invalid action, no change in agent position
            nmode = 'invalid'

        # new state
        self.agent_state = (nrow, ncol, pcb, nmode)
        self.grid[self.agent[0], self.agent[1], self.agent[2]] = visited_cell_mark
        self.agent = (nrow, ncol, pcb)
        self.grid[self.agent[0], self.agent[1], self.agent[2]] = agent_cell_mark

    def get_reward(self):
        agent_row, agent_col, pcb, mode = self.agent_state
        if agent_row == self.destination[0] and agent_col == self.destination[1]:
            return GOAL
        if mode == 'blocked':
            #return self.min_reward - 1 ###################????
            return REVISIT
        if (agent_row, agent_col, pcb) in self.visited:
            return REVISIT
        if mode == 'invalid':
            return INVALID
        if mode == 'valid':
            return VALID

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.grid)
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        agent_row, agent_col, pcb, mode = self.agent_state
        # nrows, ncols, _ = self.grid.shape
        if agent_row == self.destination[0] and agent_col == self.destination[1]:
            return 'win'

        return 'not_over'

    def get_valid_actions(self, cell=None):
        if cell is None:
            row, col, pcb, mode = self.agent_state
        else:
            row, col, pcb = cell
        actions = [LEFT, UP, RIGHT, DOWN, VIA]
        nrows, ncols, _ = self.grid.shape
        if row == 0:
            actions.remove(UP)
        elif row == nrows-1:
            actions.remove(DOWN)

        if col == 0:
            actions.remove(LEFT)
        elif col == ncols-1:
            actions.remove(RIGHT)

        if row>0:
            if self.grid[row-1,col,pcb] == obstacle_cell_mark or self.grid[row-1,col,pcb] == visited_cell_mark:
                actions.remove(UP)
        if row<nrows-1:
            if self.grid[row+1,col,pcb] == obstacle_cell_mark or self.grid[row+1,col,pcb] == visited_cell_mark:
                actions.remove(DOWN)

        if col>0:
            if self.grid[row,col-1,pcb] == obstacle_cell_mark or self.grid[row,col-1,pcb] == visited_cell_mark:
                actions.remove(LEFT)
        if col<ncols-1:
            if self.grid[row,col+1,pcb] == obstacle_cell_mark or self.grid[row,col+1,pcb] == visited_cell_mark:
                actions.remove(RIGHT)

        if pcb == 1 or (row, col, 1) in self.visited:
            actions.remove(VIA)

        return actions
    
    def update_obstacle_cells(self, new_obstacle_cells):
        for cell in new_obstacle_cells:
            self.grid[cell[0], cell[1], cell[2]] = obstacle_cell_mark
            if cell in self.free_cells:
                self.free_cells.remove(cell)
            self.obstacle_cells.append(cell)

def render(env, pcb):
    plt.grid('on')
    nrows, ncols, _ = env.grid.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    canvas = np.copy(env.grid[:,:,pcb])
    # canvas.reshape(nrows , ncols)
    
    cmap = plt.cm.colors.ListedColormap(['black', 'green', 'blue', 'skyblue', 'gray'])
    bounds = [0.0, 0.299, 0.5, 0.799, 0.801, 1.0]
    #bounds=[0.0, 0.3, 0.5, 0.8, 1.0]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    img = plt.imshow(canvas, cmap=cmap, norm=norm, interpolation='none')
    return img

def play_game(model, env, start, dest, obstacle_cells):
    env.reset(start, dest, obstacle_cells)
    envstate = env.observe()
    count = 0
    action_list = []
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        count += 1
        action = np.argmax(q[0])
        action_list.append(action)

        # apply action, get rewards and new state
        envstate, reward, game_status = env.act(action)
        if game_status == 'win':
            print("Reached the goal in {} steps".format(count))
            return True
        elif game_status == 'lose':
            print("Failed to reach the goal in {} steps".format(count))
            print("Action list: {}".format(action_list))
            return False

def completion_check(env):
    if not env.get_valid_actions(env.start):
        print("Invalid start cell")
        return False
    if not env.get_valid_actions(env.destination):
        print("Invalid end cell")
        return False
    return True


# Memory for Episodes
class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode), 첫번째 episode 의 첫번째 element(envstate[1,64])
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
    
def build_model(env, lr=0.001):
    model = Sequential()
    model.add(Dense(env.grid.size, input_shape=(env.grid.size,)))
    model.add(PReLU())
    model.add(Dense(env.grid.size))
    model.add(PReLU())
    model.add(Dense(ACTION_NUMS))
    model.compile(optimizer='adam', loss='mse')
    return model

def qtrain(model, shape, start, dest, obstacle_cells, epochs, max_memory, data_size, weights_file = "", name = ""):
    global epsilon
    
    n_epoch = epochs
    max_memory = max_memory
    data_size = data_size
    weights_file = weights_file
    name = name
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment
    env = Environment(shape, start, dest, obstacle_cells)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []   # history of win/lose game
    hsize = env.grid.size//2   # history window size
    win_rate = 0.0

    for epoch in range(n_epoch):
        loss = 0.0
        env.reset(start, dest, obstacle_cells)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = env.observe()
        # print(envstate.reshape(env.grid.shape))

        n_episodes = 0
        while not game_over:
            valid_actions = env.get_valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = env.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
                # print(envstate.reshape(env.grid.shape))
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
                # print(envstate.reshape(env.grid.shape))
            else:
                game_over = False

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                nb_epoch=8,
                batch_size=16,
                verbose=0,
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
        if sum(win_history[-hsize:]) == hsize:
            if completion_check(env) and play_game(model, env, start, dest, obstacle_cells):
                print("Reached 100%% win rate at epoch: %d" % (epoch,))
                epsilon = 0.1
                break
            else :
                print("Failed completion check at epoch: %d" % (epoch,))
        print("epsilon: %f | sum of win: %f | hsize: %f" % (epsilon, sum(win_history[-hsize:]), hsize))

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
    return 

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

def demonstrate(model, episode_num, obstacle_cells, shape, start, dest):
    rendered_imgs_top = []
    rendered_imgs_bottom = []
    count = 0
    
    env = Environment(shape, start, dest, obstacle_cells)
    envstate = env.observe()
    rendered_imgs_top.append(render(env,0))
    plt.savefig("demo/"+str(episode_num)+"-"+str(count)+"-top.png")
    rendered_imgs_bottom.append(render(env,1))
    plt.savefig("demo/"+str(episode_num)+"-"+str(count)+"-bottom.png")
    count += 1
    action_list = []
    while True:
        prev_envstate = envstate
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])
        action_list.append(action)
        envstate, reward, game_status = env.act(action)
        obstacle_cells.append(env.agent)
        rendered_imgs_top.append(render(env,0))
        plt.savefig("demo/"+str(episode_num)+"-"+str(count)+"-top.png")
        rendered_imgs_bottom.append(render(env,1))
        plt.savefig("demo/"+str(episode_num)+"-"+str(count)+"-bottom.png")
        count += 1
        if game_status == 'win':
            print("win")
            break
        elif game_status == 'lose':
            print("lose")
            print("action list: ", action_list)
            break

routing2 = np.ones((12, 12, 2))
start_dest_pair = [[(2, 2, 0), (9, 9, 0)], [(2, 9, 0), (9, 2, 0)]]
obstacles = [(5, 4, 0),(5, 5, 0),(5, 6, 0),(5, 7, 0),(6, 4, 0),(6, 5, 0),(6, 6, 0),(6, 7, 0)]
path_list = []
for start, dest in start_dest_pair:
    obstacles.append(start)
    obstacles.append(dest)

count = 0
for start, dest in start_dest_pair:
    temp_env = Environment(routing2.shape, start, dest, obstacles)
    model = build_model(temp_env)
    qtrain(model, routing2.shape, start, dest, obstacles, epochs=100000, max_memory=8*temp_env.grid.size, data_size=32, weights_file="result_weight.h5", name="result_weight")
    demonstrate(model, count, obstacles, routing2.shape, start, dest)
    count += 1