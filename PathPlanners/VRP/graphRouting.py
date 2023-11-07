import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib


class PatrollingGraphRoutingProblem:
	
	def __init__(self, navigation_map: np.ndarray,
	             scale: int,
	             n_agents: int,
	             max_distance: float,
	             initial_positions: np.ndarray,
	             final_positions: np.ndarray = None):
		
		self.navigation_map = navigation_map
		self.scale = scale
		self.n_agents = n_agents
		self.max_distance = max_distance
		self.initial_positions = initial_positions
		self.final_positions = final_positions
		self.waypoints = {agent_id: [] for agent_id in range(n_agents)}
		
		# Create the graph
		self.G = create_graph_from_map(self.navigation_map, self.scale)
		
		# Create the ground truth #
		self.information_map = np.ones_like(self.navigation_map)
		
		# Rendering variables #
		self.fig = None
		self.colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan',
		               'magenta']
		self.markers = ['o', 'v', '*', 'p', '>', 's', 'p', '*', 'h', 'H', 'D', 'd']
	
	def reset(self):
		# Reset all the variables of the scenario #
		
		self.information_map = np.ones_like(self.navigation_map)
		
		self.agent_positions = self.initial_positions.copy()
		
		# Reset the rewards #
		self.rewards = 0.0
		
		self.waypoints = {agent_id: [list(self.G.nodes[initial_position]['position'])] for agent_id, initial_position in zip(range(self.n_agents), self.initial_positions)}
		self.agent_distances = {agent_id: 0 for agent_id in range(self.n_agents)}
		
		# Input the initial positions to the model
		new_position_coordinates = np.array([self.G.nodes[new_position]['position'] for new_position in self.agent_positions])
		
		for position in new_position_coordinates:
			
			self.information_map[position[1]-self.scale:position[1]+self.scale+1, position[0]-self.scale:position[0]+self.scale+1] -= 0.5
	
	def update_maps(self):
		""" Update the idleness and information maps """
		
		# Input the initial positions to the model
		
		new_position_coordinates = np.array([self.G.nodes[new_position]['position'] for new_position in self.agent_positions if new_position != -1])
		
		# Check if no new positions are available
		if new_position_coordinates.shape[0] != 0:
			
			for position in new_position_coordinates:
				self.rewards += self.information_map[position[1] - self.scale:position[1] + self.scale + 1, position[0] - self.scale : position[0] + self.scale + 1].sum()
				self.information_map[position[1] - self.scale :position[1] + self.scale + 1, position[0] - self.scale:position[0] + self.scale + 1] -= 0.5
			
	
	def step(self, new_positions: np.ndarray):
		
		# Check if the new positions are neighbors of the current positions of the agents
		for i in range(self.n_agents):
			
			if new_positions[i] == -1:
				continue
			
			if new_positions[i] not in list(self.G.neighbors(self.agent_positions[i])):
				raise ValueError('The new positions are not neighbors of the current positions of the agents')
			
			# Compute the distance traveled by the agents using the edge weight
			self.agent_distances[i] += self.G[self.agent_positions[i]][new_positions[i]]['weight']
		
		# Update the positions of the agents
		self.agent_positions = new_positions.copy()
		
		# Update the waypoints
		for agent_id, new_position in enumerate(new_positions):
			if new_position != -1:
				# Append the position from the node
				self.waypoints[agent_id].append(list(self.G.nodes[new_position]['position']))
		
		# Update the idleness and information maps with the rewards
		self.update_maps()
		
		done = np.asarray([agent_distance > self.max_distance for agent_distance in self.agent_distances.values()]).all()
		
		done = done or np.asarray([agent_position == -1 for agent_position in self.agent_positions]).all()
		
		# Return the rewards
		return self.rewards, done
	
	def evaluate_path(self, multiagent_path: dict, render=False) -> float:
		""" Evaluate a path """
		
		self.reset()
		
		if render:
			self.render()
		
		done = False
		t = 0
		
		final_rewards = 0.0
		
		while not done:
			
			next_positions = np.zeros_like(self.agent_positions)
			
			for i in range(self.n_agents):
				if t < len(multiagent_path[i]):
					next_positions[i] = multiagent_path[i][t]
				else:
					next_positions[i] = -1
			
			new_rewards, done = self.step(next_positions)
			
			final_rewards += new_rewards
			
			if render:
				self.render()
			
			t += 1
		
		return final_rewards
	
	def render(self):
		
		if self.fig is None:
			
			self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
			
			self.ax.imshow(self.navigation_map, cmap='gray', vmin=0, vmax=1, alpha=1)
			self.d1 = self.ax.imshow(self.information_map, cmap='hot', vmin=0, vmax=1, alpha=0.5)
			
			self.agents_render_pos = []
			
			for i in range(self.n_agents):
				# Obtain the agent position from node to position
				agent_position_coords = self.G.nodes[self.agent_positions[i]]['position']
				self.agents_render_pos.append(
						self.ax.plot(agent_position_coords[0], agent_position_coords[1], color=self.colors[i], marker=self.markers[i], markersize=10, alpha=0.35)[0])
		
		else:
			
			for i in range(self.n_agents):
				traj = np.asarray(self.waypoints[i])
				# Plot the trajectory of the agent
				self.agents_render_pos[i].set_data(traj[:, 0], traj[:, 1])
			
			self.d1.set_data(self.information_map)
		
		self.fig.canvas.draw()
		plt.pause(0.01)


def create_graph_from_map(navigation_map: np.ndarray, resolution: int):
	""" Create a graph from a navigation map """
	
	# Obtain the scaled navigation map
	scaled_navigation_map = navigation_map[::resolution, ::resolution]
	
	# Obtain the positions of the nodes
	visitable_positions = np.column_stack(np.where(scaled_navigation_map == 1))
	
	# Create the graph
	G = nx.Graph()
	
	# Add the nodes
	for i, position in enumerate(visitable_positions):
		G.add_node(i, position=position[::-1] * resolution, coords=position * resolution)
	
	# Add the edges
	for i, position in enumerate(visitable_positions):
		for j, other_position in enumerate(visitable_positions):
			if i != j:
				if np.linalg.norm(position - other_position) <= np.sqrt(2):
					G.add_edge(i, j, weight=np.linalg.norm(position - other_position) * resolution)
	
	return G


def plot_graph(G: nx.Graph, path: list = None, ax=None, cmap_str='Reds', draw_nodes=True):
	if ax is None:
		plt.figure()
		ax = plt.gca()
	
	positions = nx.get_node_attributes(G, 'position')
	positions = {key: np.asarray([value[0], -value[1]]) for key, value in positions.items()}
	
	if draw_nodes:
		nx.draw(G, pos=positions, with_labels=True, node_color='gray', arrows=True, ax=ax)
	
	if path is not None:
		cmap = matplotlib.colormaps[cmap_str]
		red_shades = cmap(np.linspace(0, 1, len(path)))
		nx.draw_networkx_nodes(G, pos=positions, nodelist=path, node_color=red_shades, ax=ax)
	
	return ax


def path_length(G: nx.Graph, path: list) -> float:
	length = 0
	
	for i in range(len(path) - 1):
		length += G[path[i]][path[i + 1]]['weight']
	
	return length


def random_shorted_path(G: nx.Graph, p0: int, p1: int) -> list:
	random_G = G.copy()
	for edge in random_G.edges():
		random_G[edge[0]][edge[1]]['weight'] = np.random.rand()
	
	return nx.shortest_path(random_G, p0, p1, weight='weight')[1:]


def create_random_path_from_nodes(G: nx.Graph, start_node: int, distance: float, final_node: int = None) -> list:
	""" Select random nodes and create random path to reach them """
	
	path = []
	remain_distance = distance
	
	# Append the start node
	path.append(start_node)
	
	while path_length(G, path) < distance:
		# Select a random node
		next_node = np.random.choice(G.nodes())
		
		# Obtain a random path to reach it
		new_path = random_shorted_path(G, path[-1], next_node)
		path.extend(new_path)
		
		# Compute the distance of path
		remain_distance -= path_length(G, new_path)
	
	# Append the shortest path to the start node
	G_random = G.copy()
	# Generate random weights
	for edge in G_random.edges():
		G_random[edge[0]][edge[1]]['weight'] = np.random.rand()
	
	# Append the shortest path to the start node
	if final_node is not None:
		path.extend(nx.shortest_path(G_random, path[-1], final_node, weight='weight')[1:])
	else:
		path.extend(nx.shortest_path(G_random, path[-1], start_node, weight='weight')[1:])
	
	return path[1:]


def create_multiagent_random_paths_from_nodes(G, initial_positions, distance, final_positions=None):
	
	if final_positions is not None:
		multiagent_path = {agent_id: create_random_path_from_nodes(G, initial_positions[agent_id], distance, final_positions[agent_id]) for agent_id in
		                   range(len(initial_positions))}
	else:
		multiagent_path = {agent_id: create_random_path_from_nodes(G, initial_positions[agent_id], distance) for
		                   agent_id in range(len(initial_positions))}
	
	return multiagent_path


def cross_operation_between_paths(G: nx.Graph, path1, path2):
	""" Perform a cross operation between two paths. """
	
	# Transform the paths into numpy arrays
	path1 = np.asarray(path1)
	path2 = np.asarray(path2)
	
	# Obtain the split points
	i = np.random.randint(0, len(path1), size=2)
	i.sort()
	
	j = np.random.randint(0, len(path2), size=2)
	j.sort()
	
	resulting_path_1 = np.concatenate((path1[:i[0]],
	                                   nx.shortest_path(G, path1[i[0]], path2[j[0]])[:-1],
	                                   path2[j[0]:j[1]],
	                                   nx.shortest_path(G, path2[j[1]], path1[i[1]])[:-1],
	                                   path1[i[1]:]
	                                   ))
	
	resulting_path_2 = np.concatenate((path2[:j[0]],
	                                   nx.shortest_path(G, path2[j[0]], path1[i[0]])[:-1],
	                                   path1[i[0]:i[1]],
	                                   nx.shortest_path(G, path1[i[1]], path2[j[1]])[:-1],
	                                   path2[j[1]:]
	                                   ))
	
	return resulting_path_1.tolist(), resulting_path_2.tolist()


def mutation_operation(G: nx.Graph, path, mut_prob=0.1):
	""" Alter a random node to its closest neighbor. """
	
	new_path = path.copy()
	
	# Select a random node
	for i in range(1, len(new_path) - 1):
		
		if np.random.rand() < mut_prob:
			
			# Obtain the common neighbors between the node i and the next node
			common_neighbors_1 = list(nx.neighbors(G, new_path[i - 1]))
			common_neighbors_2 = list(nx.neighbors(G, new_path[i + 1]))
			common_neighbors = [node for node in common_neighbors_1 if
			                    node in common_neighbors_2 and node != new_path[i]]
			
			# Select a random neighbor
			if len(common_neighbors) > 0:
				new_path[i] = np.random.choice(common_neighbors)
	
	return new_path


if __name__ == '__main__':
	np.random.seed(0)
	
	navigation_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
	N_agents = 2
	initial_positions = np.array([10, 10, 30, 40])[:N_agents]
	final_positions = np.array([10, 10, 30, 40])[:N_agents]
	scale = 3
	
	environment = PatrollingGraphRoutingProblem(navigation_map=navigation_map,
	                                            n_agents=N_agents,
	                                            initial_positions=initial_positions,
	                                            final_positions=final_positions,
	                                            scale=scale,
	                                            max_distance=350.0,
	                                            )
	
	path = create_multiagent_random_paths_from_nodes(environment.G, initial_positions, 150, final_positions)
	
	path_1, path_2 = cross_operation_between_paths(environment.G, path[0], path[1])
	path_crossed = {0: path_1, 1: path_2}
	
	environment.evaluate_path(path, render=True)
	environment.evaluate_path(path_crossed, render=True)
	
	plt.pause(1000)
	
	# Plot the graph to visualize the crossing
	fig, axs = plt.subplots(2, 2, figsize=(10, 5))
	plot_graph(environment.G, path=path[0], draw_nodes=True, ax=axs[0, 0])
	plot_graph(environment.G, path=path[1], draw_nodes=True, ax=axs[0, 1], cmap_str='Greens')
	plot_graph(environment.G, path=path_crossed[0], draw_nodes=True, ax=axs[1, 0])
	plot_graph(environment.G, path=path_crossed[1], draw_nodes=True, ax=axs[1, 1], cmap_str='Greens')
	plt.show()
	
	plt.show()
