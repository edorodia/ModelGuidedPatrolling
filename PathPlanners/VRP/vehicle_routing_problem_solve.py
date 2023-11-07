"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


def create_data_model():
	"""Stores the data for the problem."""
	data = {}
	resolution = 4
	nav_map = np.genfromtxt("Environment/Maps/map.txt", delimiter=' ')
	X, Y = np.meshgrid(np.arange(resolution+1, nav_map.shape[0], step=resolution),
	                   np.arange(resolution+1, nav_map.shape[1], step=resolution))
	places = np.vstack((X.ravel(), Y.ravel())).T
	
	coordinates = []
	for place in places:
		if (nav_map[place[0] - 2:place[0] + 3, place[1] - 2:place[1] + 3] == 1).all():
			coordinates.append(place)
	
	coordinates = np.array(coordinates)
	
	plt.imshow(nav_map)
	plt.plot(coordinates[:, 1], coordinates[:, 0], 'bo')
	plt.show()
	
	data["coordinates"] = coordinates
	data["distance_matrix"] = distance_matrix(coordinates, coordinates)
	data["distance_matrix"] = data["distance_matrix"].astype(int)
	
	# If a distance is larger than 2sqrt(2) * resolution, then it is not possible to go from one point to the other
	# data["distance_matrix"][data["distance_matrix"] > np.sqrt(2) * resolution] = 10000
	
	
	data["num_vehicles"] = 4
	data["start"] = [100, 103, 102, 101]
	data["end"] = [100, 103, 102, 101]
	return data


def print_solution(data, manager, routing, solution):
	"""Prints solution on console."""
	print(f"Objective: {solution.ObjectiveValue()}")
	max_route_distance = 0
	for vehicle_id in range(data["num_vehicles"]):
		index = routing.Start(vehicle_id)
		plan_output = f"Route for vehicle {vehicle_id}:\n"
		route_distance = 0
		while not routing.IsEnd(index):
			plan_output += f" {manager.IndexToNode(index)} -> "
			previous_index = index
			index = solution.Value(routing.NextVar(index))
			route_distance += routing.GetArcCostForVehicle(
					previous_index, index, vehicle_id
			)
		plan_output += f"{manager.IndexToNode(index)}\n"
		plan_output += f"Distance of the route: {route_distance}m\n"
		print(plan_output)
		max_route_distance = max(route_distance, max_route_distance)
	print(f"Maximum of the route distances: {max_route_distance}m")


def solution_to_paths(data, manager, routing, solution):
	"""Prints solution on console."""
	print(f"Objective: {solution.ObjectiveValue()}")
	max_route_distance = 0
	
	paths = []
	
	for vehicle_id in range(data["num_vehicles"]):
		
		index = routing.Start(vehicle_id)
		plan_output = f"Route for vehicle {vehicle_id}:\n"
		route_distance = 0
		
		path = [data["coordinates"][manager.IndexToNode(index)]]
		
		while not routing.IsEnd(index):
			plan_output += f" {manager.IndexToNode(index)} -> "
			previous_index = index
			index = solution.Value(routing.NextVar(index))
			route_distance += routing.GetArcCostForVehicle(
					previous_index, index, vehicle_id
			)
			
			path.append(data["coordinates"][manager.IndexToNode(index)])
		
		# Append the initial position to the end of the path
		# path.append(data["coordinates"][manager.IndexToNode(index)])
		paths.append(np.array(path))
		
		plan_output += f"{manager.IndexToNode(index)}\n"
		plan_output += f"Distance of the route: {route_distance}m\n"
		print(plan_output)
		max_route_distance = max(route_distance, max_route_distance)
	
	print(f"Maximum of the route distances: {max_route_distance}m")
	
	return paths


def plot_solution(nav, paths):
	plt.imshow(nav, cmap='gray')
	
	for path in paths:
		plt.plot(path[:, 1], path[:, 0], 'o-')
	
	plt.show()


def main():
	"""Entry point of the program."""
	# Instantiate the data problem.
	data = create_data_model()
	
	# Create the routing index manager.
	manager = pywrapcp.RoutingIndexManager(
			len(data["distance_matrix"]), data["num_vehicles"], data['start'], data['end']
	)
	
	# Create Routing Model.
	routing = pywrapcp.RoutingModel(manager)
	
	# Create and register a transit callback.
	def distance_callback(from_index, to_index):
		"""Returns the distance between the two nodes."""
		# Convert from routing variable Index to distance matrix NodeIndex.
		from_node = manager.IndexToNode(from_index)
		to_node = manager.IndexToNode(to_index)
		return data["distance_matrix"][from_node][to_node]
	
	transit_callback_index = routing.RegisterTransitCallback(distance_callback)
	
	# Define cost of each arc.
	routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
	
	# Add Distance constraint.
	dimension_name = "Distance"
	routing.AddDimension(
			transit_callback_index,
			0,  # no slack
			1000,  # vehicle maximum travel distance
			False,  # start cumul to zero
			dimension_name,
	)
	distance_dimension = routing.GetDimensionOrDie(dimension_name)
	distance_dimension.SetGlobalSpanCostCoefficient(100)
	
	# Setting first solution heuristic.
	search_parameters = pywrapcp.DefaultRoutingSearchParameters()
	search_parameters.first_solution_strategy = (
			routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
	)
	
	# Solve the problem.
	solution = routing.SolveWithParameters(search_parameters)
	
	# Print solution on console.
	if solution:
		print_solution(data, manager, routing, solution)
	else:
		print("No solution found !")
	
	paths = solution_to_paths(data, manager, routing, solution)
	
	
	# Store the paths as pickle
	import pickle
	with open('PathPlanners/VRP/vrp_paths.pkl', 'wb') as f:
		pickle.dump(paths, f)
	
	# Plot the paths
	plot_solution(np.genfromtxt("Environment/Maps/map.txt", delimiter=' '), paths)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("Interrupted by user")
