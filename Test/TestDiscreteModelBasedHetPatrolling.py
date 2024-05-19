import sys
sys.path.append('..')

from Environment.PatrollingEnvironment import DiscreteModelBasedHetPatrolling
import numpy as np
import unittest

class TestDiscreteModelBasedHetPatrolling(unittest.TestCase):

	navigation_map = np.genfromtxt('../Environment/Maps/map.txt', delimiter=' ')
	# Valid Positions for drone and ASVs #
	position_drone = np.array([[20,23]])
	positions_asv = np.array([[40,45],[10,40]])

	def test_random_drone_action(self):
		env = DiscreteModelBasedHetPatrolling( initial_air_positions = self.position_drone,
					max_air_distance = 1000,
					influence_side = 9,
					forgetting_air_factor = 0.01,	
					drone_idleness_influence = 0.20,
					n_agents = 2,
					navigation_map = self.navigation_map,
					initial_positions = self.positions_asv,
					model_based = True,
					movement_length = 2,
					resolution = 1,
					max_distance = 400,
					influence_radius = 2,
					forgetting_factor= 0.01,
					reward_type='weighted_importance',
					reward_weights=[10, 10],
					benchmark = 'shekel',
					model = 'none',
					dynamic = False,
					seed = 50000,
					int_observation = True,
					previous_exploration = False,
					pre_exploration_policy = None,
					pre_exploration_steps = 0, 
					camera_fov_angle = 160,						
					drone_height = 120,							
					n_drones = 1,									
					drone_direct_idleness_influece = False,		
					blur_data = False			
					)

		env.reset()

		posizione_random = env.action_to_position_drone(env.random_drone_action())

		print(posizione_random)

		self.assertEqual(self.navigation_map[posizione_random[0]][posizione_random[1]], 1)

	def test__check_drone_presence_true(self):
		positions_drone = np.array([[20,21]])
		env = DiscreteModelBasedHetPatrolling( initial_air_positions = positions_drone,
				max_air_distance = 1000,
				influence_side = 9,
				forgetting_air_factor = 0.01,	
				drone_idleness_influence = 0.20,
				n_agents = 2,
				navigation_map = self.navigation_map,
				initial_positions = self.positions_asv,
				model_based = True,
				movement_length = 2,
				resolution = 1,
				max_distance = 400,
				influence_radius = 2,
				forgetting_factor= 0.01,
				reward_type='weighted_importance',
				reward_weights=[10, 10],
				benchmark = 'shekel',
				model = 'none',
				dynamic = False,
				seed = 50000,
				int_observation = True,
				previous_exploration = False,
				pre_exploration_policy = None,
				pre_exploration_steps = 0, 
				camera_fov_angle = 160,						
				drone_height = 120,							
				n_drones = 1,									
				drone_direct_idleness_influece = False,		
				blur_data = False							
				)
		env.reset()
		self.assertTrue(env._check_drone_presence(np.array([20,21])))

	def test__check_drone_presence_false(self):
		positions_drone = np.array([[20,21]])
		env = DiscreteModelBasedHetPatrolling( initial_air_positions = positions_drone,
				max_air_distance = 1000,
				influence_side = 9,
				forgetting_air_factor = 0.01,	
				drone_idleness_influence = 0.20,
				n_agents = 2,
				navigation_map = self.navigation_map,
				initial_positions = self.positions_asv,
				model_based = True,
				movement_length = 2,
				resolution = 1,
				max_distance = 400,
				influence_radius = 2,
				forgetting_factor= 0.01,
				reward_type='weighted_importance',
				reward_weights=[10, 10],
				benchmark = 'shekel',
				model = 'none',
				dynamic = False,
				seed = 50000,
				int_observation = True,
				previous_exploration = False,
				pre_exploration_policy = None,
				pre_exploration_steps = 0, 
				camera_fov_angle = 160,						
				drone_height = 120,							
				n_drones = 1,									
				drone_direct_idleness_influece = False,		
				blur_data = False							
				)
		env.reset()
		self.assertFalse(env._check_drone_presence(np.array([20,22])))

	def test_action_to_position_drone(self):
		env = DiscreteModelBasedHetPatrolling( initial_air_positions = self.position_drone,
				max_air_distance = 1000,
				influence_side = 9,
				forgetting_air_factor = 0.01,	
				drone_idleness_influence = 0.20,
				n_agents = 2,
				navigation_map = self.navigation_map,
				initial_positions = self.positions_asv,
				model_based = True,
				movement_length = 2,
				resolution = 1,
				max_distance = 400,
				influence_radius = 2,
				forgetting_factor= 0.01,
				reward_type='weighted_importance',
				reward_weights=[10, 10],
				benchmark = 'shekel',
				model = 'none',
				dynamic = False,
				seed = 50000,
				int_observation = True,
				previous_exploration = False,
				pre_exploration_policy = None,
				pre_exploration_steps = 0, 
				camera_fov_angle = 160,						
				drone_height = 120,							
				n_drones = 1,									
				drone_direct_idleness_influece = False,		
				blur_data = False							
				)
		env.reset()
		self.assertTrue(np.array_equal(env.action_to_position_drone(0), np.array([1,1])))

	def test_position_to_action(self):
		env = DiscreteModelBasedHetPatrolling( initial_air_positions = self.position_drone,
				max_air_distance = 1000,
				influence_side = 9,
				forgetting_air_factor = 0.01,	
				drone_idleness_influence = 0.20,
				n_agents = 2,
				navigation_map = self.navigation_map,
				initial_positions = self.positions_asv,
				model_based = True,
				movement_length = 2,
				resolution = 1,
				max_distance = 400,
				influence_radius = 2,
				forgetting_factor= 0.01,
				reward_type='weighted_importance',
				reward_weights=[10, 10],
				benchmark = 'shekel',
				model = 'none',
				dynamic = False,
				seed = 50000,
				int_observation = True,
				previous_exploration = False,
				pre_exploration_policy = None,
				pre_exploration_steps = 0, 
				camera_fov_angle = 160,						
				drone_height = 120,							
				n_drones = 1,									
				drone_direct_idleness_influece = False,		
				blur_data = False							
				)
		env.reset()
		self.assertEqual(env.position_to_action_drone(np.array([1,1])), 0)


if __name__ == '__main__' :
	unittest.main()