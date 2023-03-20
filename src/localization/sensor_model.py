import numpy as np
# from localization.scan_simulator_2d import PyScanSimulator2D
# # Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# # if any error re: scan_simulator_2d occurs

# import rospy
# import tf
# from nav_msgs.msg import OccupancyGrid
# from tf.transformations import quaternion_from_euler

import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle")
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        # Precompute the sensor model table
        self.sensor_model_table = None
        # s = time.time()
        # self.precompute_sensor_model()
        # print("Original method: ", time.time()-s)

        s = time.time()
        self.faster_precompute_sensor_model()
        print("Fast method: ", time.time()-s)


        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)

    def probability_sensor_model(self, z_k, d):
        p_hit = p_short = p_max = p_rand = 0
        z_max = self.table_width
        eps = 1

        # Compute the normalization term for the Gaussian probability in Case 1
        guassian = lambda x: 1 / np.sqrt(2*np.pi*self.sigma_hit*self.sigma_hit) * np.exp(-(x - d)**2 / (2*self.sigma_hit*self.sigma_hit))
        eta = np.sum([guassian(x) for x in range(z_max)])

        # Case 1: the laser scan hit the obstacle; Gaussian distribution around the ground truth obstacle d
        if 0 <= z_k and z_k <= z_max:
            p_hit = (1 / eta) * 1 / np.sqrt(2*np.pi*self.sigma_hit*self.sigma_hit) * np.exp(-(z_k - d)**2 / (2*self.sigma_hit*self.sigma_hit))
        # Case 2: the scan is too short due to unknown obstacles
        if 0 <= z_k and z_k <= d and d != 0:
            p_short = 2 / d * (1 - z_k / d)
        # Case 3: the scan is very large due to beams not bouncing back
        if z_max - eps <= z_k and z_k <= z_max:
            p_max = 1 / eps
        # Case 4: assume a random scan
        if 0 <= z_k and z_k <= z_max:
            p_rand = 1 / z_max
        
        p_zk = self.alpha_hit * p_hit + self.alpha_short * p_short + self.alpha_max * p_max + self.alpha_rand * p_rand
        return p_zk

    def faster_probability_sensor_model(self, Z, D):
        # Z, D is 2darray of shape (W, W)

        p_hit = p_short = p_max = p_rand = 0
        z_max = self.table_width - 1
        eps = 1

        # Case 1
        mask1 = (Z >= 0) * (Z <= z_max)
        guassian = 1 / np.sqrt(2*np.pi*self.sigma_hit*self.sigma_hit) * np.exp(- np.square(Z - D) / (2*self.sigma_hit*self.sigma_hit))
        eta = np.sum(guassian, axis=0, keepdims=1)
        P_hit = mask1 * (guassian / eta)

        # Case 2
        mask2 = (Z >= 0) * (Z <= D) * (D != 0)
        maskd = (D == 0)
        D = D + maskd*1e-8  # in case divided by zero
        P_short = mask2 * (2 * (1 - Z / D) / D)

        # Case 3
        mask3 = (Z == z_max)
        P_max = mask3 * 1

        # Case 4
        mask4 = (Z >= 0) * (Z <= z_max)
        P_rand = mask4 * (1 / z_max)

        return self.alpha_hit * P_hit + self.alpha_short * P_short + self.alpha_max * P_max + self.alpha_rand * P_rand
    

    def faster_precompute_sensor_model(self):
        Z, D = np.meshgrid(np.arange(self.table_width), np.arange(self.table_width), indexing='ij')
        self.sensor_model_table = self.faster_probability_sensor_model(Z, D)
        # ax = plt.figure().add_subplot(projection='3d')
        # x = np.arange(self.table_width)
        # y = np.arange(self.table_width)

        # x, y = np.meshgrid(x, y)

        # surf = ax.plot_surface(x, y, self.sensor_model_table)
        # plt.savefig("table.png")



    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        func = np.vectorize(self.probability_sensor_model)
        self.sensor_model_table = np.fromfunction(func, (self.table_width, self.table_width))

        # Normalize the table so that each column sum to 1
        self.sensor_model_table = self.sensor_model_table / self.sensor_model_table.sum(axis=0, keepdims=1)

        ax = plt.figure().add_subplot(projection='3d')
        x = np.arange(self.table_width)
        y = np.arange(self.table_width)

        x, y = np.meshgrid(x, y)

        surf = ax.plot_surface(x, y, self.sensor_model_table)
        plt.savefig("table.png")



    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles)   # (N, num_beams_per_particle)
        N, m = scans.shape

        lidar_scale_to_map_scale = 
        # Scale lidar observations and ray casting scans from meters to pixels
        observation = observation / (self.map_resolution * 1)
        scans = scans / (self.map_resolution)

        # Clip all lidar and ray casting distances between [0, zmax]
        observation = np.clip(observation, 0, self.table_width) # (num_beams_per_particle)
        scans = np.clip(scans, 0, self.table_width)

        observation = np.repeat(observation, N, axis=0)
        probability = self.sensor_model_table[observation, scans]

        probability = np.prod(probability, axis=1)

        return probability

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        # March 20: Added here to get resolution (not know if it is desired)
        self.map_resolution = map_msg.info.resolution

        print("Map initialized")


model = SensorModel()