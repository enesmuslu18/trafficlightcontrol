import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_01_GREEN = 0
PHASE_01_YELLOW = 1
PHASE_01_RED = 2

PHASE_02_GREEN = 0
PHASE_02_YELLOW = 1
PHASE_02_RED = 2

PHASE_03_GREEN = 0
PHASE_03_YELLOW = 1
PHASE_03_RED = 2



class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store_all = []
        self._cumulative_wait_store_bus_and_taxi = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._co2_emission_store = []
        self._mean_speed_store = []
        self._noise_emission_store = []


    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times_all = {}
        self._waiting_times_bus_and_taxi = {}
        self._waiting_times_for_reward = {}
        self._co2_emissions = {}
        self._noise_emissions = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time_all = 0
        self._sum_waiting_time_bus_and_taxi = 0
        self._sum_co2_emission = 0
        self._sum_mean_speed = 0
        self._sum_noise_emission = 0
        old_total_wait_for_reward = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()
            

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait_for_reward = self._collect_waiting_times_for_reward()
            reward = old_total_wait_for_reward - current_total_wait_for_reward

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait_for_reward = current_total_wait_for_reward

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            total_waiting_time_bus_and_taxi = self._collect_waiting_times_bus_and_taxi()
            self._sum_waiting_time_bus_and_taxi += total_waiting_time_bus_and_taxi
            total_waiting_time_all = self._collect_waiting_times_all()
            self._sum_waiting_time_all += total_waiting_time_all
            total_co2_emission = self._co2_emission_value()
            self._sum_co2_emission += total_co2_emission
            total_mean_speed = self._get_mean_speed()
            self._sum_mean_speed += total_mean_speed
            total_noise_emission = self._get_noise_emission()
            self._sum_noise_emission += total_noise_emission
            
    def _get_noise_emission(self):
        
        incoming_roads = ["E2", "E3", "E4"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            noise_emission = traci.vehicle.getNoiseEmission(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._noise_emissions[car_id] = noise_emission
            else:
                if car_id in self._noise_emissions:
                    del self._noise_emissions[car_id]
        total_noise_emission = sum(self._noise_emissions.values())
        return total_noise_emission
            
            
    def _collect_waiting_times_all(self):
        
        incoming_roads = ["E2", "E3", "E4"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times_all[car_id] = wait_time
            else:
                if car_id in self._waiting_times_all: # a car that was tracked has cleared the intersection
                    del self._waiting_times_all[car_id] 
        total_waiting_time_all = sum(self._waiting_times_all.values())
        return total_waiting_time_all


    def _collect_waiting_times_bus_and_taxi(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2", "E3", "E4"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_type = traci.vehicle.getTypeID(car_id)
            if car_type == "bus":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
                if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                    self._waiting_times_bus_and_taxi[car_id] = wait_time
                else:
                    if car_id in self._waiting_times_bus_and_taxi: # a car that was tracked has cleared the intersection
                        del self._waiting_times_bus_and_taxi[car_id]
            elif car_type == "taxi":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                if road_id in incoming_roads:
                    self._waiting_times_bus_and_taxi[car_id] = wait_time
                else:
                    if car_id in self._waiting_times_bus_and_taxi:
                        del self._waiting_times_bus_and_taxi[car_id]
        total_waiting_time_bus_and_taxi = sum(self._waiting_times_bus_and_taxi.values())
        return total_waiting_time_bus_and_taxi
    
    
    def _collect_waiting_times_for_reward(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2", "E3", "E4"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_type = traci.vehicle.getTypeID(car_id)
            if car_type == "bus":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
                if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                    self._waiting_times_for_reward[car_id] = wait_time * 2
                else:
                    if car_id in self._waiting_times_for_reward: # a car that was tracked has cleared the intersection
                        del self._waiting_times_for_reward[car_id]
            elif car_type == "taxi":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                if road_id in incoming_roads:
                    self._waiting_times_for_reward[car_id] = wait_time * 1
                else:
                    if car_id in self._waiting_times_for_reward:
                        del self._waiting_times_for_reward[car_id]
        total_waiting_time_for_reward = sum(self._waiting_times_for_reward.values())
        return total_waiting_time_for_reward
    
    
    def _co2_emission_value(self):
        
        incoming_roads = ["E2", "E3", "E4"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            carbon_dioxide = traci.vehicle.getCO2Emission(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._co2_emissions[car_id] = carbon_dioxide
            else:
                if car_id in self._co2_emissions:
                    del self._co2_emissions[car_id]
        total_co2_emission = sum(self._co2_emissions.values())
        return total_co2_emission
            


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 0 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        if old_action == 0:
            traci.trafficlight.setPhase("tl_01", yellow_phase_code)
#            if action_number == 1:
#                traci.trafficlight.setPhase("tl_02", yellow_phase_code)
#            elif action_number == 2:
#                traci.trafficlight.setPhase("tl_03", yellow_phase_code)
        elif old_action == 1:
            traci.trafficlight.setPhase("tl_02", yellow_phase_code)
#            if action_number == 0:
#                traci.trafficlight.setPhase("tl_01", yellow_phase_code)
#            elif action_number == 2:
#                traci.trafficlight.setPhase("tl_03", yellow_phase_code)
        elif old_action == 2:
            traci.trafficlight.setPhase("tl_03", yellow_phase_code)
#            if action_number == 0:
#                traci.trafficlight.setPhase("tl_01", yellow_phase_code)
#            elif action_number == 1:
#                traci.trafficlight.setPhase("tl_02", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("tl_01", PHASE_01_GREEN)
            traci.trafficlight.setPhase("tl_02", PHASE_02_RED)
            traci.trafficlight.setPhase("tl_03", PHASE_02_RED)
        elif action_number == 1:
            traci.trafficlight.setPhase("tl_01", PHASE_01_RED)
            traci.trafficlight.setPhase("tl_02", PHASE_02_GREEN)
            traci.trafficlight.setPhase("tl_03", PHASE_03_RED)
        elif action_number == 2:
            traci.trafficlight.setPhase("tl_01", PHASE_01_RED)
            traci.trafficlight.setPhase("tl_02", PHASE_02_RED)
            traci.trafficlight.setPhase("tl_03", PHASE_03_GREEN)



    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        
        halt_dogu = traci.edge.getLastStepHaltingNumber("E2")
        halt_kuzey = traci.edge.getLastStepHaltingNumber("E3")
        halt_batı = traci.edge.getLastStepHaltingNumber("E4")
        queue_length = halt_dogu + halt_kuzey + halt_batı
        return queue_length
    
    def _get_mean_speed(self):
        
        mean_speed_dogu = traci.edge.getLastStepMeanSpeed("E2")
        mean_speed_kuzey = traci.edge.getLastStepMeanSpeed("E3")
        mean_speed_batı = traci.edge.getLastStepMeanSpeed("E4")
        total_mean_speed = mean_speed_dogu + mean_speed_kuzey + mean_speed_batı
        return total_mean_speed


    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
       
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            
            car_type = traci.vehicle.getTypeID(car_id)
            
            if (car_type == "bus") or (car_type == "taxi"):
                lane_pos = traci.vehicle.getLanePosition(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                lane_pos = 300 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 300 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 5:
                    lane_cell = 0
                elif lane_pos < 10:
                    lane_cell = 1
                elif lane_pos < 15:
                    lane_cell = 2
                elif lane_pos < 20:
                    lane_cell = 3
                elif lane_pos < 30:
                    lane_cell = 4
                elif lane_pos < 50:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 150:
                    lane_cell = 7
                elif lane_pos < 200:
                    lane_cell = 8
                elif lane_pos <= 300:
                    lane_cell = 9

                # finding the road where the car is located 
            
                if road_id == "E2":
                    road_group = 0
                elif road_id == "E3":
                    road_group = 1
                elif road_id == "E4":
                    road_group = 2
                else:
                    road_group = -1

                if road_group >= 1 and road_group <= 2:
                    car_position = int(str(road_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-29
                    valid_car = True
                elif road_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    #car_speed = float(traci.vehicle.getSpeed(car_id) / traci.vehicle.getMaxSpeed(car_id))
                    #a = str(car_speed)
                    #x = len(a.split(".")[0]) + 1
                    #y = 1
                    #z = x + y
                    if car_type == "bus":
                        state[car_position] = 2  # write the position of the car car_id in the state array in the form of "cell occupied"
                    elif car_type == "taxi":
                        state[car_position] = 1
                    #state[car_position] = a[0:z]
                    
#            else:
#                pass

        return state


    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch

            # prediction
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store_all.append(self._sum_waiting_time_all)  # total number of seconds waited by cars in this episode
        self._cumulative_wait_store_bus_and_taxi.append(self._sum_waiting_time_bus_and_taxi)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        self._co2_emission_store.append(self._sum_co2_emission * self._max_steps)
        self._mean_speed_store.append(self._sum_mean_speed / self._max_steps)
        self._noise_emission_store.append(self._sum_noise_emission)


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store_all(self):
        return self._cumulative_wait_store_all
    
    @property
    def cumulative_wait_store_bus_and_taxi(self):
        return self._cumulative_wait_store_bus_and_taxi

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
    
    @property
    def CO2_store(self):
        return self._co2_emission_store
    
    @property
    def mean_speed_store(self):
        return self._mean_speed_store
    
    @property
    def noise_emission_store(self):
        return self._noise_emission_store