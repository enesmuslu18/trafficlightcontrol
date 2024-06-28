import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_TL_GREEN = 0
PHASE_TL_YELLOW = 1
PHASE_TL_RED = 2



class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states_1, num_actions_1, training_epochs, num_states_2, num_actions_2):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states_1 = num_states_1
        self._num_actions_1 = num_actions_1
        self._reward1_store = []
        self._reward2_store = []
        self._cumulative_wait_store_all = []
#        self._cumulative_wait_store_bus_and_taxi = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._co2_emission_store = []
        self._mean_speed_store = []
        self._noise_emission_store = []
        self._num_states_2 = num_states_2
        self._num_actions_2 = num_actions_2


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
#        self._waiting_times_bus_and_taxi = {}
        self._waiting_times_for_reward1 = {}
        self._waiting_times_for_reward2 = {}
        self._co2_emissions = {}
        self._noise_emissions = {}
        self._sum_total_reward = 0
        self._sum_neg_reward1 = 0
        self._sum_neg_reward2 = 0
        self._sum_queue_length = 0
        self._sum_waiting_time_all = 0
#        self._sum_waiting_time_bus_and_taxi = 0
        self._sum_co2_emission = 0
        self._sum_mean_speed = 0
        self._sum_noise_emission = 0
        old_total_wait_for_reward1 = 0
        old_total_wait_for_reward2 = 0
        old_state_1 = -1
        old_state_2 = -1
        old_action_1 = -1
        old_action_2 = -1
        action_1 = 0
        action_2 = 0
        nAd = 0
        nAd_1 = 0
        nAd_2 = 0
        yellow_flag_1 = 0
        yellow_flag_2 = 0
        

        while self._step < self._max_steps:
            
            if nAd_1==0:
                current_state_1 = self._get_state_1()          
                current_total_wait_for_reward1 = self._collect_waiting_times_for_reward1()# calculate reward of previous action: (change in cumulative waiting time between actions)
                reward1 = old_total_wait_for_reward1 - current_total_wait_for_reward1 # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
                if self._step != 0:
                    self._Memory.add_sample_1((old_state_1, old_action_1, reward1, current_state_1))                    
            
            if nAd_2==0:
                current_state_2 = self._get_state_2()
                current_total_wait_for_reward2 = self._collect_waiting_times_for_reward2()
                reward2 = old_total_wait_for_reward2 - current_total_wait_for_reward2
                if self._step != 0:
                    self._Memory.add_sample_2((old_state_2, old_action_2, reward2, current_state_2))
                    
                    
            if nAd_1==0:
                if yellow_flag_1==1:
                    action_1 = self._choose_action_1(current_state_1, epsilon)
                if self._step != 0 and old_action_1 != action_1 and yellow_flag_1==1:
                    self._set_yellow_phase_1(old_action_1)
                    nAd_1=self._yellow_duration
                    yellow_flag_1=0
                else:
                    self._set_green_phase_1(action_1)
                    nAd_1=self._green_duration
                    yellow_flag_1=1

            if nAd_2==0:
                if yellow_flag_2==1:
                    action_2 = self._choose_action_2(current_state_2, epsilon)
                if self._step != 0 and old_action_2 != action_2 and yellow_flag_2==1:
                    self._set_yellow_phase_2(old_action_2)
                    nAd_2=self._yellow_duration
                    yellow_flag_2=0
                else:
                    self._set_green_phase_2(action_2)
                    nAd_2=self._green_duration
                    yellow_flag_2=1
                    
                    
            nAd=min(nAd_1,nAd_2)
#            print(nAd)
            self._simulate(nAd)               
            nAd_1=nAd_1-nAd
            nAd_2=nAd_2-nAd
            



            # saving variables for later & accumulate reward
            if nAd_1==0:
                old_state_1 = current_state_1
                old_action_1 = action_1
                old_total_wait_for_reward1 = current_total_wait_for_reward1
            if nAd_2==0:
                old_state_2 = current_state_2
                old_action_2 = action_2
                old_total_wait_for_reward2 = current_total_wait_for_reward2
            
           
            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward1 < 0 and nAd_1==0:
                self._sum_neg_reward1 += reward1
            if reward2 < 0 and nAd_2==0:
                self._sum_neg_reward2 += reward2
            

        self._save_episode_stats()
        print("Total reward1:", self._sum_neg_reward1, "Total reward2:", self._sum_neg_reward2, "Total reward:", self._sum_total_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay_1()
        training_time_1 = round(timeit.default_timer() - start_time, 1)
        
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay_2()
        training_time_2 = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time_1, training_time_2


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
#            total_waiting_time_bus_and_taxi = self._collect_waiting_times_bus_and_taxi()
#            self._sum_waiting_time_bus_and_taxi += total_waiting_time_bus_and_taxi
            total_waiting_time_all = self._collect_waiting_times_all()
            self._sum_waiting_time_all += total_waiting_time_all
            total_co2_emission = self._co2_emission_value()
            self._sum_co2_emission += total_co2_emission
            total_mean_speed = self._get_mean_speed()
            self._sum_mean_speed += total_mean_speed
            total_noise_emission = self._get_noise_emission()
            self._sum_noise_emission += total_noise_emission
            
    def _get_noise_emission(self):
        
        incoming_roads = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
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
        
        incoming_roads = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
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

        """
    def _collect_waiting_times_bus_and_taxi(self):
        
        Retrieve the waiting time of every car in the incoming roads
        
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
        """
    
    def _collect_waiting_times_for_reward1(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E1", "E2", "E3"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_type = traci.vehicle.getTypeID(car_id)
            if car_type == "bus":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                if road_id in incoming_roads:
                    self._waiting_times_for_reward1[car_id] = wait_time * 2
                else:
                    if car_id in self._waiting_times_for_reward1:
                        del self._waiting_times_for_reward1[car_id]
            elif car_type == "taxi":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
                if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                    self._waiting_times_for_reward1[car_id] = wait_time * 1
                else:
                    if car_id in self._waiting_times_for_reward1: # a car that was tracked has cleared the intersection
                        del self._waiting_times_for_reward1[car_id]
            
        total_waiting_time_for_reward1 = sum(self._waiting_times_for_reward1.values())
        return total_waiting_time_for_reward1
    
    
    def _collect_waiting_times_for_reward2(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E4", "E5", "E6", "E7"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_type = traci.vehicle.getTypeID(car_id)
            if car_type == "bus":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                if road_id in incoming_roads:
                    self._waiting_times_for_reward2[car_id] = wait_time * 2
                else:
                    if car_id in self._waiting_times_for_reward2:
                        del self._waiting_times_for_reward2[car_id]
            elif car_type == "taxi":
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
                if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                    self._waiting_times_for_reward2[car_id] = wait_time * 1
                else:
                    if car_id in self._waiting_times_for_reward2: # a car that was tracked has cleared the intersection
                        del self._waiting_times_for_reward2[car_id]
            
        total_waiting_time_for_reward2 = sum(self._waiting_times_for_reward2.values())
        return total_waiting_time_for_reward2
    
    
    def _co2_emission_value(self):
        
        incoming_roads = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
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
            


    def _choose_action_1(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions_1 - 1) # random action
        else:
            return np.argmax(self._Model.predict_one_1(state)) # the best action given the current state
        
        
    def _choose_action_2(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions_2 - 1) # random action
        else:
            return np.argmax(self._Model.predict_one_2(state)) # the best action given the current state


    def _set_yellow_phase_1(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 0 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        if old_action == 0:
            traci.trafficlight.setPhase("tl_01", yellow_phase_code)
        elif old_action == 1:
            traci.trafficlight.setPhase("tl_02", yellow_phase_code)
        elif old_action == 2:
            traci.trafficlight.setPhase("tl_03", yellow_phase_code)
            
            
    def _set_yellow_phase_2(self, old_action):
        
        yellow_phase_code = old_action * 0 + 1
        if old_action == 0:
            traci.trafficlight.setPhase("tl_04", yellow_phase_code)
        elif old_action == 1:
            traci.trafficlight.setPhase("tl_05", yellow_phase_code)
        elif old_action == 2:
            traci.trafficlight.setPhase("tl_06", yellow_phase_code)
        elif old_action == 3:
            traci.trafficlight.setPhase("tl_07", yellow_phase_code)


    def _set_green_phase_1(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("tl_01", PHASE_TL_GREEN)
            traci.trafficlight.setPhase("tl_02", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_03", PHASE_TL_RED)
        elif action_number == 1:
            traci.trafficlight.setPhase("tl_01", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_02", PHASE_TL_GREEN)
            traci.trafficlight.setPhase("tl_03", PHASE_TL_RED)
        elif action_number == 2:
            traci.trafficlight.setPhase("tl_01", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_02", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_03", PHASE_TL_GREEN)
            
            
    def _set_green_phase_2(self, action_number):
        
        if action_number == 0:
            traci.trafficlight.setPhase("tl_04", PHASE_TL_GREEN)
            traci.trafficlight.setPhase("tl_05", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_06", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_07", PHASE_TL_RED)
        elif action_number == 1:
            traci.trafficlight.setPhase("tl_04", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_05", PHASE_TL_GREEN)
            traci.trafficlight.setPhase("tl_06", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_07", PHASE_TL_RED)
        elif action_number == 2:
            traci.trafficlight.setPhase("tl_04", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_05", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_06", PHASE_TL_GREEN)
            traci.trafficlight.setPhase("tl_07", PHASE_TL_RED)
        elif action_number == 3:
            traci.trafficlight.setPhase("tl_04", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_05", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_06", PHASE_TL_RED)
            traci.trafficlight.setPhase("tl_07", PHASE_TL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        
        halt_1 = traci.edge.getLastStepHaltingNumber("E1")
        halt_2 = traci.edge.getLastStepHaltingNumber("E2")
        halt_3 = traci.edge.getLastStepHaltingNumber("E3")
        halt_4 = traci.edge.getLastStepHaltingNumber("E4")
        halt_5 = traci.edge.getLastStepHaltingNumber("E5")
        halt_6 = traci.edge.getLastStepHaltingNumber("E6")
        halt_7 = traci.edge.getLastStepHaltingNumber("E7")
        
        queue_length = halt_1 + halt_2 + halt_3 + halt_4 + halt_5 + halt_6 + halt_7
        return queue_length
    
    def _get_mean_speed(self):
        
        mean_speed_1 = traci.edge.getLastStepMeanSpeed("E1")
        mean_speed_2 = traci.edge.getLastStepMeanSpeed("E2")
        mean_speed_3 = traci.edge.getLastStepMeanSpeed("E3")
        mean_speed_4 = traci.edge.getLastStepMeanSpeed("E4")
        mean_speed_5 = traci.edge.getLastStepMeanSpeed("E5")
        mean_speed_6 = traci.edge.getLastStepMeanSpeed("E6")
        mean_speed_7 = traci.edge.getLastStepMeanSpeed("E7")
        
        total_mean_speed = mean_speed_1 + mean_speed_2 + mean_speed_3 + mean_speed_4 + mean_speed_5 + mean_speed_6 + mean_speed_7
        return total_mean_speed


    def _get_state_1(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state_1 = np.zeros(self._num_states_1)
       
        car_list_1 = traci.vehicle.getIDList()
        karmasa_1 = np.zeros(30)

        for car_id in car_list_1:
            
            car_type_1 = traci.vehicle.getTypeID(car_id)
            
            if (car_type_1 == "bus") or (car_type_1 == "taxi"):
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                lane_pos = 500 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 300 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 10:
                    lane_cell = 0
                elif lane_pos < 20:
                    lane_cell = 1
                elif lane_pos < 35:
                    lane_cell = 2
                elif lane_pos < 55:
                    lane_cell = 3
                elif lane_pos < 80:
                    lane_cell = 4
                elif lane_pos < 110:
                    lane_cell = 5
                elif lane_pos < 145:
                    lane_cell = 6
                elif lane_pos < 215:
                    lane_cell = 7
                elif lane_pos < 350:
                    lane_cell = 8
                elif lane_pos <= 500:
                    lane_cell = 9

                # finding the road where the car is located 
                
                if road_id == "E1":
                    road_group = 0
                elif road_id == "E2":
                    road_group = 1
                elif road_id == "E3":
                    road_group = 2
                else:
                    road_group = -1
                    
                    
                lane_speed = traci.lane.getMaxSpeed(lane_id)  # şerit hızı
                car_speed = traci.vehicle.getSpeed(car_id)  # aracın hızı
                

                if road_group >= 1 and road_group <= 2:
                    car_position_1 = int(str(road_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-29
                    car_count = int(str(road_group) + str(lane_cell))  # for car count in the cells
                    valid_car = True
                elif road_group == 0:
                    car_position_1 = lane_cell
                    car_count = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    
                    if car_type_1 == "bus":
                        state_1[car_position_1] = 2  # write the position of the car car_id in the state array in the form of "cell occupied"                   
                    elif car_type_1 == "taxi":
                        state_1[car_position_1] = 1
                        
                        
                    if(car_speed / lane_speed) > 1:
                        veh_mean_speed = 1
                    else:
                        veh_mean_speed = (car_speed / lane_speed)
                        
                    
                    if karmasa_1[car_count] == 0:
                        state_1[car_position_1 + 30] = veh_mean_speed
                        karmasa_1[car_count] += 1
                        
                    else:
                        state_1[car_position_1 + 30] = ((state_1[car_position_1 + 30] * karmasa_1[car_count] + veh_mean_speed) * lane_speed) / ((karmasa_1[car_count] + 1) * lane_speed)
                        karmasa_1[car_count] += 1
                        
                    
#            else:
#                pass

        return state_1
    
    
    def _get_state_2(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state_2 = np.zeros(self._num_states_2)
       
        car_list_2 = traci.vehicle.getIDList()
        karmasa_2 = np.zeros(40)

        for car_id in car_list_2:
            
            car_type_2 = traci.vehicle.getTypeID(car_id)
            
            if (car_type_2 == "bus") or (car_type_2 == "taxi"):
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                road_id = traci.vehicle.getRoadID(car_id)
                lane_pos = 500 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 300 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 10:
                    lane_cell = 0
                elif lane_pos < 20:
                    lane_cell = 1
                elif lane_pos < 35:
                    lane_cell = 2
                elif lane_pos < 55:
                    lane_cell = 3
                elif lane_pos < 80:
                    lane_cell = 4
                elif lane_pos < 110:
                    lane_cell = 5
                elif lane_pos < 145:
                    lane_cell = 6
                elif lane_pos < 215:
                    lane_cell = 7
                elif lane_pos < 350:
                    lane_cell = 8
                elif lane_pos <= 500:
                    lane_cell = 9

                # finding the road where the car is located 
                
                if road_id == "E4":
                    road_group = 0
                elif road_id == "E5":
                    road_group = 1
                elif road_id == "E6":
                    road_group = 2
                elif road_id == "E7":
                    road_group = 3
                else:
                    road_group = -1
                    
                    
                lane_speed = traci.lane.getMaxSpeed(lane_id)  # şerit hızı
                car_speed = traci.vehicle.getSpeed(car_id)  # aracın hızı
                

                if road_group >= 1 and road_group <= 3:
                    car_position_2 = int(str(road_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-29
                    car_count = int(str(road_group) + str(lane_cell))  # for car count in the cells
                    valid_car = True
                elif road_group == 0:
                    car_position_2 = lane_cell
                    car_count = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    
                    if car_type_2 == "bus":
                        state_2[car_position_2] = 2  # write the position of the car car_id in the state array in the form of "cell occupied"
                        
                    elif car_type_2 == "taxi":
                        state_2[car_position_2] = 1
                        
                    if(car_speed / lane_speed) > 1:
                        veh_mean_speed = 1
                    else:
                        veh_mean_speed = (car_speed / lane_speed)
                        
                    
                    if karmasa_2[car_count] == 0:
                        state_2[car_position_2 + 40] = veh_mean_speed
                        karmasa_2[car_count] += 1
                        
                    else:
                        state_2[car_position_2 + 40] = ((state_2[car_position_2 + 40] * karmasa_2[car_count] + veh_mean_speed) * lane_speed) / ((karmasa_2[car_count] + 1) * lane_speed)
                        karmasa_2[car_count] += 1
                        
                    
#            else:
#                pass

        return state_2


    def _replay_1(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch1 = self._Memory.get_samples_1(self._Model.batch_size)
        

        if len(batch1) > 0:  # if the memory is full enough
            states_1 = np.array([val[0] for val in batch1])  # extract states from the batch
            next_states_1 = np.array([val[3] for val in batch1])  # extract next states from the batch

            # prediction
            q_s_a_1 = self._Model.predict_batch_1(states_1)  # predict Q(state), for every sample
            q_s_a_d_1 = self._Model.predict_batch_1(next_states_1)  # predict Q(next_state), for every sample

            # setup training arrays
            x1 = np.zeros((len(batch1), self._num_states_1))
            y1 = np.zeros((len(batch1), self._num_actions_1))
            
        

            for i, b in enumerate(batch1):
                state1, action1, reward1, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q_1 = q_s_a_1[i]  # get the Q(state) predicted before
                current_q_1[action1] = reward1 + self._gamma * np.amax(q_s_a_d_1[i])  # update Q(state, action)
                x1[i] = state1
                y1[i] = current_q_1  # Q(state) that includes the updated action value
                
            

            self._Model.train_batch_1(x1, y1)  # train the NN
            
            
            
    def _replay_2(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        
        batch2 = self._Memory.get_samples_2(self._Model.batch_size)

        
            
        if len(batch2) > 0:  # if the memory is full enough
            states_2 = np.array([val[0] for val in batch2])  # extract states from the batch
            next_states_2 = np.array([val[3] for val in batch2])  # extract next states from the batch

            # prediction
            q_s_a_2 = self._Model.predict_batch_2(states_2)  # predict Q(state), for every sample
            q_s_a_d_2 = self._Model.predict_batch_2(next_states_2)  # predict Q(next_state), for every sample

            # setup training arrays
            x2 = np.zeros((len(batch2), self._num_states_2))
            y2 = np.zeros((len(batch2), self._num_actions_2))

            
                
            for i, b in enumerate(batch2):
                state2, action2, reward2, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q_2 = q_s_a_2[i]  # get the Q(state) predicted before
                current_q_2[action2] = reward2 + self._gamma * np.amax(q_s_a_d_2[i])  # update Q(state, action)
                x2[i] = state2
                y2[i] = current_q_2  # Q(state) that includes the updated action value

            
            self._Model.train_batch_2(x2, y2)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward1_store.append(self._sum_neg_reward1)  # how much negative reward in this episode
        self._reward2_store.append(self._sum_neg_reward2)
        self._cumulative_wait_store_all.append(self._sum_waiting_time_all)  # total number of seconds waited by cars in this episode
#        self._cumulative_wait_store_bus_and_taxi.append(self._sum_waiting_time_bus_and_taxi)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        self._co2_emission_store.append(self._sum_co2_emission * self._max_steps)
        self._mean_speed_store.append(self._sum_mean_speed / self._max_steps)
        self._noise_emission_store.append(self._sum_noise_emission)


    @property
    def reward1_store(self):
        return self._reward1_store
    
    @property
    def reward2_store(self):
        return self._reward2_store


    @property
    def cumulative_wait_store_all(self):
        return self._cumulative_wait_store_all
    
#    @property
#    def cumulative_wait_store_bus_and_taxi(self):
#        return self._cumulative_wait_store_bus_and_taxi

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