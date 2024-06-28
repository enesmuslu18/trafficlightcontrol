from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from model import TrainModel
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path


if __name__ == "__main__":

    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'], 
        input_dim_1=config['num_states_1'],
        output_dim_1=config['num_actions_1'],
        input_dim_2=config['num_states_2'],
        output_dim_2=config['num_actions_2']
        
    )

    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states_1'],
        config['num_actions_1'],
        config['training_epochs'],
        config['num_states_2'],
        config['num_actions_2']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time_1, training_time_2 = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time1:', training_time_1, 's-Training time2:', training_time_2, 's - Total:', round(simulation_time+training_time_1+training_time_2, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    Model.save_model_1(path)
    Model.save_model_2(path)

    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    Visualization.save_data_and_plot(data=Simulation.reward1_store, filename='reward1', xlabel='Episode', ylabel='Cumulative negative reward1')
    Visualization.save_data_and_plot(data=Simulation.reward2_store, filename='reward2', xlabel='Episode', ylabel='Cumulative negative reward2')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store_all, filename='delay_all', xlabel='Episode', ylabel='Cumulative delay of all (s)')
#    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store_bus_and_taxi, filename='delay_bus_and_taxi', xlabel='Episode', ylabel='Cumulative delay of bus and taxi (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')
    Visualization.save_data_and_plot(data=Simulation._co2_emission_store, filename ='CO2', xlabel='Episode', ylabel='CO2 emission (mg/s)')
    Visualization.save_data_and_plot(data=Simulation.mean_speed_store, filename ='mean_speed', xlabel='Episode', ylabel='Average mean speed (m/s)')
    Visualization.save_data_and_plot(data=Simulation.noise_emission_store, filename ='noise_emission', xlabel='Episode', ylabel='Noise emission (db)')