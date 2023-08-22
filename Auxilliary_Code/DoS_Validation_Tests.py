import get_data
import analyse_data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import time
import multiprocessing as mp
import warnings
from pathlib import Path


def run_dos_test(dos_pop_path, dos_results_path, n_processes, need_to_run_lifesim=True):
    """
    Runs the DoS test for the given population files
    :param dos_pop_path: Path to the folder containing the population files
    :param dos_results_path: Path to the folder where the results will be saved
    :param n_processes: Number of processes to run in parallel
    :param need_to_run_lifesim: Boolean to determine if the LIFEsim needs to be run first in order to generate the
    results or if you already have the results in the subdirectory from a previous simulation run.
    :return:
    """

    pythonpath = str(Path(__file__).parent.resolve().parent.joinpath("LIFEsim-Rick_Branch"))
    script_path = Path(__file__).parent.resolve().joinpath("LIFEsim_ExoSim_Inputs.py")

    if need_to_run_lifesim:
        # Get the path to every population file in the DoS_Test_Populations folder
        list_of_populations = os.listdir(dos_pop_path)
        populations_per_process = int(len(list_of_populations) / n_processes) + 1
        processes = []
        start_index = 0

        for i in range(n_processes):
            if i == n_processes - 1:
                end_index = -1
            else:
                end_index = start_index + populations_per_process

            process_populations = list_of_populations[start_index:end_index]
            # Run the LIFEsim for each population file
            process_results = ["sim_results/" + pop[:-5] + ".hdf5" for pop in process_populations]

            process = mp.Process(target=get_data.run_life_single,
                                 args=(pythonpath, script_path, dos_pop_path, process_populations, process_results))
            processes.append(process)

            start_index = end_index

        for process in processes:
            process.start()

        for process in processes:
            process.join()

    list_of_sims = os.listdir(dos_pop_path.joinpath("sim_results"))
    results_per_process = int(len(list_of_sims) / n_processes) + 1
    processes = []
    start_index = 0

    for i in range(n_processes):
        if i == n_processes - 1:
            end_index = start_index
        else:
            end_index = start_index + results_per_process

        sim_names = list_of_sims[start_index:end_index]

        process = mp.Process(target=analyse_data.analyse_one_dos, args=(dos_pop_path, sim_names, dos_results_path))
        processes.append(process)
        start_index = end_index

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    return None


if __name__ == '__main__':

    dos_results_path = Path(__file__).parent.parent.resolve().joinpath("Results/DoS_Stress_Test/")

    # Check if the DoS_Results folder exists, if not create it
    if not os.path.exists(dos_results_path):
        os.makedirs(dos_results_path)

    dos_pop_path = Path(__file__).parent.resolve().joinpath("Analysis/Populations/Dos_Test_Populations/")
    # Check if the DoS_Results folder exists, if not create it
    if not os.path.exists(dos_pop_path):
        os.makedirs(dos_pop_path)

    sim_results_path = Path(__file__).parent.resolve().joinpath("Analysis/Populations/Dos_Test_Populations/sim_results")
    # Check if the DoS_Results folder exists, if not create it
    if not os.path.exists(sim_results_path):
        os.makedirs(sim_results_path)

    run_dos_test(dos_pop_path, dos_results_path, 30, True)
