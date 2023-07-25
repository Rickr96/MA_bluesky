import os.path
import sys
import csv
import time
import pickle
import random
import pandas as pd
import glob
from pathlib import Path

parent_dir = Path(__file__).parents[1]
os.chdir(parent_dir.joinpath("EXOSIMS"))
sys.path.append(os.getcwd())

import EXOSIMS
import EXOSIMS.MissionSim
import EXOSIMS.util.read_ipcluster_ensemble
import EXOSIMS.SurveyEnsemble.IPClusterEnsemble as ipce


def ens_to_dict(ens, sim):
    """
    Converts the ensemble of DRMs to a dictionary
    :param ens: ensemble from exosim_run
    :return: dictionary of the results
    """
    cat = list(ens[0][0].keys())
    lists_dict = {name: [] for name in cat}
    lists_dict["N_sim"] = []
    lists_dict["Rp_found"] = []
    lists_dict["Mp_found"] = []
    lists_dict["a_found"] = []
    for i in range(0, len(ens)):
        result = ens[i]
        print("I am here ", i)
        for row in result:
            # print("I am here ", i, " ", row)
            lists_dict["N_sim"].append(i)
            lists_dict["Rp_found"].append(sim.SimulatedUniverse.Rp[row['plan_inds']])
            for name in cat:
                lists_dict[name].append(row[name])

    return lists_dict


def dict_to_csv(data, file_path):
    # Determine maximum length of any list in the dictionary
    max_length = max(len(lst) for lst in data.values())

    # Pad shorter lists with None up to the maximum length
    for key in data:
        if len(data[key]) < max_length:
            data[key].extend([None] * (max_length - len(data[key])))

    # Write CSV file
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data.keys())
        writer.writerows(zip(*data.values()))
    print("CSV file successfully created.")


def save_ppop_exosims(dict, outpath):
    """
    Saves the exoplanet population from the synthetic universes of the exosims run
    dict keys: 'a','e','I','O','w','M0','Mp','mu','Rp','p','plan2star','star'
    :param dict:
    :param outpath:
    :return:
    """
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(outpath + "/ppop_exosims.csv")

    return 0


def save_TargetList_exosims(dict, outpath):
    """
    Saves the target list from the synthetic universes of the exosims run

    :param dict:
    :param outpath:
    :return:
    """
    # remove I as its longer than the others
    dict.pop("I")
    df = pd.DataFrame.from_dict(dict)
    df.to_csv(outpath + "/TargetList_exosims.csv")

    return 0


def run_one(sim, outpath, genNewPlanets=True, rewindPlanets=True):
    sim.run_sim()
    DRM = sim.SurveySimulation.DRM[:]
    systems = sim.SurveySimulation.SimulatedUniverse.dump_systems()
    systems["MsTrue"] = sim.SurveySimulation.TargetList.MsTrue
    systems["MsEst"] = sim.SurveySimulation.TargetList.MsEst
    seed = sim.SurveySimulation.seed
    # reset simulation at the end of each simulation
    sim.SurveySimulation.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)

    pklname = (
            "run"
            + str(int(time.perf_counter() * 100))
            + "".join(["%s" % random.randint(0, 9) for num in range(5)])
            + ".pkl"
    )
    pklpath = os.path.join(outpath, pklname)
    with open(pklpath, "wb") as f:
        pickle.dump({"DRM": DRM, "systems": systems, "seed": seed}, f)

    return 0


def exosim_run(sim, outpath, N_sim=1, nprocess=1):
    """
    Runs the exosims simulation
    :param sim: EXOSIMS.MissionSim.MissionSim(scriptfile)
    :param N_sim: Number of simulations to run
    :param nprocess: Number of processes to run in parallel
    :return: Saves the results in a pickle file to the outpath given in run_one
    """
    # Clear the output directory of any previous runs
    file_list = glob.glob(os.path.join(outpath.joinpath("EXOSIMS"), "*.pkl"))
    for file_path in file_list:
        os.remove(file_path)

    if nprocess > 1:
        t1 = time.time()
        # TODO: Still not fucking working
        t2 = time.time()
        print("Time to run ", N_sim, " simulations with", nprocess, " Processes: ", t2 - t1, " seconds")
    elif nprocess == 1:
        t1 = time.time()
        for i in range(N_sim):
            print("start of simulation", (i+1), "/", N_sim)
            run_one(sim, outpath.joinpath("EXOSIMS"))
            print("end of simulation", (i+1), "/", N_sim)
        t2 = time.time()
        print("Time to run ", N_sim, " simulations: ", t2 - t1, " seconds")

    return 0


def __main__():
    # config Rick
    current_dir = Path(__file__).parent.resolve()
    scriptfile = current_dir.joinpath("Running_Sims/inputconfig.json")
    outpath = current_dir.joinpath("Analysis/Output")
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
    exosim_run(sim, outpath, N_sim=2, nprocess=1)
