import get_data as gd
import Run_ExoSims as rexo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.patches as mpatches
import time
import warnings
from pathlib import Path
from scipy.stats import gaussian_kde, iqr
import forecaster.mr_forecast as mr
import seaborn as sns

parent_dir = Path(__file__).parents[1]
os.chdir(parent_dir.joinpath("EXOSIMS"))
sys.path.append(os.getcwd())

import EXOSIMS
import EXOSIMS.MissionSim

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Important: All parameter space plots require some limits given. In order to have this the same everywhere and so that we
only need to change it at one place, all limits are defined here globally. 
"""

Rp_lim = [0, 20]  # Planet Radius [R_E]
Rp_lim_log = [-1, 1.5]  # Planet Radius [R_E]
Mp_lim = [0, 100]  # Planet Mass [M_E]
Mp_lim_log = [-1, 4]  # Planet Mass [M_E]
d_orbit_lim = [0, 15]  # Orbital Distance [AU]
d_orbit_lim_log = [-2, 2]  # Orbital Distance [AU]
d_orbit_scaled_lim = [0, 15]  # Orbital Distance [AU]
d_orbit_scaled_lim_log = [-2, 2]  # Orbital Distance [AU]
d_system_lim = [0, 30]  # System Distance [pc]
d_system_lim_log = [0, 1.5]  # System Distance [pc]


def get_whitelist_from_diff(scen1, scen2, parameter: str, gol: str):
    """
        :param scen1: Pandas DataFrame Scenario 1 that shall be compared to Scenario 2
        :param scen2: Pandas DataFrame Scenario 2 that shall be compared to Scenario 1
        :param parameter: String highlighting which parameter shall be compared
        :param gol: string either "gain" or "loss" whether indices of changes towards gain or loss should be detected
        :return: whitelist of indices in scen1 for which a difference between scen1 and scen2 could be detected
                  in the given parameter and based whether it was a gain or loss as stated through the gol string
        """
    para_list1_bool = scen1[parameter]
    para_list1 = [int(x) for x in para_list1_bool]
    para_list2_bool = scen2[parameter]
    para_list2 = [int(x) for x in para_list2_bool]
    whitelist = []

    # As Scen1 and Scen2 must be based on the same ppop catalog, they must be equally long
    if len(para_list1) != len(para_list2):
        print("ERROR Scenarios not based on same input catalogue!")
        sys.exit()
    if gol == 'gain':
        for idx, val in enumerate(para_list1):
            if val > para_list2[idx]:
                whitelist.append(idx)
            else:
                continue
    elif gol == 'loss':
        for idx, val in enumerate(para_list1):
            if val < para_list2[idx]:
                whitelist.append(idx)
            else:
                continue
    else:
        print("ERROR argument must be either 'gain' or 'loss' as a string!")
        sys.exit()

    return whitelist


def table_to_plot(final_table, result_path):
    """
    :param result_path: Path to the folder where results shall be saved
    :param final_table: A Pandas Dataframe containing the SNR results of the differenc scenarios and types
                        that the class should have considered. Generally a direct output of the function
                        scenario_snr_analysis
    :return: Excel output file, Plot Version of the Excel output file, histogram of the different SNR and detections
    """

    # Checking if Path exists, otherwise creating folder
    if not (os.path.exists(result_path)):
        os.makedirs(result_path)

    # Printing the Excel File
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(final_table)

    final_table.to_excel(result_path + "\output.xlsx")

    ############################################################
    # Preparing the Plot with X-Axis in split (sub-)categories #
    ############################################################

    # Getting Columns
    column_names = final_table.columns

    # Extract categories for x_axis of Histrograms and bar-plots
    categories = final_table[column_names[0]]
    ptypes = []
    stypes = []

    for cat_idx, cat in enumerate(categories):
        planet = cat[0]
        star = cat[1]
        ptypes.append(planet)
        stypes.append(star)
    ptypes_set = list(set(ptypes))
    stypes_set = list(set(stypes))
    ptypes_unique = sorted(ptypes_set, key=ptypes.index)  # Reordering correctly
    stypes_unique = sorted(stypes_set, key=stypes.index)  # Reordering correctly

    # Construction Data Dictionary
    dicp = dict()
    dics = dict()

    dicp_err = dict()
    dics_err = dict()

    dicp_det = dict()
    dics_det = dict()

    dicp_det_err = dict()
    dics_det_err = dict()
    for cat_idx in range(0, len(categories)):
        dicsc = dict()  # reset scenario dictionary
        dicsc_err = dict()
        dicsc_det = dict()
        dicsc_det_err = dict()
        for scen_idx in np.arange(1, len(column_names)):
            scen = column_names[scen_idx]  # getting Scenario Name as str
            val_line = final_table[scen][cat_idx]  # Getting Values from Scenario Value list
            SNR = float(val_line[0])  # as floats
            SNR_std = float(val_line[1])
            ndet = float(val_line[2])  # or ints
            ndet_err = float(val_line[3])  # or floats
            ntot = int(val_line[4])

            scen_short = scen[:2]  # shortening scen name for table
            dicsc[str(scen_short)] = SNR  # write SNR data into scenario dictionary
            dicsc_err[str(scen_short)] = SNR_std
            dicsc_det[str(scen_short)] = ndet
            dicsc_det_err[str(scen_short)] = ndet_err

        dics[stypes[cat_idx]] = dicsc  # fill solar dictionary with the scenario dictionaries
        dics_err[stypes[cat_idx]] = dicsc_err
        dics_det[stypes[cat_idx]] = dicsc_det
        dics_det_err[stypes[cat_idx]] = dicsc_det_err

        if cat_idx == (len(categories) - 1):
            dicp[ptypes[cat_idx]] = dics  # fill planetary dictionary with the solar dictionaries
            dicp_err[ptypes[cat_idx]] = dics_err
            dicp_det[ptypes[cat_idx]] = dics_det
            dicp_det_err[ptypes[cat_idx]] = dics_det_err

        # only when next planet type is different we want to write into planetary dictionary and reset solar dictionary
        elif ptypes[cat_idx] != ptypes[cat_idx + 1]:
            dicp[ptypes[cat_idx]] = dics
            dicp_err[ptypes[cat_idx]] = dics_err
            dicp_det[ptypes[cat_idx]] = dics_det
            dicp_det_err[ptypes[cat_idx]] = dics_det_err
            dics = dict()  # reset solar dictionary
            dics_err = dict()
            dics_det = dict()
            dics_det_err = dict()

    # Split the Dictionary into sub Dics such that the Plot is not too large
    n_graphs = 5
    for pix in range(0, n_graphs):
        # only 3 entries (hot, warm, cold) for each graph --> As we have 15 total cats, we split it into 5
        subdic = dict(list(dicp.items())[((pix * len(dicp)) // n_graphs):(((1 + pix) * len(dicp)) // n_graphs)])
        subdic_err = dict(
            list(dicp_err.items())[((pix * len(dicp_err)) // n_graphs):(((1 + pix) * len(dicp_err)) // n_graphs)])
        subdic_det = dict(
            list(dicp_det.items())[((pix * len(dicp_det)) // n_graphs):(((1 + pix) * len(dicp_det)) // n_graphs)])
        subdic_det_err = dict(
            list(dicp_det_err.items())[
            ((pix * len(dicp_det_err)) // n_graphs):(((1 + pix) * len(dicp_det_err)) // n_graphs)])

        # Plot the finished full Dictionary
        fig = plt.figure(figsize=(15, 10))

        ax2 = fig.add_subplot(1, 1, 1)
        color2 = 'g'
        label_group_bar(ax2, subdic_det, color2, subdic_det_err)
        ax2.set_ylabel('Number of Detections', fontsize=22)

        ax1 = ax2.twinx()
        color1 = 'mediumblue'
        label_group_point(ax1, subdic, color1, subdic_err)
        ax1.set_ylabel('SNR', fontsize=22)

        # Creating Manual Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        patch1 = mpatches.Patch(color=color1, label='SNR')
        patch2 = mpatches.Patch(color=color2, label='Detections')
        handles.extend([patch1, patch2])

        plt.legend(handles=handles, fontsize=22)

        plt.title("SNR and Detections of LIFEsim Run", fontsize=22)
        fig.subplots_adjust(bottom=0.3)
        savestring = result_path + "\\" + ptypes_unique[(pix * 3)] + ".pdf"  # *3 because hot, warm, cold
        fig.savefig(savestring)

        if pix >= n_graphs - 1:
            break

    return "Analysis Done"


def mk_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups


def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)


def label_group_bar(ax, data, color_c, data_errb=None):
    groups = mk_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)

    y_errb = 0
    if data_errb:
        groups_errb = mk_groups(data_errb)
        xy_errb = groups_errb.pop()
        x_errb, y_errb = zip(*xy_errb)

    ax.bar(xticks, y, yerr=y_errb, align='center', edgecolor='black',
           color=color_c, ecolor='red', capsize=4.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(x, fontsize=14)
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True, color=color_c)

    scale = 1. / ly
    for pos in range(ly + 1):  # change xrange to range for python3
        add_line(ax, pos * scale, -.1)
    ypos = -.2
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes, fontsize=14)
            add_line(ax, pos * scale, ypos)
            pos += rpos
        add_line(ax, pos * scale, ypos)
        ypos -= .1


def label_group_point(ax, data, color_c, data_err=None):
    groups = mk_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)

    y_err = 0
    if data_err:
        groups_err = mk_groups(data_err)
        xy_err = groups_err.pop()
        x_err, y_err = zip(*xy_err)

    ax.errorbar(xticks, y, yerr=y_err, fmt="o", ms=10,
                color=color_c, ecolor=color_c, capsize=4.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(x, fontsize=14)
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True, color=color_c)

    scale = 1. / ly
    for pos in range(ly + 1):  # change xrange to range for python3
        add_line(ax, pos * scale, -.1)
    ypos = -.2
    while groups:
        group = groups.pop()
        pos = 0
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes, fontsize=14)
            add_line(ax, pos * scale, ypos)
            pos += rpos
        add_line(ax, pos * scale, ypos)
        ypos -= .1


def stype_translator(stype: str):
    if stype == 'A':
        stype_val = 0
    elif stype == 'F':
        stype_val = 1
    elif stype == 'G':
        stype_val = 2
    elif stype == 'K':
        stype_val = 3
    elif stype == 'M':
        stype_val = 4

    return stype_val


def add_ptype_to_df(df, ptypes):
    """
    Adds a column with the planet type to the dataframe
        Shall provide a split of the complete data frame into
        distinct categories of planets that still cover the entire population. Defined after
        Kopperapu et al. 2015:
                hot_rocky  --  182 F_E < F_p <= 1 F_E; R_p <= 1 R_E
                warm_rocky --  1 F_E < F_p <= 0.28 F_E; R_p <= 1 R_E
                cold_rocky --  0.28 F_E < F_p <= 0.0035 F_E; R_p <= 1 R_E
                hot_SE   --  188 F_E < F_p <= 1.15 F_E; 1 R_E < R_p <= 1.75 R_E
                warm_SE  --  1.15 F_E < F_p <= 0.32 F_E; 1 R_E < R_p <= 1.75 R_E
                cold_SE --  0.32 F_E < F_p <= 0.0030 F_E; 1 R_E < R_p <= 1.75 R_E
                hot_sub-Neptunes   --  220 F_E < F_p <= 1.65 F_E; 1.75 R_E < R_p <= 3.5 R_E
                warm_sub-Neptunes  --  1.65 F_E < F_p <= 0.45 F_E; 1.75 R_E < R_p <= 3.5 R_E
                cold_sub-Neptunes  --  0.45 F_E < F_p <= 0.0030 F_E; 1.75 R_E < R_p <= 3.5 R_E
                hot_sub-Jovians    --  220 F_E < F_p <= 1.65 F_E; 3.5 R_E < R_p <= 6 R_E
                warm_sub-Jovians   --  1.65 F_E < F_p <= 0.40 F_E; 3.5 R_E < R_p <= 6 R_E
                cold_sub-Jovians   --  0.40 F_E < F_p <= 0.0025 F_E; 3.5 R_E < R_p <= 6 R_E
                hot_Jovians        --  220 F_E < F_p <= 1.70 F_E; 6 R_E < R_p <= 14.3 R_E
                warm_Jovians       --  1.70 F_E < F_p <= 0.45 F_E; 6 R_E < R_p <= 14.3 R_E
                cold_Jovians       --  0.45 F_E < F_p <= 0.0025 F_E; 6 R_E < R_p <= 14.3 R_E
    :param df: Dataframe from EXOsim or LIFEsim
    :param ptype: list of strings highlighting what ptypes we want to add
    :return: Dataframe with added column
    """
    print("Started adding ptypes to DF of length: " + str(len(df)))
    t1 = time.time()
    df_ptypes = pd.DataFrame(columns=['ptype'])
    if ptypes == 'Kop2015':
        for i in range(len(df)):
            if df['radius_p'][i] < 1:
                if df['Fp'][i] > 1:
                    df_ptypes.loc[i] = 'Hot Rocky'
                elif df['Fp'][i] > 0.28:
                    df_ptypes.loc[i] = 'Warm Rocky'
                else:
                    df_ptypes.loc[i] = 'Cold Rocky'
            elif df['radius_p'][i] < 1.75:
                if df['Fp'][i] > 1.15:
                    df_ptypes.loc[i] = 'Hot SE'
                elif df['Fp'][i] > 0.32:
                    df_ptypes.loc[i] = 'Warm SE'
                else:
                    df_ptypes.loc[i] = 'Cold SE'
            elif df['radius_p'][i] < 3.5:
                if df['Fp'][i] > 1.65:
                    df_ptypes.loc[i] = 'Hot Sub-Neptune'
                elif df['Fp'][i] > 0.45:
                    df_ptypes.loc[i] = 'Warm Sub-Neptune'
                else:
                    df_ptypes.loc[i] = 'Cold Sub-Neptune'
            elif df['radius_p'][i] < 6:
                if df['Fp'][i] > 1.65:
                    df_ptypes.loc[i] = 'Hot Sub-Jovian'
                elif df['Fp'][i] > 0.40:
                    df_ptypes.loc[i] = 'Warm Sub_Jovian'
                else:
                    df_ptypes.loc[i] = 'Cold Sub-Jovian'
            else:
                if df['Fp'][i] > 1.70:
                    df_ptypes.loc[i] = 'Hot Jovian'
                elif df['Fp'][i] > 0.45:
                    df_ptypes.loc[i] = 'Warm Jovian'
                else:
                    df_ptypes.loc[i] = 'Cold Jovian'
    else:
        print("ERROR: ptypes not recognized")
    df = pd.concat([df, df_ptypes], axis=1)
    t2 = time.time()
    print("Ended adding ptypes to DF after: " + str(t2 - t1) + " seconds")

    return df


def add_HZ_to_exo(dfexo, dflife):
    """
    Adds the HZ to the EXOsim dataframe read from the already calced HZ from the LIFEsim dataframe
    :param dfexo: EXOsim dataframe
    :param dflife: LIFEsim dataframe
    :return:
    """
    print("Started adding habitability data to DF of length: " + str(len(dfexo)))
    t1 = time.time()
    unique_snames_exo = dfexo['sname'].unique()
    unique_snames_life = dflife['name_s'].unique()

    for sname in unique_snames_exo:
        if sname in unique_snames_life:
            hz_in = dflife.loc[dflife['name_s'] == sname, 'hz_in'].iloc[0]
            hz_out = dflife.loc[dflife['name_s'] == sname, 'hz_out'].iloc[0]

            # Update dfexo with the HZ-in and HZ-out values for the current 'sname'
            dfexo.loc[dfexo['sname'] == sname, 'hz_in'] = hz_in
            dfexo.loc[dfexo['sname'] == sname, 'hz_out'] = hz_out

    # Add habitability boolean based on HZ data and orbit distance
    dfexo['habitable'] = (dfexo['semimajor_p'] > dfexo['hz_in']) & (dfexo['semimajor_p'] < dfexo['hz_out'])

    t2 = time.time()
    print("Finished adding habitability data. It took " + str(t2 - t1) + " seconds.")

    return dfexo


def detection_statistics(df, first_param, first_param_str, second_param, second_param_str):
    """
    Calculates the detection statistics for a given planet type and star type accounting for the "different" statistics
    of the binary nature of detections to sum over detections per universe and only then take the mean.
    :param df: dataframe either from EXOsim or LIFEsim
    :param first_param: first parameter over which we want the statistics e.g. planet type
    :param second_param: second parameter over which we want the statistics e.g.stellar type
    :param first_param_str: string first parameter over which we take the statistics
    :param second_param_str: string second parameter over which we take the statistics
    :return: mean and standard deviation over all universe realizations for the given planet and stellar type of the
    detections of that planet type around that stellar type.
    """
    Nuni_max = max(df["nuniverse"])
    det_list = []
    for nuni in np.arange(Nuni_max + 1):
        if second_param_str == 'stype':
            mask = (df[first_param_str] == first_param) \
                   & (df[second_param_str].str[0] == second_param) \
                   & (df['nuniverse'] == nuni)
        else:
            mask = (df[first_param_str] == first_param) \
                   & (df[second_param_str] == second_param) \
                   & (df['nuniverse'] == nuni)

        det_uni_data = (df[mask]['detected'])
        det_uni = np.sum(det_uni_data * (det_uni_data == 1))
        det_list.append(det_uni)
    det_array = np.array(det_list)
    mean = det_array.mean()
    std_dev = det_array.std()
    mean = round(mean, ndigits=2)
    std_dev = round(std_dev, ndigits=2)

    return mean, std_dev


def bar_cat_plot(df_life, df_exo, first_params, first_param_str, second_params, second_param_str,
                 save=False, result_path=None, name="bar_cat_plot"):
    """
    Plots the SNR and detections like I did in the Semester Thesis
    :param name: name of the plot
    :param result_path: path to save the plot
    :param save: boolean if plot should be saved
    :param first_params: list of strings highlighting what parameters need to be plotted
    :param first_param_str: string first parameter over which we take the statistics
    :param second_params: list of strings highlighting what parameters need to be plotted
    :param second_param_str: string second parameter over which we take the statistics
    :param df_life: dataframe with LIFE data
    :param df_exo: dataframe with Exo data
    :return: prints (and saves) the plot
    """
    new_dir = result_path.joinpath("Bar_Cat_Plots")

    if not os.path.exists(new_dir):
        # Create the "Radius_Mass_Doublecheck" directory if it doesn't exist
        os.makedirs(new_dir)

    # Update results_path to point to the "Radius_Mass_Doublecheck" directory
    result_path = new_dir

    t1 = time.time()

    # Bring Data in wished Format
    life_table, exo_table, cat = [], [], []
    for first_param in first_params:
        for second_param in second_params:
            # LIFE Data
            life_mask = (df_life[first_param_str] == first_param) \
                        & (df_life[second_param_str] == second_param) \
                        & (df_life["detected"] == True)
            life_dettime = df_life[life_mask]['int_time']
            life_dettime /= 3600  # convert from seconds to hours
            mean_dettime_life = life_dettime.mean()
            if len(life_dettime) > 0:
                life_percentile_10 = np.percentile(life_dettime, 33)
                life_percentile_90 = np.percentile(life_dettime, 67)
                std_dettime_life = (life_percentile_90 - life_percentile_10) / 2
            else:
                std_dettime_life = 0
            mean_det_life, std_det_life = detection_statistics(df_life, first_param, first_param_str,
                                                               second_param, second_param_str)
            total_planets_life = len(df_life[life_mask])

            ################
            # EXOsim Data #
            ################

            # EXOsim SNR is not broken down into SNR_1h. EXOsim time  in unit of days, SNR scales with sqrt(t) !
            if second_param_str == "stype":
                exo_mask = (df_exo[first_param_str] == first_param) \
                           & (df_exo[second_param_str].str[0] == second_param) \
                           & (df_exo["detected"] == 1)
            else:
                exo_mask = (df_exo[first_param_str] == first_param) \
                           & (df_exo[second_param_str] == second_param) \
                           & (df_exo["detected"] == 1)
            exo_dettime = df_exo[exo_mask]['dettime']
            exo_dettime *= 24  # convert from days to hours
            mean_dettime_exo = exo_dettime.mean()
            if len(exo_dettime) > 1:
                exo_percentile_10 = np.percentile(exo_dettime, 33)
                exo_percentile_90 = np.percentile(exo_dettime, 67)
                std_dettime_exo = (exo_percentile_90 - exo_percentile_10) / 2
            else:
                std_dettime_exo = 0
            if np.isnan(mean_dettime_exo):
                mean_dettime_exo = 0
                std_dettime_exo = 0
            if np.isnan(mean_dettime_life):
                mean_dettime_life = 0
                std_dettime_life = 0
            mean_det_exo, std_det_exo = detection_statistics(df_exo, first_param, first_param_str,
                                                             second_param, second_param_str)
            total_planets_exo = len(df_exo[exo_mask])

            # Category
            cat.append([first_param, second_param])
            # Saving it together
            life_table.append([mean_dettime_life, std_dettime_life, mean_det_life, std_det_life, total_planets_life])
            exo_table.append([mean_dettime_exo, std_dettime_exo, mean_det_exo, std_det_exo, total_planets_exo])

    final_table = pd.DataFrame(columns=['Category', 'LIFE', 'Exo'])
    value_table = pd.DataFrame({'Category': cat,
                                'LIFE': life_table,
                                'Exo': exo_table})
    final_table = pd.concat([final_table, value_table], ignore_index=True)
    final_table.to_csv(result_path.joinpath(name + ".csv"), index=False)
    column_names = final_table.columns
    t2 = time.time()
    print("Bar Cat Plot: It took " + str(t2 - t1) + " seconds to bring the data in the right format")

    # Extract categories for x_axis of Histrograms and bar-plots
    categories = final_table[column_names[0]]
    ptypes_cat = []
    stypes_cat = []
    for cat_idx, cat in enumerate(categories):
        planet = cat[0]
        star = cat[1]
        ptypes_cat.append(planet)
        stypes_cat.append(star)

    # Construction Data Dictionary
    dicp = dict()
    dics = dict()

    dicp_err = dict()
    dics_err = dict()

    dicp_det = dict()
    dics_det = dict()

    dicp_det_err = dict()
    dics_det_err = dict()
    for cat_idx in range(0, len(categories)):
        dicsc = dict()  # reset scenario dictionary
        dicsc_err = dict()
        dicsc_det = dict()
        dicsc_det_err = dict()
        for scen_idx in np.arange(1, len(column_names)):
            scen = column_names[scen_idx]  # getting Scenario Name as str
            val_line = final_table[scen][cat_idx]  # Getting Values from Scenario Value list
            SNR = float(val_line[0])  # as floats
            SNR_std = float(val_line[1])
            ndet = float(val_line[2])  # or ints
            ndet_err = float(val_line[3])  # or floats

            scen_short = scen[:2]  # shortening scen name for table
            dicsc[str(scen_short)] = SNR  # write SNR data into scenario dictionary
            dicsc_err[str(scen_short)] = SNR_std
            dicsc_det[str(scen_short)] = ndet
            dicsc_det_err[str(scen_short)] = ndet_err

        dics[stypes_cat[cat_idx]] = dicsc  # fill solar dictionary with the scenario dictionaries
        dics_err[stypes_cat[cat_idx]] = dicsc_err
        dics_det[stypes_cat[cat_idx]] = dicsc_det
        dics_det_err[stypes_cat[cat_idx]] = dicsc_det_err

        if cat_idx == (len(categories) - 1):
            dicp[ptypes_cat[cat_idx]] = dics  # fill planetary dictionary with the solar dictionaries
            dicp_err[ptypes_cat[cat_idx]] = dics_err
            dicp_det[ptypes_cat[cat_idx]] = dics_det
            dicp_det_err[ptypes_cat[cat_idx]] = dics_det_err

        # only when next planet type is different we want to write into planetary dictionary and reset solar dictionary
        elif ptypes_cat[cat_idx] != ptypes_cat[cat_idx + 1]:
            dicp[ptypes_cat[cat_idx]] = dics
            dicp_err[ptypes_cat[cat_idx]] = dics_err
            dicp_det[ptypes_cat[cat_idx]] = dics_det
            dicp_det_err[ptypes_cat[cat_idx]] = dics_det_err
            dics = dict()  # reset solar dictionary
            dics_err = dict()
            dics_det = dict()
            dics_det_err = dict()

    if first_param_str == 'ptype':
        # Split the Dictionary into sub Dics such that the Plot is not too large
        n_graphs = 5
        for pix in range(0, n_graphs):
            # only 3 entries (hot, warm, cold) for each graph --> As we have 15 total cats, we split it into 5
            subdic = dict(list(dicp.items())[((pix * len(dicp)) // n_graphs):(((1 + pix) * len(dicp)) // n_graphs)])
            subdic_err = dict(
                list(dicp_err.items())[((pix * len(dicp_err)) // n_graphs):(((1 + pix) * len(dicp_err)) // n_graphs)])
            subdic_det = dict(
                list(dicp_det.items())[((pix * len(dicp_det)) // n_graphs):(((1 + pix) * len(dicp_det)) // n_graphs)])
            subdic_det_err = dict(
                list(dicp_det_err.items())[
                ((pix * len(dicp_det_err)) // n_graphs):(((1 + pix) * len(dicp_det_err)) // n_graphs)])

            # Plot the finished full Dictionary
            fig = plt.figure(figsize=(15, 10))

            ax2 = fig.add_subplot(1, 1, 1)
            color2 = 'g'
            label_group_bar(ax2, subdic_det, color2, subdic_det_err)
            ax2.set_ylabel('Number of Detections', fontsize=22)

            ax1 = ax2.twinx()
            color1 = 'mediumblue'
            label_group_point(ax1, subdic, color1, subdic_err)
            ax1.set_ylabel('Detection Time [h]', fontsize=22)

            # Creating Manual Legend
            handles, labels = plt.gca().get_legend_handles_labels()
            patch1 = mpatches.Patch(color=color1, label='SNR')
            patch2 = mpatches.Patch(color=color2, label='Detections')
            handles.extend([patch1, patch2])

            plt.legend(handles=handles, fontsize=22)

            plt.title("Detections and Detection Time of LIFEsim and EXOsim split by planetary type", fontsize=18)
            fig.subplots_adjust(bottom=0.3)
            if save:
                savestring = result_path.joinpath(
                    name + "_" + first_params[(pix * 3)] + ".pdf")  # *3 because hot, warm, cold
                fig.savefig(savestring)
            plt.clf()
            if pix >= n_graphs - 1:
                break
    else:
        # Split the Dictionary into sub Dics such that the Plot is not too large
        n_graphs = 1
        for pix in range(0, n_graphs):
            # only 3 entries (hot, warm, cold) for each graph --> As we have 15 total cats, we split it into 5
            subdic = dict(list(dicp.items())[((pix * len(dicp)) // n_graphs):(((1 + pix) * len(dicp)) // n_graphs)])
            subdic_err = dict(
                list(dicp_err.items())[((pix * len(dicp_err)) // n_graphs):(((1 + pix) * len(dicp_err)) // n_graphs)])
            subdic_det = dict(
                list(dicp_det.items())[((pix * len(dicp_det)) // n_graphs):(((1 + pix) * len(dicp_det)) // n_graphs)])
            subdic_det_err = dict(
                list(dicp_det_err.items())[
                ((pix * len(dicp_det_err)) // n_graphs):(((1 + pix) * len(dicp_det_err)) // n_graphs)])

            # Plot the finished full Dictionary
            fig = plt.figure(figsize=(15, 10))

            ax2 = fig.add_subplot(1, 1, 1)
            color2 = 'g'
            label_group_bar(ax2, subdic_det, color2, subdic_det_err)
            ax2.set_ylabel('Number of Detections', fontsize=22)

            ax1 = ax2.twinx()
            color1 = 'mediumblue'
            label_group_point(ax1, subdic, color1, subdic_err)
            ax1.set_ylabel('Integration Time [h]', fontsize=22)

            # Creating Manual Legend
            handles, labels = plt.gca().get_legend_handles_labels()
            patch1 = mpatches.Patch(color=color1, label='Integration Time')
            patch2 = mpatches.Patch(color=color2, label='# Detections')
            handles.extend([patch1, patch2])

            plt.legend(handles=handles, fontsize=22)

            plt.title("Integration Time and Detections of LIFEsim and EXOsim", fontsize=22)
            fig.subplots_adjust(bottom=0.3)
            if save:
                savestring = result_path.joinpath(name + ".pdf")
                fig.savefig(savestring)
            plt.clf()
            if pix >= n_graphs - 1:
                break
    return None


def kde_distr_plot(life_data_d, exo_data_d, sample_data_d, result_path, name, xlabel, xlim,
                   detected=False, draw_HZ=False, HZ_inner=None, HZ_outer=None, log=False):
    """
    """
    new_dir = result_path.joinpath("1D-Distributions")

    if not os.path.exists(new_dir):
        # Create the "Radius_Mass_Doublecheck" directory if it doesn't exist
        os.makedirs(new_dir)

    result_path = new_dir

    # Rename columns ready for df to be merged for seaborn
    life_data_d_new, exo_data_d_new = life_data_d.to_frame(), exo_data_d.to_frame()
    sample_data_d_new = sample_data_d.to_frame()
    life_col_name, exo_col_name = life_data_d_new.columns[0], exo_data_d_new.columns[0]
    sample_col_name = sample_data_d_new.columns[0]
    # Merge the Dataframes for Seaborn to work with it
    data = pd.concat([life_data_d_new.rename(columns={life_col_name: xlabel}),
                      exo_data_d_new.rename(columns={exo_col_name: xlabel}),
                      sample_data_d_new.rename(columns={sample_col_name: xlabel})],
                     keys=['LIFEsim', 'EXOSIMS', 'Sample'], names=['Simulation'])

    fig = plt.figure(figsize=(8, 7))

    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1])
    ax_kde = plt.subplot(gs[0])

    g = sns.kdeplot(data, hue="Simulation", x=xlabel, palette='viridis', fill=True, alpha=.5, multiple='layer',
                    common_norm=False, cut=0.1, bw_adjust=0.25, legend=True)

    g.set_xlim(xlim)

    # Vertical Lines with location of HZ and the mean of the data
    vert_lines = []
    if log:
        life_mean = np.log10((10 ** life_data_d).mean())
        exo_mean = np.log10((10 ** exo_data_d).mean())
    else:
        life_mean = float(life_data_d.mean())
        exo_mean = float(exo_data_d.mean())

    vert_lines.append(life_mean)
    vert_lines.append(exo_mean)

    if draw_HZ:
        vert_lines.append(HZ_inner)
        vert_lines.append(HZ_outer)

    for i, line in enumerate(vert_lines):
        if i == 0:
            plt.axvline(x=line, color='r', linestyle='dashed', linewidth=2)
        elif i == 1:
            plt.axvline(x=line, color='r', linestyle='dotted', linewidth=2)
        else:
            plt.axvline(x=line, color='k', linewidth=2)

    ax_textbox = plt.subplot(gs[1])
    ax_textbox.axis('off')

    if detected:
        text_string_life = "#Detections LIFEsim: " + str(len(life_data_d))
        text_string_exo = "#Detections EXOSIMS: " + str(len(exo_data_d))
        textboxtext = text_string_life + "\n" + text_string_exo
        ax_textbox.text(0.5, -0.1, textboxtext, bbox={'alpha': 0.5, 'pad': 10}, ha='center', va='top')
    plt.savefig(result_path.joinpath(name + '.svg'))

    plt.suptitle(name, fontsize=16)
    plt.savefig(result_path.joinpath(name + '.svg'))

    # Clear Figure so it does not interfere with other plots
    plt.clf()

    return 0


def histogram_distribution_plot(life_data_d, exo_data_d, result_path, name, xlabel, xlim=None, ylim=None,
                                detected=False, draw_HZ=False, HZ_inner=None, HZ_outer=None, log=False):
    """
    Plots the distribution of the given data from life and EXOsim into a histogram plot showing the
    distribution of the arbitrary value given as input. As LIFEsim and EXOsim data do not have the same total
    underlying amount of planets, the data is normalized to the total amount of planets in the respective data set.
    Additionally, the boolean parameter 'detected' can be set to True, in which case the additional information of the
    total amount of detected planets is added to the plot.
    :param life_data_d: Dataframe containing the LIFEsim data
    :param exo_data_d: Dataframe containing the EXOsim data
    :param result_path: Path where the results should be saved
    :param name: name of the plot
    :param xlabel: label of the x-axis
    :param xlim: limits of the x-axis
    :param ylim: limits of the y-axis
    :param detected: boolean parameter if True additional total amount of detections is added to the plot
    :param draw_HZ: boolean parameter if True the habitable zone is added to the plot
    :param HZ_inner: inner edge of the habitable zone
    :param HZ_outer: outer edge of the habitable zone
    :param log: boolean parameter if True the data is logarithmic and the mean calculations needs to be adjusted
                accordingly
    :return: histogram plot showing the distribution of any value given as input
    """
    new_dir = result_path.joinpath("Histograms")

    if not os.path.exists(new_dir):
        # Create the "Radius_Mass_Doublecheck" directory if it doesn't exist
        os.makedirs(new_dir)

    # Update results_path to point to the "Radius_Mass_Doublecheck" directory
    result_path = new_dir

    life_data_original = life_data_d
    exo_data_original = exo_data_d

    if xlim is not None:
        life_data_d = life_data_d[life_data_d <= xlim[1]]
        exo_data_d = exo_data_d[exo_data_d <= xlim[1]]
        life_data_d = life_data_d[life_data_d >= xlim[0]]
        exo_data_d = exo_data_d[exo_data_d >= xlim[0]]

    fig, axs = plt.subplots(1, 2, sharey=True)
    hist_val0, bin_edges0, _ = axs[0].hist(life_data_d.astype(float), alpha=0.9, bins=25, color='green', density=True,
                                           label='LIFEsim')
    hist_val1, bin_edges1, _ = axs[1].hist(exo_data_d.astype(float), alpha=0.9, bins=25, color='blue', density=True,
                                           label='EXOsim')

    fig.text(0.5, 0.04, xlabel, ha="center", va="center", fontsize=10)
    axs[0].set_ylabel('probability density function', fontsize=10, labelpad=10)

    # Calculating the location of the mean -- Calculation different for log data as mean(log(x)) != log(mean(x))
    if log:
        axs[0].axvline(np.log10((10 ** life_data_d).mean()), color='r', linewidth=1)
        axs[1].axvline(np.log10((10 ** exo_data_d).mean()), color='r', linewidth=1)
    else:
        axs[0].axvline(life_data_d.mean(), color='r', linewidth=1)
        axs[1].axvline(exo_data_d.mean(), color='r', linewidth=1)

    if draw_HZ:
        axs[0].axvline(HZ_inner, color='k', linewidth=2)
        axs[0].axvline(HZ_outer, color='k', linewidth=2)
        axs[1].axvline(HZ_inner, color='k', linewidth=2)
        axs[1].axvline(HZ_outer, color='k', linewidth=2)

    # Pre-Amble to scale the y-axis accordingly
    # Find the maximum density value
    max_density0 = np.max(hist_val0)
    max_density1 = np.max(hist_val1)
    max_density = max(max_density0, max_density1)

    # Determine the maximum y-value with some spacing
    max_y = max_density * 1.1

    # Translate Nan to 0
    max_y = 0 if np.isnan(max_y) else max_y

    # Calculate the number of dashed lines needed
    num_lines = int(max_y / 0.1)

    for i in range(1, num_lines + 1):
        axs[0].axhline(0.1 * i, color='k', linestyle='dashed', linewidth=1)
        axs[1].axhline(0.1 * i, color='k', linestyle='dashed', linewidth=1)

    axs[0].set_title('LIFEsim', fontsize=12)
    axs[1].set_title('EXOsim', fontsize=12)

    if xlim is not None:
        axs[0].set_xlim(left=xlim[0], right=xlim[1])
        axs[1].set_xlim(left=xlim[0], right=xlim[1])
    else:
        axs[0].set_xlim(left=0)
        axs[1].set_xlim(left=0)
    if ylim is not None:
        axs[0].set_ylim(bottom=ylim[0], top=ylim[1])
        axs[1].set_ylim(bottom=ylim[0], top=ylim[1])
    else:
        axs[0].set_ylim(bottom=0, top=max_y)
        axs[1].set_ylim(bottom=0, top=max_y)

    # Add marker of Earth Size
    xticks0 = axs[0].get_xticks().tolist()
    xticks0.append(1.0)
    axs[0].set_xticks(sorted(xticks0))
    xticks1 = axs[1].get_xticks().tolist()
    xticks1.append(1.0)
    axs[1].set_xticks(sorted(xticks1))

    fig.suptitle(name, fontsize=16)
    if detected:
        text_string_life = "#Detections: " + str(len(life_data_original))
        text_string_exo = "#Detections: " + str(len(exo_data_original))
        axs[0].text(0.9, 0.95, text_string_life, bbox={'alpha': 0.5, 'pad': 10}, transform=axs[0].transAxes,
                    ha='right', va='top')
        axs[1].text(0.9, 0.95, text_string_exo, bbox={'alpha': 0.5, 'pad': 10}, transform=axs[1].transAxes, ha='right',
                    va='top')
    plt.savefig(result_path.joinpath(name + '.svg'))

    # Clear Figure so it does not interfere with other plots
    plt.clf()

    return 0


def histogram_single(life_data_d, result_path, name, xlabel, xlim=None, ylim=None, log=False):
    """
        Plots the distribution of the given data from lifesim as a showcase of the underlying sampled population
        :param life_data_d: Dataframe containing the LIFEsim data
        :param result_path: Path where the results should be saved
        :param name: name of the plot
        :param xlabel: label of the x-axis
        :param xlim: limits of the x-axis
        :param ylim: limits of the y-axis
        :param log: boolean parameter if True the data is logarithmic and the mean calculations needs to be adjusted
        :return: histogram plot showing the distribution of any value given as input
        """
    new_dir = result_path.joinpath("Histograms")

    if not os.path.exists(new_dir):
        # Create the "Radius_Mass_Doublecheck" directory if it doesn't exist
        os.makedirs(new_dir)

    # Update results_path to point to the "Radius_Mass_Doublecheck" directory
    result_path = new_dir

    life_data_original = life_data_d

    if xlim is not None:
        life_data_d = life_data_d[life_data_d <= xlim[1]]
        life_data_d = life_data_d[life_data_d >= xlim[0]]

    fig = plt.figure()
    hist_val, bin_edges, _ = plt.hist(life_data_d.astype(float), alpha=0.9, bins=25, color='green', density=True)

    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel('probability density function', fontsize=10, labelpad=10)

    if log:
        plt.axvline(np.log10((10 ** life_data_d).mean()), color='r', linewidth=1)
    else:
        plt.axvline(life_data_d.mean(), color='r', linewidth=1)

    # Pre-Amble to scale the y-axis accordingly
    # Find the maximum density value
    max_density = np.max(hist_val)
    # Determine the maximum y-value with some spacing
    max_y = max_density * 1.1
    # Translate Nan to 0
    max_y = 0 if np.isnan(max_y) else max_y
    # Calculate the number of dashed lines needed
    num_lines = int(max_y / 0.1)

    for i in range(1, num_lines + 1):
        plt.axhline(0.1 * i, color='k', linestyle='dashed', linewidth=1)

    if xlim is not None:
        plt.xlim(left=xlim[0], right=xlim[1])
    else:
        plt.xlim(left=0)
    if ylim is not None:
        plt.ylim(bottom=ylim[0], top=ylim[1])
    else:
        plt.ylim(bottom=0, top=max_y)

    # Add marker of Earth Size
    xticks = plt.xticks()[0].tolist()
    xticks.append(1.0)
    plt.xticks(sorted(xticks))

    fig.suptitle(name, fontsize=16)
    plt.savefig(result_path.joinpath(name + '.svg'))

    # Clear Figure so it does not interfere with other plots
    plt.clf()

    return None


def kde_distribution_plot(data_d1, data_d2, result_path, name, xlabel, ylabel, xlim=None, ylim=None, detected=False):
    """
    Plots the kernel density estimator of the given data from life and EXOsim showing the
    distribution of the arbitrary value given as input. As LIFEsim and EXOsim data do not have the same total
    underlying amount of planets, the data is normalized to the total amount of planets in the respective data set.
    Additionally, the boolean parameter 'detected' can be set to True, in which case the additional information of the
    total amount of detected planets is added to the plot.
    :param data_d1: Dataframe containing the first parameter
    :param data_d2: Dataframe containing the second parameter
    :param xlim: x-axis limits
    :param ylim: y-axis limits
    :param result_path: Path where the results should be saved
    :param name: name of the plot
    :param xlabel: label of the x-axis
    :param ylabel: label of the y-axis
    :param detected: boolean parameter if True additional total amount of detections is added to the plot
    :return: histogram plot showing the distribution of any value given as input
    """

    new_dir = result_path.joinpath("KDE_Distributions")

    if not os.path.exists(new_dir):
        # Create the "Radius_Mass_Doublecheck" directory if it doesn't exist
        os.makedirs(new_dir)

    # Update results_path to point to the "Radius_Mass_Doublecheck" directory
    result_path = new_dir

    if xlim is None:
        xlim = [data_d1.min(), data_d1.max()]
    if ylim is None:
        ylim = [data_d2.min(), data_d2.max()]

    # Plot KDE Plot
    kde = gaussian_kde([data_d1, data_d2])
    xmin, xmax = xlim[0], xlim[1]
    ymin, ymax = ylim[0], ylim[1]
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    # Preamble for the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)
    # Plot the KDE plot
    plt.imshow(np.rot90(Z), extent=[xmin, xmax, ymin, ymax], aspect='auto', cmap='hot')
    plt.colorbar(label='Density')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)

    if detected:
        text_string = "#Detections: " + str(len(data_d1))
        ax.text(0.9, 0.95, text_string, bbox={'alpha': 0.5, 'pad': 10}, transform=ax.transAxes,
                ha='right', va='top')
    plt.savefig(result_path.joinpath(name + '.svg'))

    # Clear Figure so it does not interfere with other plots
    plt.clf()

    return 0


def run_it_and_save_it(results_path, modes=None):
    # Run EXOsim in the given config as highlighted in Run_EXOsim.py and the input config file
    rexo.__main__()
    # Get the produced EXOsim data, convert it to LIFEsim and run LIFEsim with that according to the get_data.py code
    gd.__main__(ppop_path_gd=None, life_results_path_gd=None, modes=modes)

    current_dir = Path(__file__).parent.resolve()
    exo_output_path = current_dir.joinpath("Analysis/Output/EXOSIMS")
    life_output_path = current_dir.joinpath("Analysis/Output/LIFEsim")
    stellar_cat_path = current_dir.joinpath("Analysis/Populations/TargetList_exosims.csv")

    # Import DataFrames
    if modes is None:
        modes = ["demo1"]
    for i in range(len(modes)):
        life_data_nop, exo_data_nop = gd.import_data(exo_output_path, life_output_path, modes[i] + ".hdf5",
                                                     stellar_cat_path)

        # Add Planet Category
        life_data = add_ptype_to_df(life_data_nop, "Kop2015")
        exo_data_noHZ = add_ptype_to_df(exo_data_nop, "Kop2015")
        print("Length before adding HZ:", len(exo_data_noHZ))
        # Add HZ of LIFesim to EXOsim Data
        exo_data = add_HZ_to_exo(exo_data_noHZ, life_data)
        print("Length After adding HZ:", len(exo_data))

        # Save DataFrames
        life_data.to_csv(results_path.joinpath(modes[i] + "_life_data.csv"))
        exo_data.to_csv(results_path.joinpath(modes[i] + "_exo_data.csv"))

    return None


def plots(life_data, exo_data, life_data_det, exo_data_det, results_path):
    # Planet and Stellar categories considered
    ptypes = ["Hot Rocky", "Warm Rocky", "Cold Rocky",
              "Hot SE", "Warm SE", "Cold SE",
              "Hot Sub-Neptune", "Warm Sub-Neptune", "Cold Sub-Neptune",
              "Hot Sub-Jovian", "Warm Sub-Jovian", "Cold Sub-Jovian",
              "Hot Jovian", "Warm Jovian", "Cold Jovian"]
    stypes = ["F", "G", "K", "M"]
    # Bar Plot
    bar_cat_plot(life_data, exo_data, ptypes, 'ptype', stypes, 'stype', save=True, result_path=results_path,
                 name="ptypes_stypes")

    habitables = [True, False]
    bar_cat_plot(life_data, exo_data, habitables, 'habitable', stypes, 'stype', save=True, result_path=results_path,
                 name="habitable_stypes")

    ##################################
    # Population Sample Histograms ###
    ##################################

    histogram_single(life_data["radius_p"].astype(float), results_path, "Distribution Planet Radius Population Sample",
                     r'Planet Radius [$R_{\oplus}$]', xlim=Rp_lim)

    histogram_single(life_data["Mp"].astype(float), results_path, "Distribution Planet Mass Population Sample",
                     r'Planet Mass [$M_{\oplus}$]', xlim=Mp_lim)

    histogram_single(life_data["rp"].astype(float), results_path, "Distribution Orbital Distance Population Sample",
                     "Orbital Distance [AU]", xlim=d_orbit_lim)

    histogram_single(life_data["distance_s"].astype(float), results_path,
                     "Distribution System Distance Population Sample",
                     "System Distance [pc]", xlim=d_system_lim)

    histogram_single(np.log10(life_data["radius_p"].astype(float)), results_path,
                     "Distribution Planet Radius Population Sample log-scale",
                     r'$log_{10}\ R/R_{\oplus}$', xlim=Rp_lim_log, log=True)

    histogram_single(np.log10(life_data["Mp"].astype(float)), results_path,
                     "Distribution Planet Mass Population Sample log-scale",
                     r'$log_{10}\ M/M_{\oplus}$', xlim=Mp_lim_log, log=True)

    histogram_single(np.log10(life_data["rp"].astype(float)), results_path,
                     "Distribution Orbital Distance Population Sample log-scale",
                     r'$log_{10}\ d_{orbit}$ [AU]', xlim=d_orbit_lim_log, log=True)

    histogram_single(np.log10(life_data["distance_s"].astype(float)), results_path,
                     "Distribution System Distance Population Sample log-scale",
                     r'$log_{10}\ d_{system}[pc]$', xlim=d_system_lim_log, log=True)

    ##################################
    # Detected Planets Histograms ####
    ##################################
    # Planet Radius
    kde_distr_plot(life_data_det["radius_p"].astype(float), exo_data_det["radius_p"].astype(float),
                   life_data["radius_p"].astype(float), results_path,
                   "Distribution Planet Radius Detected Planets",
                   r'Planet Radius [$R_{\oplus}$]', xlim=Rp_lim, detected=True)
    kde_distr_plot(np.log10(life_data_det["radius_p"].astype(float)),
                   np.log10(exo_data_det["radius_p"].astype(float)),
                   np.log10(life_data["radius_p"].astype(float)), results_path,
                   "log Distribution Planet Radius Detected Planets",
                   r'$log_{10}\ R/R_{\oplus}$', xlim=Rp_lim_log, detected=True, log=True)

    # Planet Mass
    kde_distr_plot(life_data_det["Mp"].astype(float), exo_data_det["Mp"].astype(float),
                   life_data["Mp"].astype(float), results_path,
                   "Distribution Planet Mass Detected Planets",
                   r'Planet Mass [$M_{\oplus}$]', xlim=Mp_lim, detected=True)
    kde_distr_plot(np.log10(life_data_det["Mp"].astype(float)),
                   np.log10(exo_data_det["Mp"].astype(float)),
                   np.log10(life_data["Mp"].astype(float)), results_path,
                   "log Distribution Planet Mass Detected Planets",
                   r'$log_{10}\ M/M_{\oplus}$', xlim=Mp_lim_log, detected=True, log=True)

    # Planet orbital distance
    kde_distr_plot(life_data_det["rp"].astype(float), exo_data_det["distance_p"].astype(float),
                   life_data["rp"].astype(float), results_path, "Distribution Orbital Distance Detected Planets",
                   "Orbital Distance [AU]", xlim=d_orbit_lim, detected=True)
    kde_distr_plot(np.log10(life_data_det["rp"].astype(float)),
                   np.log10(exo_data_det["distance_p"].astype(float)),
                   np.log10(life_data["rp"].astype(float)), results_path,
                   "log Distribution Orbital Distance Detected Planets",
                   r'$log_{10}\ d_{orbit}$ [AU]', xlim=d_orbit_lim_log, detected=True, log=True)

    # Also scaled by stellar luminosity HZ Values are for solar like star (according to Hz definition in LIFEsim) and
    # values will naturally be scaled to that in the histogram
    hz_in = 0.75
    hz_out = 1.77
    kde_distr_plot((life_data_det["rp"] / np.sqrt(life_data_det["l_sun"])).astype(float),
                   (exo_data_det["distance_p"] / np.sqrt(exo_data_det["L_star"])).astype(float),
                   (life_data["rp"] / np.sqrt(life_data["l_sun"])).astype(float),
                   results_path, "Distribution Orbital Distance Scaled Detected Planets",
                   r'$d_{orbit}$ [AU] / $\sqrt{L_{star}}$', xlim=d_orbit_scaled_lim, detected=True,
                   draw_HZ=True, HZ_inner=hz_in, HZ_outer=hz_out)
    kde_distr_plot(np.log10((life_data_det["rp"] / np.sqrt(life_data_det["l_sun"])).astype(float)),
                   np.log10((exo_data_det["distance_p"] / np.sqrt(exo_data_det["L_star"])).astype(float)),
                   np.log10((life_data["rp"] / np.sqrt(life_data["l_sun"])).astype(float)),
                   results_path, "log Distribution Orbital Distance Scaled Detected Planets",
                   r'$log_{10}\ (d_{orbit}$ [AU] / $\sqrt{L_{star}})$', xlim=d_orbit_scaled_lim_log,
                   detected=True, draw_HZ=True, HZ_inner=np.log10(hz_in), HZ_outer=np.log10(hz_out),
                   log=True)

    # System Distance
    kde_distr_plot(life_data_det["distance_s"].astype(float), exo_data_det["distance_s"].astype(float),
                   life_data["distance_s"].astype(float), results_path,
                   "Distribution System Distance Detected Planets",
                   "System Distance [pc]", xlim=d_system_lim, detected=True)
    kde_distr_plot(np.log10(life_data_det["distance_s"].astype(float)),
                   np.log10(exo_data_det["distance_s"].astype(float)),
                   np.log10(life_data["distance_s"].astype(float)), results_path,
                   "log Distribution System Distance Detected Planets",
                   r'$log_{10}\ d_{system}$ [pc]', xlim=d_system_lim_log, detected=True, log=True)

    ############################
    # Virtual Population Plots #
    ############################
    # Rp vs Mp
    kde_distribution_plot(life_data["radius_p"].astype(float), life_data["Mp"].astype(float), results_path,
                          "Sample Population Rp-Mp",
                          r'Planet Radius [$R_{\oplus}$]', r'Planet Mass [$M_{\oplus}$]',
                          xlim=Rp_lim, ylim=Mp_lim, detected=False)
    kde_distribution_plot(np.log10(life_data["radius_p"].astype(float)), np.log10(life_data["Mp"].astype(float)),
                          results_path,
                          "Sample Population Rp-Mp log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ M/M_{\oplus}$',
                          xlim=Rp_lim_log, ylim=Mp_lim_log, detected=False)
    # Rp vs d_orbit
    kde_distribution_plot(life_data["radius_p"].astype(float), life_data["rp"].astype(float), results_path,
                          "Sample Population Rp-Orbit",
                          r'Planet Radius [$R_{\oplus}$]', "Orbital Distance [AU]",
                          xlim=Rp_lim, ylim=d_orbit_lim, detected=False)
    kde_distribution_plot(np.log10(life_data["radius_p"].astype(float)), np.log10(life_data["rp"].astype(float)),
                          results_path,
                          "Sample Population Rp-Orbit log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ d_{orbit}$ [AU]',
                          xlim=Rp_lim_log, ylim=d_orbit_lim_log, detected=False)

    # Rp vs d_system
    kde_distribution_plot(life_data["radius_p"].astype(float), life_data["distance_s"].astype(float), results_path,
                          "Sample Population Rp-d_s",
                          r'Planet Radius [$R_{\oplus}$]', "System Distance [pc]",
                          xlim=Rp_lim, ylim=d_system_lim, detected=False)
    kde_distribution_plot(np.log10(life_data["radius_p"].astype(float)),
                          np.log10(life_data["distance_s"].astype(float)),
                          results_path, " Sample Population Rp-d_s log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ d_{system}$ [pc]',
                          xlim=Rp_lim_log, ylim=d_system_lim_log, detected=False)

    # d_orbit vs d_system
    kde_distribution_plot(life_data["rp"].astype(float), life_data["distance_s"].astype(float), results_path,
                          "Sample Population d_orbit-d_s",
                          "Orbital Distance [AU]", "System Distance [pc]",
                          xlim=d_orbit_lim, ylim=d_system_lim, detected=False)
    kde_distribution_plot(np.log10(life_data["rp"].astype(float)), np.log10(life_data["distance_s"].astype(float)),
                          results_path, "Sample Population d_orbit-d_s log-log",
                          r'$log_{10}\ d_{orbit}$ [AU]', r'$log_{10}\ d_{system}$ [pc]',
                          xlim=d_orbit_lim_log, ylim=d_system_lim_log, detected=False)

    ##############################
    # Detected Population Plots ##
    ##############################

    # Distribution Plots Rp Mp
    ##########################
    kde_distribution_plot(exo_data_det["radius_p"].astype(float), exo_data_det["Mp"].astype(float), results_path,
                          "EXOsim Detected Rp-Mp",
                          r'Planet Radius [$R_{\oplus}$]', r'Planet Mass [$M_{\oplus}$]',
                          xlim=Rp_lim, ylim=Mp_lim, detected=True)
    kde_distribution_plot(life_data_det["radius_p"].astype(float), life_data_det["Mp"].astype(float), results_path,
                          "LIFEsim Detected Rp-Mp",
                          r'Planet Radius [$R_{\oplus}$]', r'Planet Mass [$M_{\oplus}$]',
                          xlim=Rp_lim, ylim=Mp_lim, detected=True)

    # log-log
    # TODO for some reason, the log plot for Rp-MP and Rp-Mp only halves the axis labels and the number of detections, but
    # TODO only when running on bluesky. When running locally this bug does not appear
    kde_distribution_plot(np.log10(exo_data_det["radius_p"].astype(float)), np.log10(exo_data_det["Mp"].astype(float)),
                          results_path, "EXOsim Detected Rp-Mp log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ M/M_{\oplus}$',
                          xlim=Rp_lim_log, ylim=Mp_lim_log, detected=True)
    kde_distribution_plot(np.log10(life_data_det["radius_p"].astype(float)),
                          np.log10(life_data_det["Mp"].astype(float)),
                          results_path, "LIFEsim Detected Rp-Mp log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ M/M_{\oplus}$',
                          xlim=Rp_lim_log, ylim=Mp_lim_log, detected=True)

    # Distribution Plots Rp d_orbit
    ###############################

    kde_distribution_plot(exo_data_det["radius_p"].astype(float), exo_data_det["distance_p"].astype(float),
                          results_path,
                          "EXOsim Detected Rp-Orbit",
                          r'Planet Radius [$R_{\oplus}$]', "Orbital Distance [AU]",
                          xlim=Rp_lim, ylim=d_orbit_lim, detected=True)
    kde_distribution_plot(life_data_det["radius_p"].astype(float), life_data_det["rp"].astype(float), results_path,
                          "LIFEsim Detected Rp-Orbit",
                          r'Planet Radius [$R_{\oplus}$]', "Orbital Distance [AU]",
                          xlim=Rp_lim, ylim=d_orbit_lim, detected=True)

    # log-log
    kde_distribution_plot(np.log10(exo_data_det["radius_p"].astype(float)),
                          np.log10(exo_data_det["distance_p"].astype(float)),
                          results_path, "EXOsim Detected Rp-Orbit log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ d_{orbit}$ [AU]',
                          xlim=Rp_lim_log, ylim=d_orbit_lim_log, detected=True)
    kde_distribution_plot(np.log10(life_data_det["radius_p"].astype(float)),
                          np.log10(life_data_det["rp"].astype(float)),
                          results_path, "LIFEsim Detected Rp-Orbit log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ d_{orbit}$ [AU]',
                          xlim=Rp_lim_log, ylim=d_orbit_lim_log, detected=True)

    # Distribution Plots Rp d_system
    #################################

    kde_distribution_plot(exo_data_det["radius_p"].astype(float), exo_data_det["distance_s"].astype(float),
                          results_path,
                          "EXOsim Detected Rp-d_s",
                          r'Planet Radius [$R_{\oplus}$]', "System Distance [pc]",
                          xlim=Rp_lim, ylim=d_system_lim, detected=True)
    kde_distribution_plot(life_data_det["radius_p"].astype(float), life_data_det["distance_s"].astype(float),
                          results_path,
                          "LIFEsim Detected Rp-d_s",
                          r'Planet Radius [$R_{\oplus}$]', "System Distance [pc]",
                          xlim=Rp_lim, ylim=d_system_lim, detected=True)

    # log-log
    kde_distribution_plot(np.log10(exo_data_det["radius_p"].astype(float)),
                          np.log10(exo_data_det["distance_s"].astype(float)),
                          results_path, "EXOsim Detected Rp-d_s log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ d_{system}$ [pc]',
                          xlim=Rp_lim_log, ylim=d_system_lim_log, detected=True)
    kde_distribution_plot(np.log10(life_data_det["radius_p"].astype(float)),
                          np.log10(life_data_det["distance_s"].astype(float)),
                          results_path, "LIFEsim Detected Rp-d_s log-log",
                          r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ d_{system}$ [pc]',
                          xlim=Rp_lim_log, ylim=d_system_lim_log, detected=True)

    # d_orbit vs d_system
    #####################
    kde_distribution_plot(exo_data_det["distance_p"].astype(float), exo_data_det["distance_s"].astype(float),
                          results_path,
                          "EXOsim Detected d_orbit-d_s",
                          "Orbital Distance [AU]", "System Distance [pc]",
                          xlim=d_orbit_lim, ylim=d_system_lim, detected=True)
    kde_distribution_plot(life_data_det["rp"].astype(float), life_data_det["distance_s"].astype(float), results_path,
                          "LIFEsim Detected d_orbit-d_s",
                          "Orbital Distance [AU]", "System Distance [pc]",
                          xlim=d_orbit_lim, ylim=d_system_lim, detected=True)

    # log-log
    kde_distribution_plot(np.log10(exo_data_det["distance_p"].astype(float)),
                          np.log10(exo_data_det["distance_s"].astype(float)),
                          results_path, "EXOsim Detected d_orbit-d_s log-log",
                          r'$log_{10}\ d_{orbit}$ [AU]', r'$log_{10}\ d_{system}$ [pc]',
                          xlim=d_orbit_lim_log, ylim=d_system_lim_log, detected=True)
    kde_distribution_plot(np.log10(life_data_det["rp"].astype(float)),
                          np.log10(life_data_det["distance_s"].astype(float)),
                          results_path, "LIFEsim Detected d_orbit-d_s log-log",
                          r'$log_{10}\ d_{orbit}$ [AU]', r'$log_{10}\ d_{system}$ [pc]',
                          xlim=d_orbit_lim_log, ylim=d_system_lim_log, detected=True)
    return None


def radius_mass_check(life_data, exo_data, life_data_det, exo_data_det, results_path):
    """
    Check the radius and mass parameter space with the tool of Chen and Kipping "forecaster"
    https://github.com/chenjj2/forecaster
    :param life_data: all LIFEsim data
    :param exo_data: all EXOsim data
    :param life_data_det: all LIFEsim detected data
    :param exo_data_det: all EXOsim detected data
    :param results_path: results path
    :return: parameters space plots of Radius and Mass
    """

    xlim = [-1, 3]
    # Check if "Radius_Mass_Doublecheck" directory exists in results_path
    radius_mass_dir = results_path.joinpath("Radius_Mass_Doublecheck")

    if not os.path.exists(radius_mass_dir):
        # Create the "Radius_Mass_Doublecheck" directory if it doesn't exist
        os.makedirs(radius_mass_dir)

    # Update results_path to point to the "Radius_Mass_Doublecheck" directory
    results_path = radius_mass_dir
    #############################
    # All Underlying Data
    #############################
    # Save to demo_mass.dat and demo_radius.dat
    current_location = Path(__file__).parent.resolve()
    mass_path = current_location.joinpath("forecaster/demo_mass.dat")
    radius_path = current_location.joinpath("forecaster/demo_radius.dat")
    life_data["Mp"].to_csv(mass_path, sep="\t", index=False, header=False)
    life_data["radius_p"].to_csv(radius_path, sep="\t", index=False, header=False)

    # Rmedian, Rplus, Rminus = mr.Mstat2R(mean=1.0, std=0.1, unit='Earth', sample_size=100, classify='Yes')
    # print('Sample Population R = %.2f (+ %.2f - %.2f) REarth' % (Rmedian, Rplus, Rminus))

    M1 = np.loadtxt(mass_path)
    R1 = mr.Mpost2R(M1, unit='Earth', classify='Yes')
    plt.hist(np.log10(R1), bins=25)
    plt.xlabel(r'$log_{10}\ R/R_{\oplus}$')
    title = 'Sample Population Radius from Mass'
    plt.title(title)
    plt.savefig(results_path.joinpath(title + ".svg"))
    plt.close()

    # Mmedian, Mplus, Mminus = mr.Rstat2M(mean=0.1, std=0.01, unit='Earth', sample_size=100, grid_size=1e3, classify='Yes')
    # print('Sample Population M = %.3f (+ %.3f - %.3f) MEarth' % (Mmedian, Mplus, Mminus))

    R2 = np.loadtxt(radius_path)
    M2 = mr.Rpost2M(R2, unit='Earth', grid_size=int(1000), classify='Yes')
    plt.hist(np.log10(M2), bins=25)
    plt.xlabel(r'$log_{10}\ M/M_{\odot}$')
    title = 'Sample Population Mass from Radius'
    plt.title(title)
    plt.savefig(results_path.joinpath(title + ".svg"))
    plt.close()

    #############################
    # LIFE Detections
    #############################
    # Save to demo_mass.dat and demo_radius.dat
    current_location = Path(__file__).parent.resolve()
    mass_path = current_location.joinpath("forecaster/demo_mass.dat")
    radius_path = current_location.joinpath("forecaster/demo_radius.dat")
    life_data_det["Mp"].to_csv(mass_path, sep="\t", index=False, header=False)
    life_data_det["radius_p"].to_csv(radius_path, sep="\t", index=False, header=False)

    # Rmedian, Rplus, Rminus = mr.Mstat2R(mean=1.0, std=0.1, unit='Earth', sample_size=100, classify='Yes')
    # print('Sample Population R = %.2f (+ %.2f - %.2f) REarth' % (Rmedian, Rplus, Rminus))

    M1 = np.loadtxt(mass_path)
    R1 = mr.Mpost2R(M1, unit='Earth', classify='Yes')
    plt.xlim(left=xlim[0], right=xlim[1])
    plt.hist(np.log10(R1), bins=25, color='green')
    plt.xlabel(r'$log_{10}\ R/R_{\oplus}$')
    title = 'LIFEsim Detections Radius from Mass'
    plt.title(title)
    plt.savefig(results_path.joinpath(title + ".svg"))
    plt.close()

    # Mmedian, Mplus, Mminus = mr.Rstat2M(mean=0.1, std=0.01, unit='Earth', sample_size=100, grid_size=1e3, classify='Yes')
    # print('Sample Population M = %.3f (+ %.3f - %.3f) MEarth' % (Mmedian, Mplus, Mminus))

    R2 = np.loadtxt(radius_path)
    M2 = mr.Rpost2M(R2, unit='Earth', grid_size=int(1000), classify='Yes')
    plt.hist(np.log10(M2), bins=25, color='green')
    plt.xlabel(r'$log_{10}\ M/M_{\odot}$')
    title = 'LIFEsim Detections Mass from Radius'
    plt.title(title)
    plt.savefig(results_path.joinpath(title + ".svg"))
    plt.close()

    #############################
    # EXOsim Detections
    #############################
    # Save to demo_mass.dat and demo_radius.dat
    current_location = Path(__file__).parent.resolve()
    mass_path = current_location.joinpath("forecaster/demo_mass.dat")
    radius_path = current_location.joinpath("forecaster/demo_radius.dat")
    exo_data_det["Mp"].to_csv(mass_path, sep="\t", index=False, header=False)
    exo_data_det["radius_p"].to_csv(radius_path, sep="\t", index=False, header=False)

    # Rmedian, Rplus, Rminus = mr.Mstat2R(mean=1.0, std=0.1, unit='Earth', sample_size=100, classify='Yes')
    # print('Sample Population R = %.2f (+ %.2f - %.2f) REarth' % (Rmedian, Rplus, Rminus))

    M1 = np.loadtxt(mass_path)
    R1 = mr.Mpost2R(M1, unit='Earth', classify='Yes')
    plt.xlim(left=xlim[0], right=xlim[1])
    plt.hist(np.log10(R1), bins=25, color='blue')
    plt.xlabel(r'$log_{10}\ R/R_{\oplus}$')
    title = 'EXOsim Detections Radius from Mass'
    plt.title(title)
    plt.savefig(results_path.joinpath(title + ".svg"))
    plt.close()

    # Mmedian, Mplus, Mminus = mr.Rstat2M(mean=0.1, std=0.01, unit='Earth', sample_size=100, grid_size=1e3, classify='Yes')
    # print('Sample Population M = %.3f (+ %.3f - %.3f) MEarth' % (Mmedian, Mplus, Mminus))

    R2 = np.loadtxt(radius_path)
    M2 = mr.Rpost2M(R2, unit='Earth', grid_size=int(1000), classify='Yes')
    plt.hist(np.log10(M2), bins=25, color='blue')
    plt.xlabel(r'$log_{10}\ M/M_{\odot}$')
    title = 'EXOsim Detections Mass from Radius'
    plt.title(title)
    plt.savefig(results_path.joinpath(title + ".svg"))
    plt.close()

    return None


def bin_parameter_space(data, x_param, xlim, y_param, ylim, n_bins):
    """
    This function, given the dataframe 'data' and the names of the two parameters, splits the data
    accordingly into n_bins^2 bins in the 2-D parameter space. This acts as a first step towards the
    calculation of the discrete Depth of Search.
    :param data: dataframe containing the data, either in form of the LIFEsim output Dataframe or the
    EXOsim output Dataframe
    :param x_param: String with the name of the parameter in the x-axis. Name must match the naming in
    the dataframe 'data'.
    :param xlim: List with the lower and upper limit of the x-axis [x_min, x_max]
    :param y_param: String with the name of the parameter in the y-axis. Name must match the naming in
    the dataframe 'data'.
    :param ylim: List with the lower and upper limit of the y-axis [y_min, y_max]
    :param n_bins: Number of bins per axis. Total number of bins will be n_bins^2. Calculation time therefore
    scales O(n_bins^2), but at the same time, higher amount of bins allows for a smoother DoS distribution.
    :return: Normalized probability of occurance in the 2-D parameter space of the dataframe 'data'
    """
    t1 = time.time()
    N_tot = len(data)
    twod_param_p, twod_paramspace = np.zeros((n_bins, n_bins)), np.zeros((n_bins, n_bins, 2))
    mask = (data[x_param] > xlim[0]) & (data[x_param] < xlim[1]) & (data[y_param] > ylim[0]) & (data[y_param] < ylim[1])
    data_masked = data[mask]
    x_data, y_data = data_masked[x_param], data_masked[y_param]
    x_linspace, y_linspace = np.linspace(xlim[0], xlim[1], n_bins), np.linspace(ylim[0], ylim[1], n_bins)
    for i in range(n_bins):
        for j in range(n_bins):
            # print("Starting round (", i, j, ")")
            try:
                mask_ij = (x_data > x_linspace[i]) & (x_data < x_linspace[i + 1]) & (y_data > y_linspace[j]) & (
                        y_data < y_linspace[j + 1])
            except IndexError:
                mask_ij = (x_data > x_linspace[i]) & (y_data > y_linspace[j])
            N_ij = np.sum(mask_ij)
            twod_param_p[i, j] = N_ij / N_tot
            twod_paramspace[i, j] = [x_linspace[i], y_linspace[j]]

    t2 = time.time()
    print("Time to bin parameter space for", x_param, y_param, ": ", t2 - t1)

    return twod_param_p, twod_paramspace


def plot_heatmap(data, xlim, xlable, ylim, ylable, dos_cont, results_path, title):
    """
    Generates and displays a heatmap based on input data and parameters.

    Parameters:
    data (array-like): Two-dimensional array of data values for the heatmap.
    xlim (list): List containing two values specifying the x-axis limits.
    xlable (str): Label for the x-axis.
    ylim (list): List containing two values specifying the y-axis limits.
    ylabel (str): Label for the y-axis.
    dos_cont (array-like): One-dimensional array of depth of search values for the contour lines.
    results_path (Path or str): Path where the SVG file will be saved.
    title (str): Title of the heatmap.

    Returns: None
    """
    # Fill NaN values with 0
    data = np.nan_to_num(data)

    # Create a heatmap
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    # Choose a colormap for color coding (e.g., 'viridis', 'plasma', 'magma', 'inferno', etc.)
    # More colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cmap = 'viridis'

    # Create the heatmap
    # For some reason it was flipped, by transposing you can fix it
    heatmap = plt.imshow(data.T, cmap=cmap, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto')
    xi, yi = np.meshgrid(np.linspace(xlim[0], xlim[1], int(np.sqrt(dos_cont.shape[0]))),
                         np.linspace(ylim[0], ylim[1], int(np.sqrt(dos_cont.shape[0]))))
    contour_levels = np.linspace(0, data.max(), 10)
    contour = plt.contour(xi, yi, dos_cont.reshape(yi.shape).T, levels=contour_levels, colors='black', linewidths=0.5)

    # Add a colorbar
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Depth of Search', rotation=270, labelpad=15)

    # Set axis labels if needed
    plt.xlabel(xlable)
    plt.ylabel(ylable)

    # Add title
    plt.title(title)
    plt.savefig(results_path.joinpath(title + ".svg"))

    # Clear Figure so it does not interfere with other plots
    plt.clf()

    return None


def contourf_single_plot(xi, yi, zi, results_path, title, xlabel, ylabel, DoS=False, logb=False):
    """
    This function generates a contourf plot of the Depth of Search or distributions for a single parameter space.
    :param xi: X grid of the parameter space
    :param yi: Y grid of the parameter space
    :param zi: Z grid of the parameter space
    :param results_path: Path where the results will be saved
    :param title: Title of the plot
    :param xlabel: Label of the x-axis
    :param ylabel: Label of the y-axis
    :return:
    """

    contour_levels = np.linspace(zi.min(), zi.max(), 10)

    if title[:15] == "Depth of Search":
        if logb:
            contour_label = r'$log_{10}\ Depth\ of\ Search$'
        else:
            contour_label = "Depth of Search"
    else:
        contour_label = "Probability Density"

    contourf = plt.contourf(xi, yi, zi.reshape(xi.shape), levels=contour_levels, cmap='viridis', extend='both')
    # Create contour lines with black color
    contour = plt.contour(xi, yi, zi.reshape(xi.shape), levels=contour_levels, colors='black', linewidths=0.5)

    plt.colorbar(contourf, label=contour_label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(results_path.joinpath(title + ".svg"))
    plt.close()

    return None


def dos_analysis_naive(sample_data, det_data, results_path, indicator, N_bin=25, sim_name=None):
    """
    This function summarizes all the analysis, calculations and plot generation with respect to the Depth of Search
    Metric. This includes (currently):
    - Splitting the underlying Sample Population of the virtual Universes into bins in different 2-D parameter spaces
    - Splitting the observed Population into bins the same bins
    - Splitting the detected Population into bins the same bins
    - Calculating the Depth of Search for each bin
    - (Not yet known if necessary) Some re-normalization along the way
    - Plotting the Depth of Search for each bin
    - Plotting the Depth of Search as a continuous distribution / heatmap

    :param sample_data: Dataframe containing the underlying Sample Population of the virtual Universes
    :param det_data: Dataframe containing the detected Population of the virtual Universes
    :param results_path: Path where the results will be saved
    :param indicator: String indicating which simulation is being analysed. Either LIFE or EXO
    :param N_bin: Number of bins per axis. Total number of bins will be n_bins^2. Calculation time therefore
    scales O(n_bins^2), but at the same time, higher amount of bins allows for a smoother DoS distribution.
    :return:
    - Discrete Depth of Search Plots
    - Continuous Depth of Search Plots
    """
    # If no sim_name is given, just add empty strings to all the titles
    if sim_name is None:
        sim_name = ""

    # Setup: List of all the parameters, the labels and the limits
    if indicator == "LIFEsim":
        parameters = ["radius_p", "rp", "Mp", "distance_s"]
    elif indicator == "EXOSIMS":
        parameters = ["radius_p", "distance_p", "Mp", "distance_s"]
    else:
        raise ValueError("Indicator must be either <LIFEsim> or <EXOSIMS>")
        return None

    labels_log = [r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ d_{orbit}$ [AU]', r'$log_{10}\ M/M_{\oplus}$',
                  r'$log_{10}\ d_{system}$ [pc]']
    labels = [r'Planet Radius [$R_{\oplus}$]', "Orbital Distance [AU]", r'Planet Mass [$M_{\oplus}$]',
              "System Distance [pc]"]
    limits = [Rp_lim, d_orbit_lim, Mp_lim, d_system_lim]
    limits_log = [Rp_lim_log, d_orbit_lim_log, Mp_lim_log, d_system_lim_log]

    # Housekeeping with the number of bins
    N_bin += 1

    # Take out any columns with strings from the dataset so the np.log10() does not fuck up later
    sample_data_reduced = sample_data[parameters]
    det_data_reduced = det_data[parameters]

    ##################################################
    # START OF THE DISCRETE DEPTH OF SEARCH ANALYSIS #
    ##################################################

    # Check if "Depth of Search Discrete" directory exists in results_path
    dos_discrete_dir_temp = results_path.joinpath("Depth_of_Search_Discrete_Naive")
    if not os.path.exists(dos_discrete_dir_temp):
        # Create the "Depth_of_Search" directory if it doesn't exist
        os.makedirs(dos_discrete_dir_temp)

    # Check if "Depth of Search Continuous" directory exists in results_path
    dos_cont_dir_temp = results_path.joinpath("Depth_of_Search_Continuous_Naive")
    if not os.path.exists(dos_cont_dir_temp):
        # Create the "Depth_of_Search" directory if it doesn't exist
        os.makedirs(dos_cont_dir_temp)

    # Check if "Distributions Continuous" Directory exists in results_path
    distr_cont_dir_temp = results_path.joinpath("Continuous_Distribution")
    if not os.path.exists(distr_cont_dir_temp):
        # Create the "Depth_of_Search" directory if it doesn't exist
        os.makedirs(distr_cont_dir_temp)

    # Loop over all the parameters
    for i in range(len(parameters) - 1):
        for j in range(i + 1, len(parameters)):
            t1 = time.time()
            # Make Directories for this Parameter Run
            # Check if "Depth of Search Discrete" directory exists in results_path
            dos_discrete_dir = dos_discrete_dir_temp.joinpath(parameters[i] + "-" + parameters[j])
            if not os.path.exists(dos_discrete_dir):
                # Create the "Depth_of_Search" directory if it doesn't exist
                os.makedirs(dos_discrete_dir)

            # Check if "Depth of Search Continuous" directory exists in results_path
            dos_cont_dir = dos_cont_dir_temp.joinpath(parameters[i] + "-" + parameters[j])
            if not os.path.exists(dos_cont_dir):
                # Create the "Depth_of_Search" directory if it doesn't exist
                os.makedirs(dos_cont_dir)

            # Check if "Distributions Continuous" Directory exists in results_path
            distr_cont_dir = distr_cont_dir_temp.joinpath(parameters[i] + "-" + parameters[j])
            if not os.path.exists(distr_cont_dir):
                # Create the "Depth_of_Search" directory if it doesn't exist
                os.makedirs(distr_cont_dir)

            xlim, ylim, xlim_log, ylim_log = limits[i], limits[j], limits_log[i], limits_log[j]

            x_sample, y_sample = sample_data[parameters[i]], sample_data[parameters[j]]
            x_det, y_det = det_data[parameters[i]], det_data[parameters[j]]
            ###################################
            # Linear Depth of Search Analysis #
            ###################################
            sample_pop_bins, sample_paramspace = bin_parameter_space(sample_data, parameters[i], xlim,
                                                                     parameters[j], ylim, n_bins=N_bin)
            det_bins, det_paramspace = bin_parameter_space(det_data, parameters[i], xlim, parameters[j], ylim,
                                                           n_bins=N_bin)
            dos = det_bins / sample_pop_bins

            mesh_N = 200j
            adjust_bw = 0.5
            xi, yi = np.mgrid[xlim[0]:xlim[1]:mesh_N, ylim[0]:ylim[1]:mesh_N]

            kde_sample = gaussian_kde([x_sample, y_sample])
            kde_sample.set_bandwidth(kde_sample.factor * adjust_bw)
            zi_sample = kde_sample(np.vstack([xi.flatten(), yi.flatten()]))

            kde_det = gaussian_kde([x_det, y_det])
            kde_det.set_bandwidth(kde_det.factor * adjust_bw)
            zi_det = kde_det(np.vstack([xi.flatten(), yi.flatten()]))

            threshold = 1e-4
            zi_sample[zi_sample < threshold * zi_sample.max()] = 0

            with np.errstate(divide='ignore', invalid='ignore'):
                dos_cont = np.true_divide(zi_det, zi_sample)
                dos_cont[dos_cont == np.inf] = 0
                dos_cont = np.nan_to_num(dos_cont)

            plot_heatmap(dos, xlim, labels[i], ylim, labels[j], dos_cont, dos_discrete_dir,
                         sim_name + "Depth of Search " + indicator + "_" + parameters[i] + "-" + parameters[j])

            contourf_single_plot(xi, yi, zi_sample, distr_cont_dir,
                                 sim_name + "Distribution Sample Population" + indicator + "_" + parameters[i] + "-" +
                                 parameters[j],
                                 labels[i], labels[j])

            contourf_single_plot(xi, yi, zi_det, distr_cont_dir,
                                 sim_name + "Distribution " + indicator + " Detections" + "_" + parameters[i] + "-" +
                                 parameters[j],
                                 labels[i], labels[j])

            contourf_single_plot(xi, yi, dos_cont, dos_cont_dir,
                                 sim_name + "Depth of Search " + indicator + "_" + parameters[i] + "-" + parameters[j],
                                 labels[i], labels[j], DoS=True)

            ################################
            # Log Depth of Search Analysis #
            ################################

            sample_pop_bins_log, sample_paramspace_log = bin_parameter_space(np.log10(sample_data_reduced),
                                                                             parameters[i],
                                                                             xlim_log,
                                                                             parameters[j], ylim_log, n_bins=N_bin)
            det_bins_log, det_paramspace_log = bin_parameter_space(np.log10(det_data_reduced), parameters[i], xlim_log,
                                                                   parameters[j], ylim_log, n_bins=N_bin)
            dos_log = det_bins_log / sample_pop_bins_log

            xi_log, yi_log = np.mgrid[xlim_log[0]:xlim_log[1]:mesh_N, ylim_log[0]:ylim_log[1]:mesh_N]

            kde_sample_log = gaussian_kde([np.log10(x_sample), np.log10(y_sample)])
            kde_sample_log.set_bandwidth(kde_sample_log.factor * adjust_bw)
            zi_sample_log = kde_sample_log(np.vstack([xi_log.flatten(), yi_log.flatten()]))

            kde_det_log = gaussian_kde([np.log10(x_det), np.log10(y_det)])
            kde_det_log.set_bandwidth(kde_det_log.factor * adjust_bw)
            zi_det_log = kde_det_log(np.vstack([xi_log.flatten(), yi_log.flatten()]))

            threshold_log = 1e-3
            zi_sample_log[zi_sample_log < threshold_log * zi_sample_log.max()] = 0

            with np.errstate(divide='ignore', invalid='ignore'):
                dos_cont_log = np.true_divide(zi_det_log, zi_sample_log)
                dos_cont_log[dos_cont_log == np.inf] = 0
                dos_cont_log = np.nan_to_num(dos_cont_log)

            plot_heatmap(dos_log, xlim_log, labels_log[i], ylim_log, labels_log[j], dos_cont_log, dos_discrete_dir,
                         sim_name + "Depth of Search " + indicator + "_" + parameters[i] + "-" + parameters[j] + "_log")

            contourf_single_plot(xi_log, yi_log, zi_sample_log, distr_cont_dir,
                                 sim_name + "Distribution Sample Population" + indicator + "_" + parameters[i] + "-" +
                                 parameters[j] + "_log",
                                 labels_log[i], labels_log[j])

            contourf_single_plot(xi_log, yi_log, zi_det_log, distr_cont_dir,
                                 sim_name + "Distribution " + indicator + " Detections" + "_" + parameters[i] + "-" +
                                 parameters[j] + "_log",
                                 labels_log[i], labels_log[j])

            contourf_single_plot(xi_log, yi_log, dos_cont_log, dos_cont_dir,
                                 sim_name + "Depth of Search " + indicator + "_" + parameters[i] + "-" + parameters[j]
                                 + "_log",
                                 labels_log[i], labels_log[j], DoS=True, logb=True)

    t2 = time.time()
    print("Time to calculate Depth of Search for", parameters[i], parameters[j], ": ", t2 - t1)

    return None


def analyse_one_dos(dos_pop_path, sim_names, results_path, indicator, N_bin=50):
    """
    Uses one of the Sims to analyse the depth of Search of it. Used by DoS_validation_tests.py
    :param sim_name: Name of the Sims to be analysed
    :param results_path: save the results
    :return:
    """
    for i in range(len(sim_names)):
        sim_name = sim_names[i]
        dost_stress_test_life_data = pd.read_hdf(dos_pop_path.joinpath("sim_results/" + sim_name))
        DoS_stress_test_data_det = gd.data_only_det(dost_stress_test_life_data)
        # For Whatever reason P-Pop is inconsistent with its naming conventions
        if 'rp' not in dost_stress_test_life_data.columns:
            dost_stress_test_life_data.rename(columns={'semimajor_p': 'rp'}, inplace=True)
            DoS_stress_test_data_det.rename(columns={'semimajor_p': 'rp'}, inplace=True)
        if 'mass_p' in dost_stress_test_life_data.columns:
            dost_stress_test_life_data.rename(columns={'mass_p': 'Mp'}, inplace=True)
            DoS_stress_test_data_det.rename(columns={'mass_p': 'Mp'}, inplace=True)

        dos_analysis_naive(dost_stress_test_life_data, DoS_stress_test_data_det, results_path, indicator, N_bin,
                           sim_name=sim_name)

    return None


def corner_plots(sample_data, life_det, exo_det, results_path):
    """
    This function generates corner plots of the underlying sample population and of the detected planets from the
    Exosim and Lifesim simulations using the seaborn package. Once for linear and once for log-log plots.
    :param sample_data:
    :param life_det:
    :param exo_det:
    :param results_path:
    :return:
    """

    save_path = results_path.joinpath("Corner_Plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    parameters_life = ["radius_p", "rp", "rp_scaled", "Mp", "distance_s"]
    parameters_exo = ["radius_p", "distance_p", "distance_p_scaled", "Mp", "distance_s"]

    labels_log = [r'$log_{10}\ R/R_{\oplus}$', r'$log_{10}\ d_{orbit}$ [AU]',
                  r'$log_{10}\ d_{orbit, scaled}$ [AU/$\sqrt{L}$]',
                  r'$log_{10}\ M/M_{\oplus}$', r'$log_{10}\ d_{system}$ [pc]']
    labels = [r'Planet Radius [$R_{\oplus}$]', "Orbital Distance [AU]", r'Orbital Distance Scaled [AU/$\sqrt{L}$]',
              r'Planet Mass [$M_{\oplus}$]', "System Distance [pc]"]
    limits = [Rp_lim, d_orbit_lim, d_orbit_lim, Mp_lim, d_system_lim]
    limits_log = [Rp_lim_log, d_orbit_lim_log, d_orbit_lim_log, Mp_lim_log, d_system_lim_log]

    # Add the scaled orbit distance to the dataframe
    sample_data["rp_scaled"] = sample_data["rp"] / np.sqrt(sample_data["l_sun"])
    life_det["rp_scaled"] = life_det["rp"] / np.sqrt(life_det["l_sun"])
    exo_det["distance_p_scaled"] = exo_det["distance_p"] / np.sqrt(exo_det["L_star"])

    sample_data_log = sample_data.applymap(lambda x: np.log10(x) if isinstance(x, (int, float)) else x)
    life_det_log = life_det.applymap(lambda x: np.log10(x) if isinstance(x, (int, float)) else x)
    exo_det_log = exo_det.applymap(lambda x: np.log10(x) if isinstance(x, (int, float)) else x)

    one_corner_plot(life_det_log, parameters_life, labels_log, limits_log, save_path, "LIFEsim Detections log-scale")
    one_corner_plot(exo_det_log, parameters_exo, labels_log, limits_log, save_path, "EXOSIMS Detections log-scale")
    one_corner_plot(sample_data_log, parameters_life, labels_log, limits_log, save_path, "Sample Population log-scale")

    one_corner_plot(life_det, parameters_life, labels, limits, save_path, "LIFEsim Detections")
    one_corner_plot(exo_det, parameters_exo, labels, limits, save_path, "EXOSIMS Detections")
    one_corner_plot(sample_data, parameters_life, labels, limits, save_path, "Sample Population")

    return None


def one_corner_plot(data, parameters, labels, limits, save_path, title, DoS=False, sample_data=None):
    """
    This function generates a corner plot of the data given in the dataframe 'data' using the seaborn package.
    :param data: Dataframe containing the data to be plotted
    :param parameters: List of the parameters to be plotted
    :param labels: List of the labels of the parameters to be plotted
    :param limits: List of the limits of the parameters to be plotted
    :param save_path: Path where the results will be saved
    :param title: Title of the plot
    :param DoS: Boolean indicating if the plot shall also contain the Depth of Search
    :param sample_data: Dataframe containing the underlying sample population. Only needed if DoS=True
    :return:
    """
    for old_name, new_name in zip(parameters, labels):
        data.rename(columns={old_name: new_name}, inplace=True)

    sns.set_style("ticks")
    sns.despine()
    sns.color_palette("viridis")
    g = sns.pairplot(data, kind="kde", vars=labels, corner=True,
                     diag_kws=dict(fill=True, bw_adjust=0.5, cut=0.1),
                     plot_kws=dict(fill=True, bw_adjust=0.5, cut=0.1, cmap="viridis"))
    # No Upper at the moment, maybe later (DoS?)
    if DoS:
        g.map_upper(DoS_corner, color="black")

    for i in range(len(g.axes)):
        for j in range(i):
            g.axes[i][j].set_xlim(limits[j])
            if j != i:
                g.axes[i][j].set_ylim(limits[i])

    g.fig.suptitle(title)
    g.savefig(save_path.joinpath(title + ".svg"))

    return None


def DoS_corner(x, y, limit=None, **kwargs):
    """
    This function is used to plot the Depth of Search in the corner plots. It is used by the seaborn package.
    :param limit: Limit of the parameter space
    :param x: X-axis of the plot
    :param y: Y-axis of the plot
    :param kwargs: Additional arguments
    :return:
    """
    # Get the current axis
    ax = plt.gca()
    # Get the current data
    x = x.values
    y = y.values
    # Get the Limits
    xlim = limit[0]
    ylim = limit[1]

    xi, yi = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
    kde_sample = gaussian_kde([x_sampel, y_sampel])
    zi_sample = kde_sample(np.vstack([xi.flatten(), yi.flatten()]))
    kde_det = gaussian_kde([x, y])
    zi_det = kde_det(np.vstack([xi.flatten(), yi.flatten()]))

    threshold = 1e-4
    zi_sample[zi_sample < threshold * zi_sample.max()] = 0
    zi_det[zi_det < threshold * zi_det.max()] = 0

    with np.errstate(divide='ignore', invalid='ignore'):
        dos_cont = np.true_divide(zi_det, zi_sample)
        dos_cont[dos_cont == np.inf] = 0
        dos_cont = np.nan_to_num(dos_cont)

    # Make the plot

    return None


def plot_one_entire_mode(results_path, mode):
    """
    Runs through all the analysis for one mode
    :param results_path: Path stating where results shall be saved to
    :param mode: gives which mode shall be run
    :return:
    """
    # Import DataFrames
    exo_data, life_data = pd.read_csv(results_path.joinpath(mode + "_exo_data.csv")), pd.read_csv(
        results_path.joinpath(mode + "_life_data.csv"))
    results_path_mode = results_path.joinpath(mode)
    # Check for folder, if it does not already exist, create it
    if not os.path.exists(results_path_mode):
        os.makedirs(results_path_mode)

    print(results_path, results_path_mode)
    # Adjust Exo Data
    exo_data_det = gd.data_only_det(exo_data)
    # Adjust LIFE Data
    life_data_det = gd.data_only_det(life_data)
    # Only Earth-Like
    mask_life_EE = (life_data_det["radius_p"] > 0.5) & (life_data_det["radius_p"] < 2.5) & \
                   (life_data_det["Mp"] > 0.5) & (life_data_det["Mp"] < 12)
    mask_exo_EE = (exo_data_det["radius_p"] > 0.5) & (exo_data_det["radius_p"] < 2.5) & \
                  (exo_data_det["Mp"] > 0.5) & (exo_data_det["Mp"] < 12)
    life_data_det_EE = life_data_det[mask_life_EE]
    exo_data_det_EE = exo_data_det[mask_exo_EE]

    """
    Here the actual code of analyse_data.py is run. Different plots, Radius-Mass Doublechecks, the Depth of Search Analysis
    and so on. Feel free to add further functions and analysis tools in form of functions later on. 
    """
    # Plotting
    plots(life_data, exo_data, life_data_det, exo_data_det, results_path_mode)

    # Checking Mass-Radius Distribution via the Forecaster Git Code from J.Chen and D.Kipping 2016
    radius_mass_check(life_data, exo_data, life_data_det, exo_data_det, results_path_mode)

    # Depth of Search Analysis
    dos_analysis_naive(life_data, life_data_det, results_path_mode, "LIFEsim", N_bin=50)
    dos_analysis_naive(exo_data, exo_data_det, results_path_mode, "EXOSIMS", N_bin=50)

    # Corner Plots
    corner_plots(life_data, life_data_det, exo_data_det, results_path_mode)

    print("Analyse Data of mode: ", mode, " Finished!")
    return None


if __name__ == '__main__':
    """
    Pre-Amble Starts here with defining paths, and running the two Sims via "run_it_and_save_it"
    """
    # Define the modes you want to run, demo1 and 'all' should be equivalent, 'demo1' is simply for backwards
    # compatibility
    modes = ['det', 'non-det', 'char', 'all']

    current_dir = Path(__file__).parent.resolve()
    parent_dir = current_dir.parent.resolve()
    results_path = parent_dir.resolve().joinpath("Results/")

    # IF YOU ALREADY HAVE SIMULATION RESULTS OF BOTH LIFESIM AND EXOsim IN THE REQUIRED CSV FORMAT, YOU CAN COMMENT OUT
    run_it_and_save_it(results_path, modes=modes)

    """
    After Sims were run and saved (whether it happened during the same run or the results are already saved because the Sims
    were run earlier) we go into importing DataFrames and applying some masks to them to differentiate between detected
    planets, underlying population sample, Earth-Likes and so on
    """
    for mode in modes:
        plot_one_entire_mode(results_path, mode)
