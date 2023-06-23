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

parent_dir = Path(__file__).parents[1]
os.chdir(parent_dir.joinpath("EXOSIMS"))
sys.path.append(os.getcwd())

import EXOSIMS
import EXOSIMS.MissionSim

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def Rp_Mp_scatter(Rp_det, Mp_det, Rp, Mp, name, title, resultpath):
    """
    :param Rp_det: List planetary radii of detected planets
    :param Mp_det: List planetary mass of detected planets
    :param Rp: List planetary radii of all planets
    :param Mp: List planetary mass of all planets
    :return: Scatter plot with detected planets in blue and all planets in red
    """
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('Planet Radius (Earth radii)')
    plt.ylabel('Planet Mass (Earth masses)')
    # plt.xlim(0, 15)
    plt.ylim(0, 500)
    plt.scatter(Rp, Mp, c='grey', alpha=0.5, label='Undetected')
    plt.scatter(Rp_det, Mp_det, c='green', alpha=0.8, label='Detected')
    plt.legend()
    plt.savefig(resultpath + name + ".png")
    plt.clf()

    return 0


def Rp_d_scatter(Rp_det, d_det, Rp, d, name, title, resultpath):
    """
    :param Rp_det: List planetary radii of detected planets
    :param Mp_det: List planetary mass of detected planets
    :param Rp: List planetary radii of all planets
    :param p: List planetary period of all planets
    :return: Scatter plot with detected planets in blue and all planets in red
    """
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel('Planet Radius (Earth radii)')
    plt.ylabel('Planet distance (AU)')
    # plt.xlim(0, 15)
    plt.ylim(0, 50)
    plt.scatter(Rp, d, c='grey', alpha=0.5, label='Undetected')
    plt.scatter(Rp_det, d_det, c='green', alpha=0.8, label='Detected')
    plt.legend()
    plt.savefig(resultpath + name + ".png")
    plt.clf()

    return 0


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
    :param df: Dataframe from Exosims or LIFEsim
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
    Adds the HZ to the ExoSim dataframe read from the already calced HZ from the LIFEsim dataframe
    :param dfexo: ExoSim dataframe
    :param dflife: LIFEsim dataframe
    :return:
    """
    print("Started adding habitability data to DF of length: " + str(len(dfexo)))
    t1 = time.time()

    # Add habitable zone information to the ExoSim dataframe
    # Create a mapping dictionary for sname and corresponding HZ information
    hz_mapping = {}
    for _, row in dflife.iterrows():
        sname = row['name_s']
        hz_in = row['hz_in']
        hz_out = row['hz_out']
        hz_mapping[sname] = (hz_in, hz_out)

    # Update dfexo with HZ information using the mapping dictionary
    dfexo['hz_in'], dfexo['hz_out'] = zip(*dfexo['sname'].map(hz_mapping.get))

    # Add habitability boolean based on HZ data and orbit distance
    dfexo['habitable'] = (dfexo['distance_p'] > dfexo['hz_in']) & (dfexo['semimajor_p'] < dfexo['hz_out'])

    t2 = time.time()
    print("Finished adding habitability data. It took " + str(t2 - t1) + " seconds.")

    return dfexo


def detection_statistics(df, first_param, first_param_str, second_param, second_param_str):
    """
    Calculates the detection statistics for a given planet type and star type accounting for the "different" statistics
    of the binary nature of detections to sum over detections per universe and only then take the mean.
    :param df: dataframe either from ExoSims or LIFEsim
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
            std_dettime_life = life_dettime.std()
            mean_det_life, std_det_life = detection_statistics(df_life, first_param, first_param_str,
                                                               second_param, second_param_str)
            total_planets_life = len(df_life[life_mask])

            ################
            # Exosims Data #
            ################

            # ExoSims SNR is not broken down into SNR_1h. Exosim time  in unit of days, SNR scales with sqrt(t) !
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
            std_dettime_exo = exo_dettime.std()
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

            plt.title("SNR and Detections of LIFEsim and EXOsim", fontsize=22)
            fig.subplots_adjust(bottom=0.3)
            if save:
                savestring = result_path.joinpath(name + "_" + first_params[(pix * 3)] + ".pdf")  # *3 because hot, warm, cold
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
            ax1.set_ylabel('Detection Time [h]', fontsize=22)

            # Creating Manual Legend
            handles, labels = plt.gca().get_legend_handles_labels()
            patch1 = mpatches.Patch(color=color1, label='SNR')
            patch2 = mpatches.Patch(color=color2, label='Detections')
            handles.extend([patch1, patch2])

            plt.legend(handles=handles, fontsize=22)

            plt.title("SNR and Detections of LIFEsim and EXOsim", fontsize=22)
            fig.subplots_adjust(bottom=0.3)
            if save:
                savestring = result_path.joinpath(name + ".pdf")
                fig.savefig(savestring)
            plt.clf()
            if pix >= n_graphs - 1:
                break
    return None


def histogram_distribution_plot(life_data_d, exo_data_d, result_path, name, xlabel, xlim=None, ylim=None,
                                detected=False):
    """
    Plots the distribution of the given data from life and exosim into a histogram plot showing the
    distribution of the arbitrary value given as input. As LIFEsim and ExoSim data do not have the same total
    underlying amount of planets, the data is normalized to the total amount of planets in the respective data set.
    Additionally, the boolean parameter 'detected' can be set to True, in which case the additional information of the
    total amount of detected planets is added to the plot.
    :param life_data_d: Dataframe containing the LIFEsim data
    :param life_data_d: Dataframe containing the EXOsim data
    :param result_path: Path where the results should be saved
    :param name: name of the plot
    :param xlabel: label of the x-axis
    :param xlim: limits of the x-axis
    :param ylim: limits of the y-axis
    :param detected: boolean parameter if True additional total amount of detections is added to the plot
    :return: histogram plot showing the distribution of any value given as input
    """
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
                                           label='ExoSim')

    fig.text(0.5, 0.04, xlabel, ha="center", va="center", fontsize=10)
    axs[0].set_ylabel('probability density function', fontsize=10, labelpad=10)
    axs[0].axvline(life_data_d.mean(), color='r', linewidth=1)
    axs[1].axvline(exo_data_d.mean(), color='r', linewidth=1)

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
    axs[1].set_title('ExoSim', fontsize=12)

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


def kde_distribution_plot(data_d1, data_d2, result_path, name, xlabel, ylabel, xlim=None, ylim=None, detected=False):
    """
    Plots the kernel density estimator of the given data from life and exosim showing the
    distribution of the arbitrary value given as input. As LIFEsim and ExoSim data do not have the same total
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


# Run EXOsims in the given config as highlighted in Run_ExoSims.py and the input config file
rexo.__main__()
# Get the produced EXOsims data, convert it to LIFEsim and run LIFEsim with that according to the get_data.py code
gd.__main__()

current_dir = Path(__file__).parent.resolve()
exo_output_path = current_dir.joinpath("Analysis/Output/EXOSIMS")
life_output_path = current_dir.joinpath("Analysis/Output/LIFEsim")
stellar_table_path = current_dir.joinpath("Analysis/Populations/LIFEsim_StellarCat_Table.csv")
stellar_cat_path = current_dir.joinpath("Analysis/Populations/TargetList_exosims.csv")
lifesim_path = current_dir.parent.joinpath("LIFEsim-Rick_Branch")

results_path = current_dir.parent.resolve().joinpath("Results/")

# Planet and Stellar categories considered
ptypes = ["Hot Rocky", "Warm Rocky", "Cold Rocky",
          "Hot SE", "Warm SE", "Cold SE",
          "Hot Sub-Neptune", "Warm Sub-Neptune", "Cold Sub-Neptune",
          "Hot Sub-Jovian", "Warm Sub-Jovian", "Cold Sub-Jovian",
          "Hot Jovian", "Warm Jovian", "Cold Jovian"]
stypes = ["F", "G", "K", "M"]

# Import DataFrames
life_data_nop, exo_data_nop = gd.import_data(exo_output_path, life_output_path, "demo1.hdf5", stellar_cat_path)

# Add Planet Category
life_data = add_ptype_to_df(life_data_nop, "Kop2015")
exo_data_noHZ = add_ptype_to_df(exo_data_nop, "Kop2015")

# Add HZ of LIFesim to EXOsim Data
exo_data = add_HZ_to_exo(exo_data_noHZ, life_data)

# Adjust Exo Data
exo_data_det = gd.data_only_det(exo_data)
# Adjust LIFE Data
life_data_det = gd.data_only_det(life_data)

# Bar Plot
bar_cat_plot(life_data, exo_data, ptypes, 'ptype', stypes, 'stype', save=True, result_path=results_path,
             name="ptypes_stypes")

habitables = [True, False]
bar_cat_plot(life_data, exo_data, habitables, 'habitable', stypes, 'stype', save=True, result_path=results_path,
             name="habitable_stypes")
# Histogram of simulated and detected planet radii
histogram_distribution_plot(life_data["radius_p"].astype(float), exo_data["radius_p"].astype(float), results_path,
                            "Distribution Radius Planets", "Planet Radius [R_E]",
                            xlim=[0, 20], detected=False)
histogram_distribution_plot(life_data_det["radius_p"].astype(float), exo_data_det["radius_p"].astype(float), results_path,
                            "Distribution Radius Planets Detected Planets",
                            "Planet Radius [R_E]", xlim=[0, 20], detected=True)

histogram_distribution_plot(np.log10(life_data["radius_p"].astype(float)), np.log10(exo_data["radius_p"].astype(float)), results_path,
                            "log Distribution Radius Planets", "Planet Radius log [R_E]",
                            xlim=[-0.25, 2.5], detected=False)
histogram_distribution_plot(np.log10(life_data_det["radius_p"].astype(float)), np.log(exo_data_det["radius_p"].astype(float)), results_path,
                            "log Distribution Radius Planets Detected Planets",
                            "Planet Radius log [R_E]", xlim=[-0.25, 2.5], detected=True)

# Histogram of simulated and detected planet masses
histogram_distribution_plot(life_data["Mp"].astype(float), exo_data["Mp"].astype(float), results_path,
                            "Distribution Mass Planets", "Planet Mass [M_E]",
                            xlim=[0, 100], detected=False)
histogram_distribution_plot(life_data_det["Mp"].astype(float), exo_data_det["Mp"].astype(float), results_path,
                            "Distribution Mass Planets Detected Planets",
                            "Planet Mass [M_E]", xlim=[0, 100], detected=True)

histogram_distribution_plot(np.log10(life_data["Mp"].astype(float)), np.log10(exo_data["Mp"].astype(float)), results_path,
                            "log Distribution Mass Planets", "Planet Mass log [M_E]",
                            xlim=[-1, 3.5], detected=False)
histogram_distribution_plot(np.log10(life_data_det["Mp"].astype(float)), np.log10(exo_data_det["Mp"].astype(float)), results_path,
                            "log Distribution Mass Planets Detected Planets",
                            "log Planet Mass [M_E]", xlim=[-1, 3.5], detected=True)

# Histogram of simulated and detected planet orbital periods
histogram_distribution_plot(life_data["rp"].astype(float), exo_data["distance_p"].astype(float), results_path,
                            "Distribution Distance Planets", "Planet Distance [AU]",
                            xlim=[0, 20], detected=False)
histogram_distribution_plot(life_data_det["rp"].astype(float), exo_data_det["distance_p"].astype(float), results_path,
                            "Distribution Distance Detected Planets",
                            "Planet Distance [AU]",
                            xlim=[0, 5], detected=True)

histogram_distribution_plot(np.log10(life_data["rp"].astype(float)), np.log10(exo_data["distance_p"].astype(float)),
                            results_path, "log Distribution Distance Planets", "log Planet Distance [AU]",
                            xlim=[-2.5, 1.5], detected=False)
histogram_distribution_plot(np.log10(life_data_det["rp"].astype(float)), np.log10(exo_data_det["distance_p"].astype(float)), results_path,
                            "log Distribution Distance Detected Planets",
                            "log Planet Distance [AU]",
                            xlim=[-2.5, 1.5], detected=True)

# Distribution Plots Rp Mp
kde_distribution_plot(exo_data["radius_p"].astype(float), exo_data["Mp"].astype(float), results_path,
                      "KDE Distribution Exosims Rp-Mp Scatter Plot",
                      "Exoplanet Radius [R_E]", "Exoplanet Mass [M_E]",
                      xlim=[0, 15], ylim=[0, 200], detected=False)
kde_distribution_plot(life_data["radius_p"].astype(float), life_data["Mp"].astype(float), results_path,
                      "KDE Distribution LIFEsim Rp-Mp Scatter Plot",
                      "Exoplanet Radius [R_E]", "Exoplanet Mass [M_E]",
                      xlim=[0, 15], ylim=[0, 200], detected=False)
kde_distribution_plot(exo_data_det["radius_p"].astype(float), exo_data_det["Mp"].astype(float), results_path,
                      "KDE Distribution Exosims Detected Rp-Mp Scatter Plot",
                      "Exoplanet Radius [R_E]", "Exoplanet Mass [M_E]",
                      xlim=[0, 15], ylim=[0, 200], detected=True)
kde_distribution_plot(life_data_det["radius_p"].astype(float), life_data_det["Mp"].astype(float), results_path,
                      "KDE Distribution LIFEsim Detected Rp-Mp Scatter Plot",
                      "Exoplanet Radius [R_E]", "Exoplanet Mass [M_E]",
                      xlim=[0, 15], ylim=[0, 200], detected=True)

# Distribution Plots log(Rp) log(Mp)
kde_distribution_plot(np.log10(exo_data["radius_p"].astype(float)), np.log10(exo_data["Mp"].astype(float)),
                      results_path, "KDE Distribution Exosims log(Rp)-log(Mp) Scatter Plot",
                      "log_10 Exoplanet Radius [R_E]", "log_10 Exoplanet Mass [M_E]",
                      xlim=[-0.25, 1.25], ylim=[-1, 2.5], detected=False)
kde_distribution_plot(np.log10(life_data["radius_p"].astype(float)), np.log10(life_data["Mp"].astype(float)),
                      results_path, "KDE Distribution LIFEsim log(Rp)-log(Mp) Scatter Plot",
                      "log_10 Exoplanet Radius [R_E]", "log_10 Exoplanet Mass [M_E]",
                      xlim=[-0.25, 1.25], ylim=[-1, 2.5], detected=False)
kde_distribution_plot(np.log10(exo_data_det["radius_p"].astype(float)), np.log10(exo_data_det["Mp"].astype(float)),
                      results_path, "KDE Distribution Exosims Detected log(Rp)-log(Mp) Scatter Plot",
                      "log_10 Exoplanet Radius [R_E]", "log_10 Exoplanet Mass [M_E]",
                      xlim=[-0.25, 1.25], ylim=[-1, 2.5], detected=True)
kde_distribution_plot(np.log10(life_data_det["radius_p"].astype(float)), np.log10(life_data_det["Mp"].astype(float)),
                      results_path, "KDE Distribution LIFEsim Detected log(Rp)-log(Mp) Scatter Plot",
                      "log_10 Exoplanet Radius [R_E]", "log_10 Exoplanet Mass [M_E]",
                      xlim=[-0.25, 1.25], ylim=[-1, 2.5], detected=True)

# Distribution Plots Orbit Mp
kde_distribution_plot(exo_data["distance_p"].astype(float), exo_data["Mp"].astype(float), results_path,
                      "KDE Distribution Exosims Orbit-Mp Scatter Plot",
                      "Exoplanet Distance [AU]", "Exoplanet Mass [M_E]",
                      xlim=[0, 20], ylim=[0, 200], detected=False)
kde_distribution_plot(life_data["rp"].astype(float), life_data["Mp"].astype(float), results_path,
                      "KDE Distribution LIFEsim Orbit-Mp Scatter Plot",
                      "Exoplanet Distance [AU]", "Exoplanet Mass [M_E]",
                      xlim=[0, 20], ylim=[0, 200], detected=False)
kde_distribution_plot(exo_data_det["distance_p"].astype(float), exo_data_det["Mp"].astype(float), results_path,
                      "KDE Distribution Exosims Detected Orbit-Mp Scatter Plot",
                      "Exoplanet Distance [AU]", "Exoplanet Mass [M_E]",
                      xlim=[0, 5], ylim=[0, 200], detected=True)
kde_distribution_plot(life_data_det["rp"].astype(float), life_data_det["Mp"].astype(float), results_path,
                      "KDE Distribution LIFEsim Detected Orbit-Mp Scatter Plot",
                      "Exoplanet Distance [AU]", "Exoplanet Mass [M_E]",
                      xlim=[0, 5], ylim=[0, 200], detected=True)

# Distribution Plots log(Orbit) log(Mp)
kde_distribution_plot(np.log10(exo_data["distance_p"].astype(float)), np.log10(exo_data["Mp"].astype(float)),
                      results_path, "KDE Distribution Exosims log(d)-log(Mp) Scatter Plot",
                      "log_10 Exoplanet Orbit [AU]", "log_10 Exoplanet Mass [M_E]",
                      xlim=[-2, 2], ylim=[-1, 2.5], detected=False)
kde_distribution_plot(np.log10(life_data["rp"].astype(float)), np.log10(life_data["Mp"].astype(float)),
                      results_path, "KDE Distribution LIFEsim log(d)-log(Mp) Scatter Plot",
                      "log_10 Exoplanet Orbit [AU]", "log_10 Exoplanet Mass [M_E]",
                      xlim=[-2, 2], ylim=[-1, 2.5], detected=False)
kde_distribution_plot(np.log10(exo_data_det["distance_p"].astype(float)), np.log10(exo_data_det["Mp"].astype(float)),
                      results_path, "KDE Distribution Exosims Detected log(d)-log(Mp) Scatter Plot",
                      "log_10 Exoplanet Orbit [AU]", "log_10 Exoplanet Mass [M_E]",
                      xlim=[-2, 2], ylim=[-1, 2.5], detected=True)
kde_distribution_plot(np.log10(life_data_det["rp"].astype(float)), np.log10(life_data_det["Mp"].astype(float)),
                      results_path, "KDE Distribution LIFEsim Detected log(d)-log(Mp) Scatter Plot",
                      "log_10 Exoplanet Orbit [AU]", "log_10 Exoplanet Mass [M_E]",
                      xlim=[-2, 2], ylim=[-1, 2.5], detected=True)
print("Analyse Data Finished!")
