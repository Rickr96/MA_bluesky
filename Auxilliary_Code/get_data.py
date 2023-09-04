import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import pickle
from astropy import units as u
from astropy.constants import G
from scipy.special import jn
import scipy as sp
import time
import subprocess
from pathlib import Path
import multiprocessing as mp
from astropy.units import Quantity

# Definition Constants
sigma = 5.670367 * 10 ** (-8)  # Stefan-Boltzmann constant in W m^-2 K^-4
L_sun = 3.828 * 10 ** 26  # Solar luminosity in W
R_sun = 6.957 * 10 ** 8  # Solar radius in m
M_sun = 1.989 * 10 ** 30  # Solar mass in kg
R_Earth = 6371 * 10 ** 3  # Earth radius in m
Flux_E = 1361.  # W m-2(theoretically = 1373 but Jens used 1361) Solar constant (= incident Flux on Earth)


# from (https://www.sciencedirect.com/topics/earth-and-planetary-sciences/solar-flux)


def planet_incident_flux_from_L(L, d):
    """
    Calculates the incident flux of the host star on the planet.
    :param L: Host Star Luminosity
    :param d: Distance between host star and planet
    :return: Incident Flux on the planet in units of Earth incident Flux
    """
    # https://en.wikipedia.org/wiki/Luminosity#Luminosity_formulae
    # Fluxdensity = L / Area of distribution of sunlight (sphere with radius dist_planet)

    # ATTENTION: Even though LIFEsim calls it incident Flux, it is actually the Flux DENSITY (W m^-2)
    L *= L_sun  # L [L_sun] --> L [W]
    Fp = L / (4 * np.pi * d.to(u.m) ** 2)  # Incident flux on the planet
    Fp /= Flux_E
    Fp *= u.m ** 2  # Remove unit m^-2 from Flux density

    return Fp


def planet_incident_flux_from_T(T, Rs, d):
    """
    Calculates the incident flux of the host star on the planet.
    :param T: Host Star Temperature
    :param Rs: stellar radius
    :param d: Distance between host star and planet
    :return: Incident Flux on the planet in units of Earth incident Flux
    """
    L = 4 * np.pi * (Rs * R_sun) ** 2 * sigma * T ** 4  # Luminosity of the host star

    # ATTENTION: Even though LIFEsim calls it incident Flux, it is actually the Flux DENSITY (W m^-2)
    Fp = L / (4 * np.pi * d.to(u.m) ** 2)  # Incident flux on the planet
    Fp /= Flux_E  # Incident flux on the planet in units of Earth incident Flux
    Fp *= u.m ** 2  # Remove unit m^-2 from Flux density

    return Fp


def skycoord_string_to_tuple(skycoord_string):
    """
    Converts the skycoord string from the star catalogue to a tuple of floats. String has the form
    '<SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, pc)\n    (0.168286, -69.67580414, 29.87)>'

    :param skycoord_string:
    :return: tuple of floats with (ra, dec, distance) of host star
    """
    dummy1 = skycoord_string.split("\n")
    dummy2 = dummy1[1].split("(", 1)
    dummy3 = dummy2[1].split(")", 1)
    result_tup = tuple(map(float, dummy3[0].split(",")))

    return result_tup


def mean_to_true_anomaly(ecc, mean_anomaly, order=100):
    """
    Calculates the true anomaly from the mean anomaly. Based on formula from Battin, R.H. (1999) found through
    Wikipedia. Mathematical formula adapted to Code with the help of ChatGPT.
    :param ecc: Eccentricity of the orbit
    :param mean_anomaly: Mean anomaly of the orbit
    :return: True anomaly of the orbit
    """
    # Initialize summ
    summ = 0.0
    # Precompute beta
    beta = ecc / (1 + np.sqrt(1 - ecc ** 2))
    t1 = time.time()
    # Compute nu
    for k in range(1, order):
        # Compute inner sum over n
        inner_sum = 0.0
        for n in range(-int(order / 10), int(order / 10) + 1):
            inner_sum += jn(n, -k * ecc) * beta ** (abs(k + n))

        # Add to outer sum
        summ += (1.0 / k) * inner_sum * np.sin(k * mean_anomaly)

    # Compute final value of v
    nu = mean_anomaly + 2.0 * summ * u.deg
    t2 = time.time()
    print("Time to compute nu_array: ", t2 - t1)

    return nu


def calc_max_ang_sep(a, d, e, w, inc):
    """
    Calculates the maximum angular separation between the star and the planet. By searching for the extrema of the
    formulas provided by the Paper "Maximum Angular Separation Epochs for Exoplanet Imaging Observations Stephen R.
    Kane 1 , Tiffany Meshkat 2"
    :param a: semi-major axis
    :param d: distance to the star
    :param e: eccentricity
    :param w: argument of periastron == periapsis but for this concrete system
    :param inc: inclination
    :return: max Angular Separation
    """
    w = w.to(u.rad)
    inc = inc.to(u.rad)
    # Function of angular separation as a function of true anomaly theta
    f = lambda theta: a / (d * u.parsec) * ((1 - e ** 2) / (1 + e * np.cos(theta))) * \
                      np.sqrt(np.cos(w + theta * u.rad) ** 2 + np.sin(w + theta * u.rad) ** 2 * np.cos(inc) ** 2)
    # Derivative of the above formula with respect to theta
    df = lambda theta: (-1) * ((a * (e - 1) ** 2) / (d * (1 + e * np.cos(theta * u.rad)) ** 2)) * \
                       1 / np.sqrt(np.cos(inc) ** 2 * np.sin(w + theta * u.rad) ** 2 + np.cos(w + theta * u.rad) ** 2) * \
                       (e * np.cos(inc) ** 2 * np.sin(theta) * np.sin(w + theta * u.rad) ** 2 +
                        ((e * np.cos(inc) ** 2 - e) * np.cos(theta) + np.cos(inc) ** 2 - 1) *
                        np.cos(w + theta * u.rad) * np.sin(w + theta * u.rad) + e * np.sin(theta) * np.cos(
                                   w + theta * u.rad) ** 2)
    # Maximize the above function based on different priors for the
    # true anomaly of the drawn planets.
    x0 = np.linspace(np.pi / 5, 2 * np.pi, 5) * u.rad
    xx = sp.optimize.fsolve(df, x0)
    ff = f(xx)

    maxAngSep = np.max(ff)  # au
    maxAngSep = maxAngSep

    return maxAngSep


def stype_from_life_table(data_path, L0):
    """
    Calculates the spectral type of the star based on the luminosity of the star. Uses the lookup table used in LIFEsim
    Attention: Code Assumes that data is sorted in descending order of Luminosity
    :param data_path: path to csv lookup table.
    :param L0: Luminosity of the star in L_sun
    :return: string with stype of the star
    """
    data = pd.read_csv(data_path)[14:49]  # Remove everything above A0 and below M8
    # Need to fix data type of Luminosity
    stype = ''
    for i, row in data.iterrows():
        if L0 >= float(row['Luminosity (in solar luminosities)']):
            stype = row['Spectral Type']
            break

    return stype


def fit_Ts_to_lum(data_path):
    """
    Fits the temperature of the star to the luminosity of the star. Underlying approximation is a linear log-log fit
    :param data_path: path to csv lookup table
    :return: starting point L0 and slope m of the linear fit of log-log
    """
    data = pd.read_csv(data_path)[14:49]  # Remove everything above A0 and below M8
    # Fit a linear function to the log-log data
    L = []
    Ts = []
    for i, val in enumerate(data['Luminosity (in solar luminosities)']):
        L_dummy = float(val)
        L.append(L_dummy)
    for i, val in enumerate(data['Temperature (K)']):
        Ts_dummy = float(val)
        Ts.append(Ts_dummy)
    logL = np.log(L)
    logTs = np.log(Ts)
    m, b = np.polyfit(logL, logTs, 1)

    return b, m


def data_only_det(dataframe):
    """
    :param dataframe: Dataframe containing the LIFEsim data
    :return: Dataframe containing only the detected planets
    """
    # 1 Means detected; 2 means characterized, -1 means inside IWA, -2 means outside OWA --> -1 and -2 are filtered out
    # even before coming here
    mask = (dataframe['detected'] == 1) | (dataframe['detected'] == 2)
    dataframe = dataframe[mask]
    return dataframe


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


def life_data_remove_m(dataframe):
    """
    :param dataframe: Dataframe containing the LIFEsim data
    :return: Dataframe containing only the planets without M-stars
    """
    dataframe = dataframe[dataframe['stype'] == stype_translator("M")]
    return dataframe


def planet_hist(R_p, M_p, name, title):
    """
    Create two histograms of planetary radii (R_p) and planetary mass (M_p)
    and save them with the given name.

    :param R_p: List of planetary radii.
    :param M_p: List of planetary masses.
    :param name: String containing the name to use when saving the histograms.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)
    # Create histogram for planetary radii
    ax1.hist(R_p, bins=100, color='b', alpha=0.5)
    ax1.set_xlabel('Planetary radius (Earth radii)')
    ax1.set_ylabel('Number of planets')
    ax1.set_title('Planetary Radii Histogram')

    # Create histogram for planetary mass
    ax2.hist(M_p, bins=5000, color='g', alpha=0.5)
    ax2.set_xlim(0, 500)
    ax2.set_xlabel('Planetary mass (Earth masses)')
    ax2.set_ylabel('Number of planets')
    ax2.set_title('Planetary Mass Histogram')

    # Save the histograms
    plt.savefig("C:/Users/Rick/OneDrive/ETH/_MA/Auxilliary_Code/Analysis/Figures/" + name + '.png')
    plt.show()


def filter_exo_res(res, det_status):
    """
    Function that shall filter the ExoSim results based on the detection status of the planets. det_status shows for
    what should be filtered.
    Note: for ppop_exo_to_life the DRM is not required, thus we omitt that the DRM is filtered as well and just
    return a filtered res[systems]
    :param res: results from ExoSim
    :param det_status: string that shows what should be filtered: 0 = only non-detected planets, 1 = only detected
    planets, 2 = characterized planets
    :return:
    """
    res_filtered = dict()
    filtered_systems = dict()
    p_index_storage = []
    for row in res["DRM"]:
        if "char_status" in row.keys():
            row["det_status"] = 2 * row["char_status"]
        if det_status == 0:
            for ix, det in enumerate(row["det_status"]):
                if det == det_status or det == -1 or det == -2:
                    p_index_storage.append(row["plan_inds"][ix])
        else:
            for ix, det in enumerate(row["det_status"]):
                if det == det_status:
                    p_index_storage.append(row["plan_inds"][ix])

    for key in res["systems"].keys():
        if key != 'MsTrue' and key != 'MsEst':
            values = [
                res["systems"][key][i].value if isinstance(res["systems"][key][i], Quantity) else res["systems"][key][i]
                for i in p_index_storage]
            filtered_systems[key] = np.array(values) * res["systems"][key].unit if isinstance(res["systems"][key][0],
                                                                                              Quantity) \
                else np.array(values)
        else:
            filtered_systems[key] = res["systems"][key]
    res_filtered["systems"] = filtered_systems
    res_filtered["DRM"] = res["DRM"]
    res_filtered["seed"] = res["seed"]

    return res_filtered


def pop_exo_to_life(pickle_path, star_cat_path, stellar_table_path, mode='demo1'):
    """
    Function that shall convert the ExoSim population to the LIFEsim format. The names of the columns are
    named after the name of the data-catalogue class in LIFEsim. The columns are:
    LIFEsim format:
    'nuniverse', 'Rp', 'Porb', 'Mp', 'ep', 'ip', 'Omegap', 'omegap', 'thetap', 'abond',
    'ageomvis', 'ageommir', 'z', 'semimajor_p', 'radius_p', 'angsep', 'maxangsep', 'Fp', 'fp', temp_p', 'Nstar',
    'radius_s', 'Ms', 'temp_s', 'distance_s', 'stype', 'ra', 'dec', name_s
    exo_ppop format:
    'a', 'e', 'I', 'O', 'w','M0', 'Mp', 'mu', 'Rp', 'p', 'plan2star', 'star', 'nEZ', 'MsTrue', 'MsEst', WA, d
    Star catalog format:
    'Name', 'Spec',	'parx', 'Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag', 'Jmag', 'Hmag', 'Kmag', 'dist',
    'BV', 'MV', 'BC', 'L', 'coords', 'pmra', 'pmdec', 'rv', 'Binary_Cut', 'MsEst', 'MsTrue', 'int_comp'

    :param: pickle_path: path to the pickle file containing the ExoSim population
    :param: star_cat_path: path to the star catalog
    :return:
    """
    # Create empty dataframe
    df_life = pd.DataFrame(columns=['fname',
                                    'nuniverse',
                                    'pindex',
                                    'Porb',
                                    'semimajor_p',
                                    'rp',
                                    'radius_p',
                                    'Mp',
                                    'ep',
                                    'ip',
                                    'Omegap',
                                    'omegap',
                                    'thetap',
                                    'abond',
                                    'ageomvis',
                                    'ageommir',
                                    'z',
                                    'angsep',
                                    'maxangsep',
                                    'Fp',
                                    'fp',
                                    'temp_p',
                                    'nstar',
                                    'radius_s',
                                    'Ms',
                                    'temp_s',
                                    'distance_s',
                                    'stype',
                                    'ra',
                                    'dec',
                                    'name_s'])

    star_cat = pd.read_csv(star_cat_path, sep=',')

    # Calculate Luminosity to Temperature Relations based on lookup table
    T0, m_Ts = fit_Ts_to_lum(stellar_table_path)

    # Read Data from Pickle Files
    pindex_max = 0
    pklfiles = glob.glob(os.path.join(pickle_path, "*.pkl"))
    for counter, f in enumerate(pklfiles):
        with open(f, "rb") as g:
            res = pickle.load(g, encoding="latin1")
        # Filter res based on mode
        if mode == 'all' or mode == 'demo1':
            print("Passing all planets to LIFEsim")
        elif mode == 'det':
            # Only feed the detected planets to LIFEsim
            res = filter_exo_res(res, 1)
            print("Passing only detected planets to LIFEsim")
        elif mode == 'non-det':
            res = filter_exo_res(res, 0)
            print("Passing only non-detected planets to LIFEsim")
        elif mode == 'char':
            res = filter_exo_res(res, 2)
            print("Passing only characterized planets to LIFEsim")
        else:
            raise ValueError("Mode not recognized")
        # Planet Index
        pindex = list(np.arange(pindex_max, pindex_max + len(res["systems"]['plan2star'])))
        if len(pindex) == 0:
            continue
        else:
            pindex_max = pindex[-1] + 1

        # Planet Radius
        Rp = res["systems"]['Rp']

        # Planet Mass
        Mp = res["systems"]['Mp']

        # Semi-major Axis
        a = res["systems"]['a']

        # Distance planet to star
        # Appears that d is sometimes greater than a in EXOsims --> Makes sense for eccentric orbits
        # Star not in center but focal point!!
        rp = res["systems"]['d']

        # Orbital Eccentricity
        ep = res["systems"]['e']

        # Orbital Inclination
        ip = res["systems"]['I']

        # Longitude of Ascending Node
        Omegap = res["systems"]['O']

        # Argument of perigee
        omegap = res["systems"]['w']

        # Initializing all lists that contain or are calculated with stellar parameters
        Porb, Ms, Ds, Stype_ext, Stype, RA, Dec, Ts, Rs, Ls = [], [], [], [], [], [], [], [], [], []
        name_s = []
        for count, sname in enumerate(res["systems"]['star']):
            #############################
            # Calculate Everything
            #############################
            name_s.append(sname)

            M = star_cat.loc[star_cat['Name'] == sname, 'MsTrue'].iloc[0]

            # Stellar Distance
            D = star_cat.loc[star_cat['Name'] == sname, 'dist'].iloc[0]

            # Stellar Luminosity as a helpful parameter -- not needed for LIFEsim
            L = star_cat.loc[star_cat['Name'] == sname, 'L'].iloc[0]

            # Calculate stellar Temperature based on LIFEsim lookup table through Luminosity
            logTs = T0 + m_Ts * np.log(L)
            Ts_dummy = np.exp(logTs)

            # Solar Type ExoSim and LIFEsim
            Exo_stype_str = star_cat.loc[star_cat['Name'] == sname, 'Spec'].iloc[0]

            # Sky Coordinates of the star (right ascension and declination)
            skycoord_tuple = skycoord_string_to_tuple(star_cat.loc[star_cat['Name'] == sname, 'coords'].iloc[0])
            ra = skycoord_tuple[0]
            dec = skycoord_tuple[1]

            # Calculate stellar radius based on Ts and L
            R = np.sqrt((L * L_sun) / (4 * np.pi * sigma * (Ts_dummy ** 4))) / R_sun

            #############################
            # Append Everything
            #############################
            Porb.append((2 * np.pi * np.sqrt(((a[count]) ** 3) / (G * (M * u.Msun)))).to(u.day))
            Ms.append(M)
            Ds.append(D)
            RA.append(ra)
            Dec.append(dec)
            Rs.append(R)
            Ts.append(Ts_dummy)
            Stype_ext.append(Exo_stype_str)
            Stype.append(Exo_stype_str[0])
            Ls.append(L)

        # Planet True Anomaly
        # 'M0' is the initial (at mission start) mean anomaly
        # true anomaly can be calculated directly from the mean anomaly via a Fourier expansion Battin, R.H. (1999)
        thetap = mean_to_true_anomaly(ep, res["systems"]['M0'], order=100)

        # ExoSim Geometric Albedo to LIFEsim Bond Albedo
        # Assuming Lambertian Scatterer, the Bond Albedo is 2/3 of the Geometric Albedo (See Documentation for why)
        Abond = 2 / 3 * res["systems"]['p']

        # AgeomVIS and AgeomMIR from Ageo general
        # Same weird prefactors as J.Kammerer used in P-Pop in his random.rand() sampling of the geometric albedo.
        AgeomVIS = 0.6 * res["systems"]['p']
        AgeomMIR = 0.1 * res["systems"]['p']

        # Lifesm z: Surface Birghtness Exozodiacal Light[unit in local zodiacal dust], Exosim nEZs same
        z = res["systems"]['starnEZ']

        # Angular Separation
        AngSep = res["systems"]['WA']

        """
        Two independent things provided me with the following workaround: 1) The the formulas provided by the paper 
        "Maximum Angular Separation Epochs for Exoplanet Imaging Observations Stephen R. Kane 1 , Tiffany Meshkat 2" 
        provided formulas that one could maximize in order to obtain the maximum angular separation. 2) Looking at 
        the code in P-Pop by J.Kammerer, this is exactly what he did!
        """
        t1 = time.time()
        # Maximum Angular Separation
        maxAngSep = []

        # Planet incident host star flux -- meaning not yet corrected for Bond Albedo
        FP = []
        for ix, w in enumerate(omegap):
            # Maximum Angular Separation
            # maxAngSep.append(calc_max_ang_sep(a[ix], Ds[ix], ep[ix], w, ip[ix]))
            # TODO not used in LIFEsim just use normal AngSep
            maxAngSep.append(AngSep[ix])

            # Planet incident host star flux
            FP.append(planet_incident_flux_from_L(Ls[ix], rp[ix]))
        t2 = time.time()
        print("Time to compute maxAngSep (and planet incident Flux): ", t2 - t1)

        # Lambertian Reflectance directly from P-Pop Code
        alphap = np.arccos(-np.sin(ip) * np.sin(omegap + thetap))  # rad
        fp = np.abs((np.sin(alphap) + (np.pi - alphap / u.rad) * np.cos(alphap)) / np.pi)

        # Planet equilibrium Temperature
        coeff = (Flux_E / (4. * sigma))
        Tp = (FP * (1 - Abond) * coeff) ** (1 / 4)

        # Placeholder nstar for later
        nstar = np.zeros(len(Rs))

        planet_df = pd.DataFrame({'fname': f,
                                  'nuniverse': counter,
                                  'pindex': pindex,
                                  'Porb': Porb,
                                  'semimajor_p': a,
                                  'rp': rp,
                                  'radius_p': Rp,
                                  'Mp': Mp,
                                  'ep': ep,
                                  'ip': ip,
                                  'Omegap': Omegap,
                                  'omegap': omegap,
                                  'thetap': thetap,
                                  'abond': Abond,
                                  'ageomvis': AgeomVIS,
                                  'ageommir': AgeomMIR,
                                  'z': z,
                                  'angsep': AngSep,
                                  'maxangsep': maxAngSep,
                                  'Fp': FP,
                                  'fp': fp,
                                  'temp_p': Tp,
                                  'nstar': nstar,
                                  'radius_s': Rs,
                                  'Ms': Ms,
                                  'temp_s': Ts,
                                  'distance_s': Ds,
                                  'stype': Stype,
                                  'stype_ext': Stype_ext,
                                  'ra': RA,
                                  'dec': Dec,
                                  'name_s': name_s})
        df_life = pd.concat([df_life, planet_df], ignore_index=True)
        print(counter + 1, "/", len(pklfiles), " are Done!")

    s_names_uniq = list(set(df_life['name_s']))
    nstar = []
    for ix, sname in enumerate(df_life['name_s']):
        nstar.append(s_names_uniq.index(sname))
    df_life['nstar'] = nstar
    return df_life


def import_data(exo_outpath, life_outpath, life_file_name, star_cat_path):
    """
    Imports the data from the LIFEsim and EXOSIM output files
    :param exo_outpath: exosim output path
    :param life_outpath: life output path
    :param life_file_name: life output file name
    :param star_cat_path: star catalog path
    :return: dataframes of life and exosim
    """

    # star cat for help
    star_cat = pd.read_csv(star_cat_path, sep=',')

    # Import LIFEsim
    print("Start LIFEsim Import")
    life_data_d = pd.read_hdf(life_outpath.joinpath(life_file_name))

    # Import Exosims
    print("Start EXOsim Import")
    pklfiles = glob.glob(os.path.join(exo_outpath, "*.pkl"))
    df_exo = pd.DataFrame(columns=['fname',
                                   'nuniverse',
                                   'sname',
                                   'stype',
                                   'pindex',
                                   'detected',
                                   'detSNR',
                                   'charSNR',
                                   'dettime',
                                   'chartime',
                                   'radius_p',
                                   'semimajor_p',
                                   'Ageo',
                                   'ecc',
                                   'Mp',
                                   'WAs',
                                   'dMags',
                                   'distance_p',
                                   'fEZ',
                                   'fZ',
                                   'tottime',
                                   'Fp'])

    for counter, f in enumerate(pklfiles):
        with open(f, "rb") as g:
            res = pickle.load(g, encoding="latin1")
        pinds, dets, WAs, dMag, rs, fEZ, fZs, dettime, chartime, tottime = [], [], [], [], [], [], [], [], [], []
        det_SNR, char_SNR, radius_p, semi_major, Ageo, ecc, Mp, FP = [], [], [], [], [], [], [], []
        star_name, stypes, Ds, L = [], [], [], []
        for row in res["DRM"]:
            if "char_status" in row.keys():
                row["det_status"] = (2) * row["char_status"]
            for ix, det in enumerate(row["det_status"]):
                # Stellar Parameters
                starname = row["star_name"]
                star_name.append(starname)

                # Star Distance
                D = star_cat.loc[star_cat['Name'] == starname, 'dist'].iloc[0]
                Ds.append(D)

                # Stellar Type
                stypes.append(star_cat.loc[star_cat['Name'] == starname, 'Spec'].iloc[0])
                # Planet Parameters

                # Planet Index
                pinds.append(row["plan_inds"][ix])

                # Detected or not 1 = detected, 0 = not detected, -1 inside IWA, -2 Outside OWA, 2 characteirzed
                # We don't care about IWA, OWA, just set it to 0
                if det == -1 or det == -2:
                    dets.append(0)
                else:
                    dets.append(det)

                # Working Angle
                if "det_params" in row.keys():
                    WAs.append(row["det_params"]["WA"][ix].to("arcsec").value)
                elif "char_params" in row.keys():
                    WAs.append(row["char_params"]["WA"][ix].to("arcsec").value)

                # Magnitude planet vs star ratio
                if "det_params" in row.keys():
                    dMag.append(row["det_params"]["dMag"][ix])
                elif "char_params" in row.keys():
                    dMag.append(row["char_params"]["dMag"][ix])

                # Distance planet to host star
                if "det_params" in row.keys():
                    distance_p = row["det_params"]["d"][ix].to("AU").value
                elif "char_params" in row.keys():
                    distance_p = row["char_params"]["d"][ix].to("AU").value
                rs.append(distance_p)

                # Exozodi brightness
                if "det_params" in row.keys():
                    fEZ.append(row["det_params"]["fEZ"][ix].value)
                elif "char_params" in row.keys():
                    fEZ.append(row["char_params"]["fEZ"][ix].value)
                # Zodiacal brightness
                if "det_params" in row.keys():
                    fZs.append(row["det_fZ"].value * len(np.where(row["det_status"] == 1)))
                elif "char_params" in row.keys():
                    fZs.append(row["char_fZ"].value * len(np.where(row["det_status"] == 1)))

                # Detection and Characterization time
                if "det_time" in row.keys():
                    det_time = row["det_time"].value
                else:
                    det_time = 0

                if "char_time" in row.keys():
                    char_time = row["char_time"].value
                else:
                    char_time = 0
                dettime.append(det_time)
                chartime.append(char_time)
                tottime.append(det_time + char_time)

                # Detection and characterization SNR
                if "det_SNR" in row.keys():
                    detSNR = row["det_SNR"][ix]
                else:
                    detSNR = 0

                if "char_SNR" in row.keys():
                    charSNR = row["char_SNR"][ix]
                else:
                    charSNR = 0
                det_SNR.append(detSNR)
                char_SNR.append(charSNR)
                # Planet Radius, Semi-major axis, Geometric Albedo, Eccentricity, Mass
                radius_p.append(res["systems"]["Rp"][row["plan_inds"][ix]] / u.R_earth)
                semi_major.append(res["systems"]["a"][row["plan_inds"][ix]].to("AU").value)
                Ageo.append(res["systems"]["p"][row["plan_inds"][ix]])
                ecc.append(res["systems"]["e"][row["plan_inds"][ix]])
                Mp.append(res["systems"]["Mp"][row["plan_inds"][ix]] / u.M_earth)
                # Incident Flux density for which stellar luminosity and planet distance are needed
                L_dummy = star_cat.loc[star_cat['Name'] == starname, 'L'].iloc[0]
                L.append(L_dummy)
                FP.append(planet_incident_flux_from_L(L_dummy, distance_p * u.AU))

        planet_df = pd.DataFrame({'fname': f,
                                  'nuniverse': counter,
                                  'sname': star_name,
                                  'stype': stypes,
                                  'distance_s': Ds,
                                  'L_star': L,
                                  'pindex': pinds,
                                  'detected': dets,
                                  'detSNR': det_SNR,
                                  'charSNR': char_SNR,
                                  'dettime': dettime,
                                  'chartime': chartime,
                                  'radius_p': radius_p,
                                  'semimajor_p': semi_major,
                                  'Ageo': Ageo,
                                  'ecc': ecc,
                                  'Mp': Mp,
                                  'WAs': WAs,
                                  'dMags': dMag,
                                  'distance_p': rs,
                                  'fEZ': fEZ,
                                  'fZ': fZs,
                                  'tottime': tottime,
                                  'Fp': FP
                                  })
        # Because of Multivisits in ExoSim, we need to merge duplicates accordingly. Adding detection times, taking
        # the maximum from the SNRs, keeping planetary parameters the same and so on
        agg_funcs = {
            'detSNR': 'max',  # Take the maximum SNR
            'charSNR': 'max',  # Take the maximum SNR
            'tottime': 'sum',  # Add the total times
            'dettime': 'sum',  # Add the detection times
            'chartime': 'sum',  # Add the characterization times
            'WAs': 'max',  # Take the maximum WA
            'dMags': 'min',  # Take the minimum dMag
            'fZ': 'sum',  # Add the zodiacal brightness
            'detected': 'max',  # If there is a 2 in detected, then
            # the planet is characterized, if there is a 1 in detected, then the planet is detected, else it is not
            # Add other columns here...
        }
        # Group by 'nuniverse', 'sname', and 'pindex', and apply aggregation functions
        merged_df = planet_df.groupby(['nuniverse', 'sname', 'pindex']).agg(agg_funcs).reset_index()

        # Merge the columns 'stype', 'radius_p', 'semimajor', etc.
        # Assuming these columns have the same values within each group
        cols_to_merge = ['fname', 'stype', 'radius_p', 'semimajor_p', 'Ageo', 'ecc', 'Mp', 'distance_p', 'fEZ', 'Fp', 'distance_s',
                         'L_star']
        merged_df[cols_to_merge] = \
            planet_df.groupby(['nuniverse', 'sname', 'pindex'])[cols_to_merge].first().reset_index(drop=True)
        df_exo = pd.concat([df_exo, merged_df], ignore_index=True)

        del res

    return life_data_d, df_exo


def run_life(pythonpath, script_path, ppop_path, life_results_path):
    """
    This function runs LIFEsim with the ExoSim Data as input

    :return:
    """

    # Run LIFEsim with that Data
    command = f"python {script_path} {ppop_path} {life_results_path}"
    env = os.environ.copy()
    env['PYTHONPATH'] = pythonpath
    subprocess.run(command, shell=True, env=env)

    return None


def run_life_single(pythonpath, script_path, dos_pop_path, population_files, results_files):
    """
    Starts an instance of LIFEsim for each population file given. This is needed for the DoS validation tests
    :param population_files:
    :param results_files:
    :return:
    """
    for i in range(len(population_files)):
        pop_file_path = dos_pop_path.joinpath(population_files[i])
        results_file_path = dos_pop_path.joinpath(results_files[i])
        run_life(pythonpath, script_path, pop_file_path, results_file_path)

    return None


def __main__(ppop_path_gd=None, life_results_path_gd=None, modes=None):
    # Paths
    current_dir = Path(__file__).parent.resolve()
    exo_output_path = current_dir.joinpath("Analysis/Output/EXOSIMS/")
    stellar_table_path = current_dir.joinpath("Analysis/Populations/LIFEsim_StellarCat_Table.csv")
    stellar_cat_path = current_dir.joinpath("Analysis/Populations/TargetList_exosims.csv")
    lifesim_path = current_dir.parent.joinpath("LIFEsim-Rick_Branch")

    # Translate Exosim Data to LIFEsim Data and safe to LIFEsim Path
    if modes is None:
        # 'demo1' was the old name of the simulated universe, thus we just fill modes with that to go back to the old
        # behavior before the modes were introduced
        modes = ['demo1']
    processes = []

    for mode in modes:
        print(mode)
        exosim_cat_path = lifesim_path.joinpath("exosim_cat/exosim_univ_" + mode + ".hdf5")
        exo_trans = pop_exo_to_life(exo_output_path, stellar_cat_path, stellar_table_path, mode)
        exo_trans.to_hdf(exosim_cat_path, key='catalog')
        if life_results_path_gd is None:
            life_results_path = current_dir.joinpath(f"Analysis/Output/LIFEsim/{mode}.hdf5")
        else:
            life_results_path = life_results_path_gd
        if ppop_path_gd is None:
            ppop_path = exosim_cat_path
        else:
            ppop_path = ppop_path_gd

        p = mp.Process(target=run_life, args=(str(lifesim_path),
                                              current_dir.joinpath("LIFEsim_ExoSim_Inputs.py"),
                                              ppop_path,
                                              life_results_path))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
