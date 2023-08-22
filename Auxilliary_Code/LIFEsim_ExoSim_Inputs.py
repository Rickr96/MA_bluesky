import sys
import os
from pathlib import Path

parent_dir = Path(__file__).parents[1]
life_dir = parent_dir.joinpath("LIFEsim-Rick_Branch")
os.chdir(life_dir)
sys.path.append(os.getcwd())
import lifesim


if __name__ == '__main__':

    # ---------- Set-Up ----------
    ppop_path = sys.argv[1]
    print(sys.argv[0])
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    # create bus
    bus = lifesim.Bus()

    # setting the options
    bus.data.options.set_scenario('baseline')

    # ---------- Loading the Catalog ----------
    bus.data.catalog_from_ppop(input_path=str(ppop_path))

    # bus.data.catalog_remove_distance(stype=0, mode='larger', dist=0.)  # remove all A stars
    # bus.data.catalog_remove_distance(stype=4, mode='larger', dist=10.)  # remove M stars > 10pc to
    # speed up calculation

    # ---------- Creating the Instrument ----------

    # create modules and add to bus
    instrument = lifesim.Instrument(name='inst')
    bus.add_module(instrument)

    # TransmissionMap is the "normal" transmission mode without planet movement.
    # It is mutually exclusive with OrbitalTransmissionMap

    transm = lifesim.TransmissionMap(name='transm')
    bus.add_module(transm)
    exo = lifesim.PhotonNoiseExozodi(name='exo')
    bus.add_module(exo)
    local = lifesim.PhotonNoiseLocalzodi(name='local')
    bus.add_module(local)
    star = lifesim.PhotonNoiseStar(name='star')
    bus.add_module(star)

    # connect all modules
    bus.connect(('inst', 'transm'))
    bus.connect(('inst', 'exo'))
    bus.connect(('inst', 'local'))
    bus.connect(('inst', 'star'))

    bus.connect(('star', 'transm'))

    # ---------- Creating the Optimizer ----------
    # After every planet is given an SNR, we want to distribute the time available in the search phase
    # such that we maximize the number of detections.

    # optimizing the result
    opt = lifesim.Optimizer(name='opt')
    bus.add_module(opt)
    ahgs = lifesim.AhgsModule(name='ahgs')
    bus.add_module(ahgs)

    bus.connect(('transm', 'opt'))
    bus.connect(('inst', 'opt'))
    bus.connect(('opt', 'ahgs'))

    # ---------- Running the Simulation ----------

    # run simulation. This function assigns every planet an SNR for 1 hour of integration time. Since
    # we are currently only simulating photon noise, the SNR will scale with the integration time as
    # sqrt(t)
    instrument.get_snr()

    opt.ahgs()
    output_dir = sys.argv[2]
    # output_dir = parent_dir.joinpath('Auxilliary_Code/Analysis/Output/LIFEsim/demo1.hdf5')
    # ---------- Saving the Results ----------
    bus.data.export_catalog(output_path=str(output_dir))
