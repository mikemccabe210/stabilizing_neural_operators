"""
Dedalus script simulating the viscous shallow water equations on a sphere. This
script demonstrates solving an initial value problem on the sphere. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_sphere.py` script can be used to produce
plots from the saved data. The simulation should about 5 cpu-minutes to run.

The script implements the test case of a barotropically unstable mid-latitude
jet from Galewsky et al. 2004 (https://doi.org/10.3402/tellusa.v56i5.14436).
The initial height field balanced the imposed jet is solved with an LBVP.
A perturbation is then added and the solution is evolved as an IVP.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shallow_water.py
    $ mpiexec -n 4 python3 plot_sphere.py snapshots/*.h5
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'

import numpy as np
import dedalus.public as d3
import logging
import argparse

logger = logging.getLogger(__name__)


# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600
day = hour * 24
year = hour * 1008 # 42 day years - Chosen based on the fact that the sim gets boring ~ 4000 hours

if __name__ == '__main__':
    # Parse IC file for PM xfer
    parser = argparse.ArgumentParser()
    parser.add_argument('--ic_file', default='/home/mike/Documents/Projects/shallow_water_gen/SWE_init.npy')
    parser.add_argument('--output_dir', default='.')
    args = parser.parse_args()
    ic_file = args.ic_file
    output_dir = args.output_dir

    # Parameters
    Nphi = 256
    Ntheta = 128
    dealias = 3/2
    R = 6.37122e6 * meter
    Omega = 7.292e-5 / second
    nu = 1e5 * meter**2 / second / 96**2 # Hyperdiffusion matched at ell=32
    g = 9.80616 * meter / second**2
    timestep = 60 * second
    burn_in = .5*year
    stop_sim_time = 3.5*year
    dtype = np.float64

    # Bases
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)
    basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

    # Fields
    u = dist.VectorField(coords, name='u', bases=basis)
    h = dist.Field(name='h', bases=basis)

    # Substitutions
    zcross = lambda A: d3.MulCosine(d3.skew(A))



    # Copy ICs from hpa 500 fields
    ICs = np.load(ic_file)
    ICs = np.swapaxes(ICs, 1, 2)
    ICs = np.flip(ICs, 2)
    u0 = ICs[:2] * meter / second #* .3
    h0 = ICs[2] * meter
    hs0 = ICs[3] * meter

    ### From experiments on removing fast initial gravity waves by repeated init/rebalancing
    # filename = sorted(['snapshots/' + p for p in os.listdir('snapshots/')], key=lambda x: os.path.getmtime(x))[-1]
    # with h5py.File('swe_92_1720.h5', mode='r') as file:
    #     u0 = file['tasks']['u'][:][-1]
    #     h0 = file['tasks']['h'][:][-1] #/ 50

    # Center h0 and redefine height so mountains are more impactful
    hs = dist.Field(name='hs', bases=basis)
    hs['g'] = hs0
    H = (h0).mean()
    h0 = h0 - H
    u['g'] = u0
    h['g'] = h0
    H = 5960 * meter # Just keeping height from Williamson 5

    # # Initial conditions: balanced height
    c = dist.Field(name='c')
    problem = d3.LBVP([h, c], namespace=locals())
    problem.add_equation("g*lap(h) + c = - div(u@grad(u) + 2*Omega*zcross(u))")
    problem.add_equation("ave(h) = 0")

    solver_init = problem.build_solver()
    solver_init.solve()

    # Momentum forcing - seasonal
    def find_center(t):
        time_of_day = t / day
        time_of_year = t / year
        max_declination = .4 # Truncated from estimate of earth's solar decline

        lon_center = time_of_day*2*np.pi # Rescale sin to 0-1 then scale to np.pi
        lat_center = np.sin(time_of_year*2*np.pi)*max_declination

        lon_anti = np.pi + lon_center  #2*np.((np.sin(-time_of_day*2*np.pi)+1) / 2)*pi  #(lon_center + np.pi) // (2*np.pi)
        # lat_anti = np.sin(np.pi + time_of_year*2*np.pi)*max_declination    # (lat_center + np.pi/2) // (np.pi)
        return lon_center, lat_center, lon_anti, lat_center

    def season_day_forcing(phi, theta, t, h_f0):
        phi_c, theta_c, phi_a, theta_a = find_center(t)
        sigma = np.pi/2
        # Coefficients aren't super well-designed - idea is one side of the planet increases
        # the other side decreases and the effect is centered around a seasonally-shifting Gaussian.
        # The original thought was to have this act on momentum, but this was harder to implement in a stable way
        # since increasing/decreasing by same factor is net energy loss.
        coefficients = np.cos(phi - phi_c) * np.exp(-(theta-theta_c)**2 / sigma**2)
        forcing = h_f0 * coefficients
        return forcing

    phi, theta = dist.local_grids(basis)
    t = dist.Field(name='t')
    lat = np.pi / 2 - theta + 0*phi
    phi_var = dist.Field(name='phi_var', bases=basis)
    phi_var['g'] += phi
    theta_var = dist.Field(name='theta_var', bases=basis)
    theta_var['g'] += lat
    h_f0 = 10 * meter  # Increasing this starts leading to fast waves (or maybe it just looks that way at 60 FPS/ 2.x day per sec)
    h_f = season_day_forcing(phi_var, theta_var, t, h_f0)

    # Problem
    problem = d3.IVP([u, h], namespace=locals(), time=t)
    problem.add_equation("dt(u) + nu*lap(lap(u)) + g*grad(h)  + 2*Omega*zcross(u) = - u@grad(u) + 1e-5*u") # 1e-5 is to offset energy loss
    problem.add_equation("dt(h) + nu*lap(lap(h)) + (H)*div(u) = - div(u*(h-hs)) + h_f")  # h is perturbation of height

    # Init to remove fast waves in sim - should probably just filter in time here, but this works.
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = burn_in
    CFL = d3.CFL(solver, initial_dt=60*second, cadence=1, safety=1, threshold=0.05,
                 max_dt=1*hour)
    CFL.add_velocity(u)
    logger.info('Trying init loop to get rid of fast waves')
    for i in range(10):
        logger.info('Starting init cycle %s' % i)
        solver_init.solve()
        for j in range(50):
            timestep = 60*second
            solver.step(timestep)
    solver_init.solve()
    # Now do burn-in

    try:
        logger.info('Starting burn-in loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            # print(uf.evaluate()['g'])
            if (solver.iteration-1) % 10 == 0:
                logger.info('Burn-in Iteration=%i, Time=%e, dt=%e' % (solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of burn loop.')
        raise

    # Now define real problem
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = stop_sim_time

    # Analysis
    snapshots = solver.evaluator.add_file_handler(output_dir,
                                                  sim_dt=1*hour, max_writes=1*year)
    snapshots.add_tasks(solver.state, layout='g')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

    ### Extras for debugging
    # snapshots.add_task(uf, name='forcing')
    # snapshots.add_task(hs, name='topography')
    #
    # snapshots.add_task(2*Omega*zcross(u), name='coriolis')
    # snapshots.add_task(u@d3.Gradient(u), name='transport')
    # snapshots.add_task(g*d3.Gradient(h), name='pressure')

    # CFL
    CFL = d3.CFL(solver, initial_dt=60*second, cadence=1, safety=1, threshold=0.05,
                 max_dt=1*hour)
    CFL.add_velocity(u)
    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            # print(uf.evaluate()['g'])
            if (solver.iteration-1) % 10 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' % (solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
