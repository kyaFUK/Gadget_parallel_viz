import os
from mpi4py import MPI
import h5py
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import matplotlib as mpl
import subprocess

#from yt.units import mp 

#MPI.Init_thread(MPI.THREAD_MULTIPLE)

# . /vol0004/apps/oss/spack/share/spack/setup-env.sh
## spack load /qbmpmn2 # python 3.11
# spack load /tq5kd2o # mpi4py
# source load .venv/bin/activate
# export LD_PRELOAD=/usr/lib/FJSVtcs/ple/lib64/libpmix.so # mpi4py実行時に必要
# mpirun --bind-to core --map-by core -np 16 python yt_mpi_IO.py

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#target="density"
target="temperature"

# Ensure MPI is initialized for thread safety

#provided = MPI.Query_thread()
#if provided < MPI.THREAD_MULTIPLE:
#    print(f"MPI does not support MPI_THREAD_MULTIPLE. Provided level: {provided}")
#else:
#    print(provided, MPI.THREAD_MULTIPLE)
#    print("MPI supports MPI_THREAD_MULTIPLE.")

if rank==0:
    print("Computing starts.")

# Start time measurement
start_time = MPI.Wtime()  # Time measurement starts here

# Define the cubic spline SPH kernel
def cubic_spline_kernel(r, h):
    q = r / h
    sigma = 8 / (np.pi * h**3)  # Normalization factor for 3D
    if q < 0.5:
        return sigma * (1 - 6 * q**2 + 6 * q**3)
    elif q < 1.0:
        return sigma * 2 * (1 - q)**3
    else:
        return 0.0

# Vectorize the kernel function to apply over arrays
vectorized_kernel = np.vectorize(cubic_spline_kernel)

# Define the read function for HDF5 files
def read_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        # Check if PartType0/Density exists
        if 'PartType0' not in f or 'Density' not in f['PartType0']:
            print(f"Warning: 'PartType0/Density' not found in {file_path}")
            return None, None, None, None  # Return None if the dataset is missing
        
        density = np.array(f['PartType0/Density'])  # Already normalized, no unit conversion needed
        internal_energy = np.array(f['PartType0/InternalEnergy'])  # Already normalized
        positions = np.array(f['PartType0/Coordinates'])  # Already in kpc
        smoothing_length = np.array(f['PartType0/SmoothingLength'])  # SPH kernel radii
    
    return density, internal_energy, positions, smoothing_length

# Function to run 'ls -lhS' to get files sorted by size
def get_file_sizes_sorted(src):
    try:
        #print(src + "/asura_fdps_00001.*")
        #result = subprocess.run(f"ls -lhS {src}asura_fdps_00001.*", capture_output=True, text=True, shell=True)
        result = subprocess.run(f"ls -lhS {src}", capture_output=True, text=True, shell=True)

        
        # Parse the output
        lines = result.stdout.splitlines()
        files_with_size = []
        
        for line in lines[1:]:  # Skip the first line (total size information)
            parts = line.split()
            size = parts[4]  # Size is usually the 5th column in 'ls -lhS' output
            filename = parts[-1]  # Filename is the last column
            files_with_size.append((filename, size))
        
        return files_with_size
    except subprocess.CalledProcessError as e:
        print(f"Error running ls command: {e}")
        return []

# Function to convert human-readable size (like '1K', '100M') to bytes
def size_in_bytes(size):
    units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    if size[-1] in units:
        return float(size[:-1]) * units[size[-1]]
    else:
        return float(size)

# Function to load data in parallel within each MPI rank
# Function to load files in parallel and filter out non-HDF5 files
def load_files_parallel(file_subset, src):
    # Filter out non-HDF5 files
    valid_files = [src + file for file in file_subset if file.endswith('.hdf5')]

    # Use ThreadPoolExecutor to parallelize file reading
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(read_hdf5_file, valid_files))

    # Filter out None results (if a file was skipped or had an error)
    results = [res for res in results if res[0] is not None]
    
    return results

# Constants for temperature calculation in Kelvin
Mproton = 1.67262192369e-24  # Proton mass in grams
Msolar = 1.989e33
kpc = 1e3*3.085678e18
Ggrav=6.67430e-8
unit_mass = 1.e10*Msolar
unit_leng = kpc

unit_time_G_1 = np.sqrt((unit_leng)**3/(Ggrav*unit_mass))
unit_time = unit_time_G_1
unit_velc = unit_leng / unit_time
unit_eng  = unit_mass * (unit_velc)**2
unit_spen = unit_eng  / unit_mass
unit_dens = 404.461303941453 #cm^-3

normalization_constant = unit_spen  # 1e-10 factor can be applied if needed
gamma = 5.0 / 3.0  # Adiabatic index, change if needed
mu = 0.6  # Mean molecular weight, change according to your simulation

# SPH interpolation function to interpolate particle data onto a 2D grid
def sph_interpolate_to_grid(positions, values, smoothing_lengths, global_grid_size, global_grid_limits):
    """
    Interpolates particle data to a 2D grid using SPH kernel.

    Parameters:
    - positions: Nx2 array of particle positions.
    - values: N array of particle values to interpolate (e.g., temperature).
    - smoothing_lengths: N array of smoothing lengths for each particle.
    - global_grid_size: Tuple (nx, ny) specifying the number of grid points in each dimension.
    - global_grid_limits: Limits of the grid in the form [[xmin, xmax], [ymin, ymax]].

    Returns:
    - grid: Interpolated 2D grid of values.
    - grid_weights: 2D grid of accumulated kernel weights.
    """
    # Define the grid
    nx, ny = global_grid_size
    xmin, xmax = global_grid_limits[0]
    ymin, ymax = global_grid_limits[1]

    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymin, ymax, ny)
    grid = np.zeros((nx, ny))
    grid_weights = np.zeros((nx, ny))

    # Loop over all particles
    for pos, val, h in zip(positions, values, smoothing_lengths):
        x, y = pos
        h_inv = 1.0 / h

        # Find the grid indices that fall within the smoothing length
        ix_min = max(int((x - h - xmin) / (xmax - xmin) * nx), 0)
        ix_max = min(int((x + h - xmin) / (xmax - xmin) * nx), nx - 1)
        iy_min = max(int((y - h - ymin) / (ymax - ymin) * ny), 0)
        iy_max = min(int((y + h - ymin) / (ymax - ymin) * ny), ny - 1)

        # Loop over the grid points within the smoothing length
        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                xg, yg = x_grid[ix], y_grid[iy]
                r = np.sqrt((xg - x)**2 + (yg - y)**2)

                # Apply the SPH kernel
                w = cubic_spline_kernel(r, h)

                # Accumulate the value and weight into the grid
                grid[ix, iy] += val * w
                grid_weights[ix, iy] += w

    return grid, grid_weights

# Function to normalize the grid based on the weights
def normalize_grid(grid, grid_weights):
    nonzero_mask = grid_weights > 0
    grid[nonzero_mask] /= grid_weights[nonzero_mask]
    return grid

# Function to create a 2D slice plot of the data
def slice_plot_2d(grid, plot_limits, title, save_dir):

    if target=="density":
        label=r"Density $[\mathrm{cm^{-3}}]$"
        vmin=1e-3
        vmax=1e1
        color='viridis'

    if target=="temperature":
        label="Temperature [K]"
        vmin=1e1
        vmax=1e7
        color='inferno'
        
    # Plot the 2D temperature grid (xy-plane)
    plt.figure(figsize=(15, 15))
    #print("vmin: {}, vmax: {}".format(vmin,vmax))
    #print(grid.max(), grid.min())
    #print(grid)
    plt.imshow(grid.T, extent=[plot_limits[0][0], plot_limits[0][1], 
                               plot_limits[1][0], plot_limits[1][1]],
               norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
               cmap=color,
               origin='lower', interpolation='bilinear')  # Bilinear interpolation for the plot

        
    plt.colorbar(label=label)  # Now the temperature is in Kelvin
    plt.title(title)
    plt.xlabel("x [kpc]")
    plt.ylabel("y [kpc]")
    plt.savefig(f"{save_dir}slice_plot_{target}_{title}.png")
    plt.close()

# Main processing block
#src = "/home/hirashima/NAS4Fujii/WLM/ML/noFUV/snapshots/"
fn=2
#src='/data/hp120286/u10158/LMCGenIC1024_InitTurb_1e-1/results/snapshots{:05d}/'.format(fn)
src='/data/hp120286/u10158/LMCFollow_4/results/snapshots{:05d}/'.format(fn)

print(src)
if rank == 0:
    # Get file sizes and sort them by size using 'ls -lhS'
    files_with_size_human = get_file_sizes_sorted(src)
    
    # Convert sizes to bytes for easier calculations and sorting
    files_with_size = [(file, size_in_bytes(size)) for file, size in files_with_size_human]

    # Sort files by size (largest first)
    sorted_files_with_size = sorted(files_with_size, key=lambda x: x[1], reverse=True)
    
    # Extract only filenames (sorted by size)
    sorted_file_list = [file for file, size in sorted_files_with_size]
else:
    sorted_file_list = None
    
# Broadcast the sorted file list to all ranks
sorted_file_list = comm.bcast(sorted_file_list, root=0)


# Distribute files by size using modulo to balance workload
file_subset = [sorted_file_list[i] for i in range(rank, len(sorted_file_list), size)]
#print(file_subset)
# Each rank loads its own subset of data
results = load_files_parallel(file_subset, src)

# Extract data from the results
density = np.concatenate([res[0] for res in results])
internal_energy = np.concatenate([res[1] for res in results])
positions = np.concatenate([res[2] for res in results])
smoothing_lengths = np.concatenate([res[3] for res in results])

if rank==0:
    print("Read Done.")
    
# Filter the particles by |z| < 1 kpc
z_limit = 0.5  # 1 kpc for z-axis filtering
mask = np.abs(positions[:, 2]) < z_limit

filtered_positions = positions[mask]
filtered_density = density[mask]
filtered_internal_energy = internal_energy[mask]
filtered_smoothing_lengths = smoothing_lengths[mask]

################
# Constants for number density calculation
mu = 0.6 #1.22  # Mean molecular weight for neutral gas, adjust as needed
m_H = 1.67e-24  # Hydrogen mass in grams

# Calculate number density: n = density / (mu * m_H)
number_density = filtered_density * unit_dens

################


# Compute temperature in Kelvin using the internal energy and the provided formula
kb=1.380649e-16
temperature = (gamma - 1.0) * filtered_internal_energy * Mproton / (kb / mu) * normalization_constant

if target=="density":
    target_physics = number_density

if target=="temperature":
    target_physics = temperature

# Define the grid and interpolation parameters
global_grid_size = (800, 800)  # Global grid size (nx, ny)
#300 Total time taken: 214.17 seconds
global_grid_limits = [[-5, 5], [-5, 5]]  # Plot region limits (in kpc)

# Interpolate particle data onto the global grid using SPH kernel
local_physics_grid, local_weights_grid = sph_interpolate_to_grid(
    filtered_positions[:, :2],  # Only x, y coordinates for the 2D interpolation
    target_physics,
    filtered_smoothing_lengths,
    global_grid_size,
    global_grid_limits
)

print("Done ", rank)

# Initialize the global grids (only root process will collect the final result)
global_physics_grid = np.zeros_like(local_physics_grid)
global_weights_grid = np.zeros_like(local_weights_grid)

# Reduce local grids to form the global grids
comm.Reduce(local_physics_grid, global_physics_grid, op=MPI.SUM, root=0)
comm.Reduce(local_weights_grid, global_weights_grid, op=MPI.SUM, root=0)

# Rank 0 will normalize the temperature grid and create the final plot
if rank == 0:
    normalized_physics_grid = normalize_grid(global_physics_grid, global_weights_grid)

    
    if target=="density":
        label="Density"
        target_physics = number_density

    if target=="temperature":
        label="Temperature"
        target_physics = temperature
        
    # Save the final plot
    DSTDIR = "./output_plots/4node_{:03d}_".format(fn)
    slice_plot_2d(normalized_physics_grid, global_grid_limits, label + " Slice in xy-plane (|z| < 1 kpc)_4", DSTDIR)

# End time measurement
end_time = MPI.Wtime()

# Output time taken
if rank == 0:
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

# Finalize MPI
comm.Barrier()
MPI.Finalize()
