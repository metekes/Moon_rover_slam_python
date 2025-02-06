import numpy as np
import matplotlib.pyplot as plt
import argparse

# Function to read the file and plot the data
def read_and_plot(file_path):
    # Read the data from the file
    gt_positions = []
    estimated_positions = []
    
    with open(file_path, "r") as file:
        # Skip the header line
        next(file)
        
        for line in file:
            # Parse each line
            values = list(map(float, line.strip().split(',')))
            gt_positions.append(values[:3])  # Ground truth (GT_x, GT_y, GT_z)
            estimated_positions.append(values[3:])  # Estimated positions (Est_x, Est_y, Est_Z)
    
    # Convert to numpy arrays
    gt_positions = np.array(gt_positions)
    estimated_positions = np.array(estimated_positions)
    
    # Calculate global axis limits
    all_positions = np.vstack((gt_positions, estimated_positions))
    global_min = np.min(all_positions)
    global_max = np.max(all_positions)
    
    # Plot in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    # Plot ground truth and estimated positions
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label="Ground Truth", color="blue", marker="o")
    ax.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label="Estimated", color="red", marker="x")
    
    # Set uniform axis limits
    ax.set_xlim([global_min, global_max])
    ax.set_ylim([global_min, global_max])
    ax.set_zlim([global_min, global_max])
    
    # Label axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Positions: Ground Truth vs Estimated")
    ax.legend()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3D trajectories from a file.")
    parser.add_argument("file_path", type=str, help="Path to the input file containing the trajectories.")
    args = parser.parse_args()
    
    read_and_plot(args.file_path)