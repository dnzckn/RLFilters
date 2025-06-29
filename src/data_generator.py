
import os
import csv
import numpy as np

def generate_synthetic_data(output_dir, num_samples, grid_size, num_freq_points):
    """
    Generates a synthetic dataset of random filter layouts and dummy S-parameters.

    Args:
        output_dir (str): The directory to save the dataset.
        num_samples (int): The number of samples to generate.
        grid_size (int): The size of the pixelated grid (grid_size x grid_size).
        num_freq_points (int): The number of frequency points for the S-parameters.
    """
    data_file = os.path.join(output_dir, "synthetic_data.csv")
    os.makedirs(output_dir, exist_ok=True)

    with open(data_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = [f"pixel_{i}" for i in range(grid_size * grid_size)]
        header += [f"s_param_{i}" for i in range(num_freq_points)]
        writer.writerow(header)

        for _ in range(num_samples):
            # Generate a random filter layout
            layout = np.random.randint(0, 2, size=(grid_size, grid_size))
            
            # Dummy S-parameter generation: a simple function of the number of "on" pixels
            num_on_pixels = np.sum(layout)
            # Create a simple resonant-like curve. The "resonance" frequency shifts with the number of pixels.
            center_freq = (num_on_pixels / (grid_size * grid_size)) * num_freq_points
            freq_axis = np.arange(num_freq_points)
            s_params = 1 / (1 + ((freq_axis - center_freq) / 5)**2) # Lorentzian-like peak
            s_params += np.random.normal(0, 0.05, num_freq_points) # Add some noise


            row = list(layout.flatten()) + list(s_params)
            writer.writerow(row)

    print(f"Generated {num_samples} samples in {data_file}")

if __name__ == "__main__":
    # Configuration
    NUM_SAMPLES = 1000
    GRID_SIZE = 16
    NUM_FREQ_POINTS = 100
    OUTPUT_DIR = "/Users/deniz/Documents/GitHub/RLFilters/data"
    
    generate_synthetic_data(OUTPUT_DIR, NUM_SAMPLES, GRID_SIZE, NUM_FREQ_POINTS)
