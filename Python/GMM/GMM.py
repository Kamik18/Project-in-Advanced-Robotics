from Python.GMM.gmr.utils import check_random_state
from Python.GMM.gmr import gmm, kmeansplusplus_initialization, covariance_initialization, MVN
from itertools import cycle
from sklearn.mixture import BayesianGaussianMixture
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np


def simulation_data(n_demonstrations, n_steps, sigma=0.25, mu=0.5,
                    start=np.zeros(3), goal=np.array([1., 2., 3.]), random_state=0):
    random_state = np.random.RandomState(random_state)

    X = np.empty((3, n_steps, n_demonstrations))

    # Generate ground-truth for plotting
    ground_truth = np.empty((3, n_steps))
    T = np.linspace(-0, 1, n_steps)
    ground_truth[0] = T
    ground_truth[1] = (T / 20 + 1 / (sigma * np.sqrt(2 * np.pi)) *
                       np.exp(-0.5 * ((T - mu) / sigma) ** 2))
    ground_truth[2] = T

    # Generate trajectories
    for i in range(n_demonstrations):
        noisy_sigma = sigma * random_state.normal(1.0, 0.1)
        noisy_mu = mu * random_state.normal(1.0, 0.1)
        X[0, :, i] = T
        X[1, :, i] = T + (1 / (noisy_sigma * np.sqrt(2 * np.pi)) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2))
        X[2, :, i] = T + (noisy_sigma * np.sqrt(2 * np.pi) *
                          np.exp(-0.5 * ((T - noisy_mu) /
                                         noisy_sigma) ** 2))

    # Spatial alignment
    current_start = ground_truth[:, 0]
    current_goal = ground_truth[:, -1]
    current_amplitude = current_goal - current_start
    amplitude = goal - start
    ground_truth = ((ground_truth.T - current_start) * amplitude /
                    current_amplitude + start).T

    for demo_idx in range(n_demonstrations):
        current_start = X[:, 0, demo_idx]
        current_goal = X[:, -1, demo_idx]
        current_amplitude = current_goal - current_start
        X[:, :, demo_idx] = ((X[:, :, demo_idx].T - current_start) *
                             amplitude / current_amplitude + start).T
    X = X.transpose(2, 1, 0)

    new_X = np.empty((n_demonstrations, n_steps, 6))
    new_X[:, :, 0:3] = X
    new_X[:, :, 3:6] = X
    X = new_X
    return X


def fetch_data_from_records(path: str, skip_size: int = 5) -> np.ndarray:
    """Fetch the demonstration data from the records

    Args:
        path (str): Path to the records. Example:
                  "./Records/Down_A/**/Record_tcp.txt"
        skip_size (int, optional): Skip every n-th data point. Defaults to 5.

    Returns:
        np.ndarray: Data from the records. Shape: (n_demonstrations, n_steps, n_degrees_of_freedom)
    """
    from glob import glob
    import numpy as np

    # Fetch the files
    files: list = glob(pathname=path, recursive=True)
    
    # List of demonstrations
    demonstrations: list = []

    # Iterate the files
    for file in files:
        # Open the file and read the data line by line
        with open(file, "r", encoding="utf-8") as f:
            # Read the data
            data: list = f.readlines()

            # Skip every n-th data point
            data: list = data[::skip_size]
            max_delta: float = 0

            # Iterate the data
            for i in range(len(data)):
                # Split the data and convert to float
                data[i] = data[i].split(",")
                data[i] = [float(x) for x in data[i]]
                if len(data[i]) != 6:
                    print(f"Error: {file}, line: {i}")
                    continue

                '''
                if i > 0:
                    for j in range(3, 6):
                        if abs(data[i][j] - data[i-1][j]) > 5:
                            if (data[i][j] - data[i-1][j]) > np.pi:
                                data[i][j] -= 2*np.pi
                            elif (data[i][j] - data[i-1][j]) < -np.pi:
                                data[i][j] += 2*np.pi

                        delta = abs(data[i][j] - data[i-1][j])
                        if delta > max_delta:
                            max_delta = delta
                '''
            if max_delta > np.pi:
                print(f"Error: {file}, max delta: {max_delta}")
                continue

            # Append the data to a list for all the demonstrations
            demonstrations.append(data)

    # Copy the data from the demonstrations to ndarray
    X: np.ndarray = np.empty((len(demonstrations), len(
        demonstrations[0]), len(demonstrations[0][0])))
    for i in range(len(demonstrations)):
        for j in range(len(demonstrations[i])):
            for k in range(len(demonstrations[i][j])):
                X[i, j, k] = demonstrations[i][j][k]

    # Return the data
    return X


class GMM:
    __data:  np.array
    """Data
    Format: 
        (dimension, steps, demonstrations)
    """
    __gmm_path: np.ndarray
    """GMM path
    """

    def __init__(self, data: np.array, n_components: int = 8) -> None:
        """Initialize the dataset

        Args:
            data (np.array): list of data points. (dimension, steps, demonstrations).
            n_components (int): Number of components in the GMM.
        """
        self.__data = data

        n_demonstrations, n_steps, n_task_dims = self.__data.shape
        X_train = np.empty((n_demonstrations, n_steps, n_task_dims + 1))
        X_train[:, :, 1:] = self.__data
        t = np.linspace(0, 1, n_steps)
        X_train[:, :, 0] = t
        X_train = X_train.reshape(n_demonstrations * n_steps, n_task_dims + 1)

        random_state = check_random_state(0)
        initial_means = kmeansplusplus_initialization(
            X_train, n_components, random_state)
        initial_covs = covariance_initialization(X_train, n_components)
        bgmm = BayesianGaussianMixture(
            n_components=n_components, max_iter=500).fit(X_train)
        self.gmm = gmm(
            n_components=n_components,
            priors=bgmm.weights_,
            means=bgmm.means_,
            covariances=bgmm.covariances_,
            random_state=random_state)

        means_over_time = []
        for step in t:
            conditional_gmm = self.gmm.condition([0], np.array([step]))
            conditional_mvn = conditional_gmm.to_mvn()
            means_over_time.append(conditional_mvn.mean)
        self.__gmm_path: np.ndarray = np.array(means_over_time)

    def plot(self, visualize: str = "path") -> None:
        path_gmm: dict = self.get_path(path="GMM")
        path_mean: dict = self.get_path(path="Mean")

        # Plot the data
        if visualize == "path":
            fig = plt.figure(figsize=(10, 5))
            #fig.suptitle(f"GMM", fontsize=16, ha="center", color="black", fontweight="bold", fontfamily="sans-serif", y=0.9)
            ax = plt.axes(projection="3d")
            
            # Plot the covariance ellipses for the GMM path
            for i in range(len(path_gmm["x"])):
                from scipy import interpolate

                # Define the points
                x_min: float = path_gmm["x"][i] - path_gmm["x_std"][i]
                x_max: float = path_gmm["x"][i] + path_gmm["x_std"][i]
                y_min: float = path_gmm["y"][i] - path_gmm["y_std"][i]
                y_max: float = path_gmm["y"][i] + path_gmm["y_std"][i]

                # Create the pairs
                pair: np.ndarray = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max]
                ])

                # Extract the x and y coordinates
                x = pair[:, 0]
                y = pair[:, 1]

                tck, u = interpolate.splprep([x, y], s=0, per=True)
                basic_form = interpolate.splev(np.linspace(0, 1, 10), tck)
                ax.plot(basic_form[0], basic_form[1],
                        path_gmm["z"][i], color='lime', lw=1, alpha=0.1)
            ax.plot([], [], [], color='lime', label="Tolerance")
                
            # Add the measurements
            for i in range(len(self.__data)):
                set: dict = {
                    "x": self.__data[i, :, 0],
                    "y": self.__data[i, :, 1],
                    "z": self.__data[i, :, 2]
                }
                ax.plot3D(set["x"], set["y"], set["z"],
                          color="black", alpha=0.25)
            ax.plot([], [], [], color="black", label="Measurements")

            # Add the GMM path
            ax.plot3D(path_gmm["x"], path_gmm["y"],
                      path_gmm["z"], color="red", alpha=1.0, label="GMR")

            ax.axis('equal')
            # Get the limits of the axis
            x_lim = (np.floor(ax.get_xlim()[0] * 10) / 10, np.ceil(ax.get_xlim()[1] * 10) / 10)
            y_lim = (np.floor(ax.get_ylim()[0] * 10) / 10, np.ceil(ax.get_ylim()[1] * 10) / 10)
            z_lim = (np.floor(ax.get_zlim()[0] * 10) / 10, np.ceil(ax.get_zlim()[1] * 10) / 10)
            # Set limits
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_zlim(z_lim)
            # Set ticks 
            ax.set_xticks(np.arange( x_lim[0], x_lim[1], 0.1))
            ax.set_yticks(np.arange( y_lim[0], y_lim[1], 0.1))
            ax.set_zticks(np.arange( z_lim[0], z_lim[1], 0.1))
            # Labels
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # Legend - outside the plot
            ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

        elif visualize == "path2d":
            fig = plt.figure(figsize=(10, 5))
            # Add space between the subplots
            fig.subplots_adjust(wspace=0.5, hspace=0.5)
            # Create title
            #fig.suptitle(f"GMM", fontsize=16, ha="center", color="black", fontweight="bold", fontfamily="sans-serif", y=1.0)

            # Create 6 subplots
            axs: list = [plt.subplot(2, 3, i+1) for i in range(6)]
            axs_ylabels: list = ["x [m]", "y [m]",
                                 "z [m]", "x [rad]", "y [rad]", "z [rad]"]

            # Plot the degrees of freedom for each axis over time
            for i in range(len(axs)):
                axs[i].plot(self.__gmm_path[:, i], color="red",
                            alpha=1.0, label="GMR")
                axs[i].plot(self.__data[:, :, i].T, color="black", alpha=0.25)

                axs[i].set_xlabel("Time [ms]")
                axs[i].set_ylabel(axs_ylabels[i])
                #axs[i].legend()
            axs[2].plot([], [], color="black", label="Measurements")

            axs[2].legend(loc='center left', bbox_to_anchor=(1.0, -0.25))

        elif visualize == "covariance":
            # Plot the covariance matrices
            fig = plt.figure(figsize=(10, 5))
            #fig.suptitle(f"Covariances", fontsize=16, ha="center", color="black", fontweight="bold", fontfamily="sans-serif", y=0.8)

            ax = plt.axes()
            ax.set_aspect("equal", "box")
            ax.plot(self.__data[:, :, 0].T, self.__data[:, :, 1].T, color="black", alpha=0.25)
            ax.plot([], [], color="black", label="Measurements")
            
            # Add the GMM path
            ax.plot(path_gmm["x"], path_gmm["y"],
                    color="red", alpha=1.0, label="GMR")

            colors = cycle(["r", "g", "b"])
            for factor in np.linspace(0.5, 4.0, 4):
                new_gmm = gmm(
                    n_components=len(self.gmm.means), priors=self.gmm.priors,
                    means=self.gmm.means[:,
                                         1:], covariances=self.gmm.covariances[:, 1:, 1:],
                    random_state=self.gmm.random_state)
                for mean, (angle, width, height) in new_gmm.to_ellipses(factor):
                    ell = Ellipse(xy=mean, width=width, height=height,
                                  angle=np.degrees(angle))
                    ell.set_alpha(0.15)
                    ell.set_color(next(colors))
                    ax.add_artist(ell)
            ax.set_xlabel("z [m]")
            ax.set_ylabel("x [m]")
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

        elif visualize == "tolerances":
            # Plot the tolerances
            fig = plt.figure(figsize=(10, 5))
            #fig.suptitle(f"Tolerances", fontsize=16, ha="center", color="black", fontweight="bold", fontfamily="sans-serif", y=0.8)

            ax = plt.axes()
            ax.set_aspect("equal", "box")
            ax.plot(self.__data[:, :, 0].T, self.__data[:, :, 1].T, color="black", alpha=0.25)
            ax.plot([], [], color="black", label="Measurements")

            # Add the GMM path
            ax.plot(path_gmm["x"], path_gmm["y"], color="red", alpha=1.0, label="GMR")

            ax.fill_between(
                path_gmm["x"],
                path_gmm["y"] - path_gmm["y_std"]*1.96,
                path_gmm["y"] + path_gmm["y_std"]*1.96,
                color="green", alpha=0.5)
            ax.plot([], [], color="green", label="Tolerance")
            
            ax.set_xlabel("z [m]")
            ax.set_ylabel("x [m]")
            ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

        # Save the plot
        plt.savefig(f"Python/GMM/results/{visualize}.pdf", format="pdf", bbox_inches="tight", transparent=True)

        # Visualize the plots
        plt.show()

    def get_path(self, path: str = "") -> np.ndarray:        
        """Get the found path with the standard deviations matrix for each step

        Args:
            path (str, optional): The path to return. Defaults to "GMM". Options are "GMM" or "Mean".
        Returns:
            np.ndarray: List of the path
        """        
        if path == "GMM":
            path_gmm: dict = {
                "x": self.__gmm_path[:, 0],
                "y": self.__gmm_path[:, 1],
                "z": self.__gmm_path[:, 2],
                "x_std": self.__data[:, :, 0].std(axis=0),
                "y_std": self.__data[:, :, 1].std(axis=0),
                "z_std": self.__data[:, :, 2].std(axis=0)
            }
            return path_gmm
        elif path == "Mean":
            path_mean: dict = {
                "x": self.__data[:, :, 0].mean(axis=0),
                "y": self.__data[:, :, 1].mean(axis=0),
                "z": self.__data[:, :, 2].mean(axis=0),
                "x_std": self.__data[:, :, 0].std(axis=0),
                "y_std": self.__data[:, :, 1].std(axis=0),
                "z_std": self.__data[:, :, 2].std(axis=0)
            }
            return path_mean
        else:
            # Create a list of standard deviations for each channel
            covariance: list = []
            for i in range(self.__data.shape[2]):
                covariance.append(self.__data[:, :, i].std(axis=0))
            
            return (self.__gmm_path, np.array(covariance).T)

if __name__ == "__main__":
    #data:np.ndarray = simulation_data(n_demonstrations=20, n_steps=150)
    data: np.ndarray = fetch_data_from_records(path="Records/Up_A/**/Record_tcp.txt", skip_size=10)  # From up to down
    data: np.ndarray = fetch_data_from_records(path="Records/Down_A/**/Record_tcp.txt", skip_size=10)
    #data: np.ndarray = fetch_data_from_records(path="./Records/Up_B/**/Record_tcp.txt", skip_size=10)
    #data: np.ndarray = fetch_data_from_records(path="./Records/Down_B/**/Record_tcp.txt", skip_size=10)
    print(data.shape)

    GMM_translation: GMM = GMM(data=data, n_components=8)
    #GMM_translation.plot("path")
    #GMM_translation.plot("path2d")
    #GMM_translation.plot("covariance")
    #GMM_translation.plot("tolerances")
    path_translation: np.ndarray = GMM_translation.get_path()

    # Plot 2D path
    data: np.ndarray = fetch_data_from_records(path="Records/Up_A/**/Record_tcp.txt", skip_size=10)

    # Itterate the 4 dimension of the data
    for demo in range(data.shape[0]):
        for step in range(data.shape[1]):
                if data[demo, step, 3] < 0:
                    data[demo, step, 3] += np.pi * 2

    GMM_translation: GMM = GMM(data=data, n_components=8)
    GMM_translation.plot("path2d")

    # Plot covariance and tolerances
    temp_data = data[:, :, 0] # Store the x axis
    data[:, :, 0] = data[:, :, 2] # Swap x and z axis 
    data[:, :, 2] = temp_data    

    GMM_translation: GMM = GMM(data=data, n_components=8)
    GMM_translation.plot("covariance")
    GMM_translation.plot("tolerances")

    # Plot 3D path 
    data: np.ndarray = fetch_data_from_records(path="Records/Down_A/**/Record_tcp.txt", skip_size=10)
    GMM_translation: GMM = GMM(data=data, n_components=8)
    GMM_translation.plot("path")

    