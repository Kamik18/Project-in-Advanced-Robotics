from gmr.utils import check_random_state
from gmr import gmm, kmeansplusplus_initialization, covariance_initialization
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

    def __init__(self, data: np.array, n_components:int = 8) -> None:
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
        path_gmm: dict = {
            "x": self.__gmm_path[:, 0],
            "y": self.__gmm_path[:, 1],
            "z": self.__gmm_path[:, 2],
            "x_std": data[:, :, 0].std(axis=0),
            "y_std": data[:, :, 1].std(axis=0),
            "z_std": data[:, :, 2].std(axis=0)
        }

        path_mean: dict = {
            "x": data[:, :, 0].mean(axis=0),
            "y": data[:, :, 1].mean(axis=0),
            "z": data[:, :, 2].mean(axis=0),
            "x_std": data[:, :, 0].std(axis=0),
            "y_std": data[:, :, 1].std(axis=0),
            "z_std": data[:, :, 2].std(axis=0)
        }

        # Plot the data
        if visualize == "path":
            fig_paths = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection="3d")
            
            # Add the measurements
            for i in range(len(self.__data)):
                set: dict = {
                    "x": self.__data[i, :, 0],
                    "y": self.__data[i, :, 1],
                    "z": self.__data[i, :, 2]
                }
                ax.plot3D(set["x"], set["y"], set["z"],
                          color="black", alpha=0.25)

            # Add the GMM path
            ax.plot3D(path_gmm["x"], path_gmm["y"],
                      path_gmm["z"], color="red", alpha=1.0, label="GMM")
            # Mean path
            ax.plot3D(path_mean["x"], path_mean["y"],
                      path_mean["z"], color="green", alpha=1.0, label="Mean")
            ax.set_title(f"GMM with {len(self.__data)} demonstrations")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend()

        elif visualize == "path2d":
            fig = plt.figure(figsize=(10, 5))
            # Add space between the subplots
            fig.subplots_adjust(wspace=0.5, hspace=0.5)
            # Create title
            fig.suptitle(f"GMM {len(self.__data)} demonstrations", fontsize=16, ha="center",
                         va="top", color="black", fontweight="bold", fontfamily="sans-serif")

            # Create 6 subplots
            axs: list = [plt.subplot(2, 3, i+1) for i in range(6)]
            axs_ylabels: list = ["x [m]", "y [m]",
                                 "z [m]", "x [rad]", "y [rad]", "z [rad]"]

            # Plot the degrees of freedom for each axis over time
            for i in range(len(axs)):
                axs[i].plot(self.__gmm_path[:, i], color="red",
                            alpha=1.0, label="GMM")
                axs[i].plot(self.__data[:, :, i].mean(axis=0),
                            color="green", alpha=1.0, label="Mean")
                axs[i].plot(self.__data[:, :, i].T, color="black", alpha=0.25)

                axs[i].set_xlabel("Time [ms]")
                axs[i].set_ylabel(axs_ylabels[i])
                axs[i].legend()

        elif visualize == "covariance":
            # Plot the covariance matrices
            fig_covariances = plt.figure(figsize=(10, 5))
            ax = plt.axes()
            ax.set_aspect("equal", "box")
            ax.plot(self.__data[:, :, 0].T, self.__data[:,
                    :, 1].T, color="black", alpha=0.25)
            # Add the GMM path
            ax.plot(path_gmm["x"], path_gmm["y"],
                    color="red", alpha=1.0, label="GMM")
            # Mean path
            ax.plot(path_mean["x"], path_mean["y"],
                    color="green", alpha=1.0, label="Mean")
            colors = cycle(["r", "g", "b"])
            for factor in np.linspace(0.5, 4.0, 4):
                new_gmm = gmm(
                    n_components=len(self.gmm.means), priors=self.gmm.priors,
                    means=self.gmm.means[:,1:], covariances=self.gmm.covariances[:, 1:, 1:],
                    random_state=self.gmm.random_state)
                for mean, (angle, width, height) in new_gmm.to_ellipses(factor):
                    ell = Ellipse(xy=mean, width=width, height=height,
                                  angle=np.degrees(angle))
                    ell.set_alpha(0.15)
                    ell.set_color(next(colors))
                    ax.add_artist(ell)
            ax.set_title(f"Covariance matrices")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.legend(loc="upper right")
        elif visualize == "tolerances":
            # Plot the tolerances
            fig_tolerances = plt.figure(figsize=(10, 5))
            ax = plt.axes()
            ax.set_aspect("equal", "box")
            ax.plot(self.__data[:, :, 0].T, self.__data[:,
                    :, 1].T, color="black", alpha=0.25)
            # Add the mean path
            ax.plot(path_gmm["x"], path_gmm["y"],
                    color="red", alpha=1.0, label="GMM")
            # Mean path
            ax.plot(path_mean["x"], path_mean["y"],
                    color="green", alpha=1.0, label="Mean")

            ax.set_title(f"GMM with {len(self.__data)} demonstrations")
            ax.fill_between(
                path_mean["x"],
                path_mean["y"] - 1.96 * path_mean["y_std"],
                path_mean["y"] + 1.96 * path_mean["y_std"],
                color="blue", alpha=0.5)
            ax.fill_between(
                path_gmm["x"],
                path_gmm["y"] - path_gmm["y_std"],
                path_gmm["y"] + path_gmm["y_std"],
                color="green", alpha=0.5)
            ax.set_title(f"Tolerances")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.legend()

        # Visualize the plots
        plt.show()

    def get_path(self, path: str = "GMM") -> np.ndarray:
        """Get the found path with the standard deviations matrix for each step

        Returns:
            np.ndarray: List of the path
        """
        if path == "GMM":
            return self.__gmm_path
        elif path == "Mean":
            # Create the mean path
            mean: np.ndarray = np.zeros(self.__data.shape)

            # Mean each degree of freedom
            for axis in range(self.__data.shape[2]):
                mean[:, :, axis] = self.__data[:, :, axis].mean(axis=0)

            return mean


if __name__ == "__main__":
    #data:np.ndarray = simulation_data(n_demonstrations=20, n_steps=150)

    data: np.ndarray = fetch_data_from_records(path="./Records/Up_A/**/Record_tcp.txt", skip_size=10) # From up to down
    #data: np.ndarray = fetch_data_from_records(path="./Records/Down_A/**/Record_tcp.txt", skip_size=10)
    #data: np.ndarray = fetch_data_from_records(path="./Records/Up_B/**/Record_tcp.txt", skip_size=10)
    #data: np.ndarray = fetch_data_from_records(path="./Records/Down_B/**/Record_tcp.txt", skip_size=10)
    print(data.shape)

    """ Plot the covariance and the tolerances
    temp_data = data[:, :, 1] # Store the x axis
    data[:, :, 1] = data[:, :, 2] # Swap x and z axis 
    data[:, :, 2] = temp_data    
    """

    GMM_translation: GMM = GMM(data=data, n_components= 4)
    GMM_translation.plot("path")
    #GMM_translation.plot("path2d")
    #GMM_translation.plot("covariance")
    #GMM_translation.plot("tolerances")
    path_translation: np.ndarray = GMM_translation.get_path()

