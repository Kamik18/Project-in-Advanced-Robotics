
class GMM:
    __path: dict = {
        "path": list(),
        "covariances": list()
    }
    """Path

    Args:
        path: List of points or quaternions representing the path.
        Covariances: List of covariances matrices for each point on the path.
    """

    __data_points: list = []

    def __init__(self, data: list = None) -> None:
        """Initialize the dataset

        Args:
            data (list, optional): list of data points, translation: (index, x, y, z) or quaternion (index, a, i, j, k). Defaults to None.
        """
        # Convert the received data from string to floats
        self.__data_points = list(range(len(data)))
        for row in range(len(self.__data_points)):
            self.__data_points[row] = [float(i) for i in data[row]]
    
    def add_data_points(self, data: list) -> None:
        """Add data points to the list

        Args:
            data (list): list of data points, translation: (index, x, y, z) or quaternion (index, a, i, j, k).
        """
        assert (len(self.__data_points) == len(
            data)),            f"The new data dimension, {len(data)}, is not the same dimensions as the existing data, {len(self.__data_points)}"

        # Convert the received data from string to floats
        for row in range(len(self.__data_points)):
            lst:list = [float(i) for i in data[row]]
            for instance in lst:
                self.__data_points[row].append(instance)

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        # Data for a three-dimensional line
        index = self.__data_points[0]

        prev: int = 0
        peaks: list = []
        for i in range(len(index)):
            if index[i] < prev:
                peaks.append(i)
            prev = index[i]
        peaks.append(len(index))

        x = self.__data_points[1]
        y = self.__data_points[2]
        z = self.__data_points[3]
        data_sets: list = []

        start: int = 0
        for peak in peaks:
            set: dict = {
                "x": x[start:peak],
                "y": y[start:peak],
                "z": z[start:peak]
            }
            data_sets.append(set)
            start = peak

        fig = plt.figure()
        ax = plt.axes(projection="3d")

        # Plot all the data sets
        for set in data_sets:
            ax.plot3D(set["x"], set["y"], set["z"], "blue")
        
        # Plot the GMM path
        if len(self.__path["path"]) > 0:
            ax.plot3D(self.__path["path"]["x"], self.__path["path"]["y"], self.__path["path"]["z"], "green")

        ax.set_title("GMM")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.show()
        plt.pause(0)


    def get_path(self) -> tuple():
        """Get the found path with the covariance matrix for each step

        Returns:
            tuple(): List of the path, and list of the covariance matrices.
        """
        return tuple(self.__path["path"], self.__path["covariances"])


def test():
    print("This is a test")


if __name__ == "__main__":
    import csv

    GMM_data: list = []
    with open("Matlab\GMM_data.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for data in csv_reader:
            GMM_data.append(data)

    test()

    GMM_translation = GMM(data=GMM_data)
    #GMM_translation.add_data_points(data=GMM_data)
    GMM_translation.plot()
    print("Hello")
