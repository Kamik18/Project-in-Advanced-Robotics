
class GMM:
    __path:dict = {
        "path":list(),
        "covariances": list()
    }
    """Path

    Args:
        path: List of points or quaternions representing the path.
        Covariances: List of covariances matrices for each point on the path.
    """

    __data_points:list = []

    def __init__(self) -> None:
        pass

    def add_data_points(self, data:list)-> None:
        """Add data points to the list

        Args:
            data (list): list of data points, translation: (index, x, y, z) or quaternion (index, a, i, j, k).
        """
        self.__data_points.append(data)

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

    GMM_data:list = []
    with open('Matlab\GMM_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for data in csv_reader:
            GMM_data.append(data)

    test()

    GMM_translation = GMM()
    GMM_translation.add_data_points(data=GMM_data)
    print("Hello")