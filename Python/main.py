import GMM

import atexit

def goodbye(name, adjective):
    print(f"Goodbye {name}. You are {adjective}.")

atexit.register(goodbye, "Kasper", "a legend")
atexit.register(goodbye, "Emil", "slightly below  average")

GMM.test()