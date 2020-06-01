import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Help")
args = vars(ap.parse_args())
for x in os.listdir("dataset/"):
    print(x)
    if os.path.isdir(os.path.join("dataset/"+args["output"])):
        if x == args["output"]:
            print ("Yes")
