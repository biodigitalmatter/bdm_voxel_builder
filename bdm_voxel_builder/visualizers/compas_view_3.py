
import os
from compas.colors import Color
from compas.geometry import Pointcloud
from helpers import *
import json
import argparse

def get_nth_newest_file_in_folder_(folder_path, n, sort_by_time = True):
    try:
        # Get a list of files in the folder
        files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
        if sort_by_time:
            # Sort the files by change time (modification time) in descending order
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        else:
            # sort files by name
            files.sort(key=lambda x: os.path.split(x)[1], reverse=True)
            
        # Return the newest file
        if files:
            return files[min(n, len(files))]
        else:
            print("Folder is empty.")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Create the parser
parser = argparse.ArgumentParser(description="provide int variable: i.")
# Add an argument
parser.add_argument('nth', type=int, help='i : nth json to open')
try:
    # Parse the command-line arguments
    args = parser.parse_args()
    Nth = int(args.nth)
except Exception as e:
    # print()
    Nth = 0


# params
# Nth = 0
show = True
radius = 1
folder_path = os.path.join(os.getcwd(), 'data/temp')
filename = get_nth_newest_file_in_folder_(folder_path, 3, False)
print(filename)
f = open(filename)
data = json.load(f)
pts = data['pointcloud']['data']['points']
try: 
    try:
        ptcloud = Pointcloud(pts)
    except Exception as e:
        print('first error:', e)
    # network = Network.from_json(file)
    
    # =============================================================================
    if show: # SHOW network
        from compas_view2.app import App

        viewer = App(width=600, height=600)
        viewer.view.camera.rx = -60
        viewer.view.camera.rz = 100
        viewer.view.camera.ty = -2
        viewer.view.camera.distance = 20

        viewer.add(ptcloud)

        green = Color.green()
        green = Color(135/255, 150/255, 100/255)
        print('show starts')
        viewer.show()


except Exception as e:
    print(f"Error: {e}")

print('show done')

