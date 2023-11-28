import sys

import numpy as np

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from peaks_troughs.group_by_cell import load_dataset

def surface_feature_tracking(cell,max_drift,max_depth):
    generations=len(cell)
    peaks_list=[]
    troughs_list=[]
    for frame_data in cell:

        x_peak=frame_data["xs"][frame_data["peaks"]]
        y_peak=frame_data["ys"][frame_data["peaks"]]
        peaks_list.append(np.vstack((x_peak,y_peak)))

        x_trough=frame_data["xs"][frame_data["troughs"]]
        y_trough=frame_data["ys"][frame_data["troughs"]]
        troughs_list.append(np.vstack((x_trough,y_trough)))
    
    

        



def main():
    pass

if __name__=="__main__":
    main()