import os
import sys

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)

from peaks_troughs.group_by_cell import load_dataset, get_peak_troughs_lineage_lists
from peaks_troughs.preprocess import evenly_spaced_resample


def plot_single_centerline(xs, ys, peaks, troughs):
    plt.figure()
    plt.plot(xs, ys, color="black")
    for x_1 in peaks:
        plt.plot(xs[x_1],ys[x_1], marker='v', color="red")
    for x_1 in troughs:
        plt.plot(xs[x_1],ys[x_1], marker='o', color="green")
    plt.xlabel("Curvilign abscissa (µm)")
    plt.ylabel("Height")
    plt.show()


def plot_cell_centerlines(*cells_and_id, dataset=''):   #first cell is the mother, each argument is a tuple (cell, id)
    
    plt.figure()
    cell_centerlines=[]
    cell_peaks=[]
    cell_troughs=[]
    cell_timestamps=[]
    base_time=cells_and_id[0][0][0]["timestamp"]
    for cell_id in cells_and_id:
        cell = cell_id[0]
        roi_id = cell_id[1]
        for frame_data in cell:
            cell_centerlines.append(np.vstack((frame_data["xs"],frame_data["ys"])))
            cell_peaks.append(frame_data["peaks"])
            cell_troughs.append(frame_data["troughs"])
            cell_timestamps.append(frame_data["timestamp"]-base_time)

        peaks_x = []
        peaks_y = []
        troughs_x = []
        troughs_y = []
        for centerline, peaks, troughs,timestamp in zip(cell_centerlines, cell_peaks,
                                            cell_troughs,cell_timestamps):
            xs = centerline[0, :]
            ys = centerline[1, :] + 2*timestamp 
            if peaks.size:
                peaks_x.extend(xs[peaks])
                peaks_y.extend(ys[peaks])
            if troughs.size:
                troughs_x.extend(xs[troughs])
                troughs_y.extend(ys[troughs])
            plt.plot(xs, ys, color='k')
        plt.scatter(peaks_x, peaks_y, c="red")
        plt.scatter(troughs_x, troughs_y, c="green")
        
        pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
        for key in pnt_ROI :
            coord_x = []
            coord_y = []
            for elem in pnt_ROI[key]:
                coord_x.append(pnt_list[elem,3])
                coord_y.append(pnt_list[elem,4] + 2*(pnt_list[elem,5]-base_time))
            plt.plot(coord_x, coord_y, color = 'b')
        
        
    
    plt.title(roi_id)
    plt.show()

def kymograph(*cells_and_id,  dataset=''):   #first cell is the mother, each argument is a tuple (cell, id)
    ax = plt.axes(projection='3d')
    cell_centerlines=[]
    cell_centerlines_renorm=[]
    cell_peaks=[]
    cell_troughs=[]
    cell_timestamps=[]
    base_time=cells_and_id[0][0][0]["timestamp"]
    pixelsize=[]
    for cell_id in cells_and_id:
        cell = cell_id[0]
        for frame_data in cell:
            pixelsize.append((frame_data["xs"][-1]-frame_data["xs"][0])/(len(frame_data["xs"])-1))
    
    step=min(pixelsize)
    for cell_id in cells_and_id:
        cell = cell_id[0]
        roi_id = cell_id[1]
        for frame_data in cell:
            xs,ys=frame_data["xs"],frame_data["ys"]
            cell_centerlines.append(np.vstack((xs,ys)))
            xs,ys=evenly_spaced_resample(xs,ys,step)
            cell_centerlines_renorm.append(np.vstack((xs,ys)))
            cell_peaks.append(frame_data["peaks"])
            cell_troughs.append(frame_data["troughs"])
            cell_timestamps.append(frame_data["timestamp"]-base_time)

        peaks_x = []
        peaks_y = []
        peaks_z = []
        troughs_x = []
        troughs_y = []
        troughs_z = []
        

        xs_data=np.ravel(np.concatenate([cent[0] for cent in cell_centerlines]))
        xs_min=np.min(xs_data)
        xs_max=np.max(xs_data)
        width=round((xs_max-xs_min)/step+1)
        shape = (len(cell_centerlines), width)
        
        xs_3d = np.zeros(shape, dtype=np.float64)
        ys_3d = np.zeros(shape, dtype=np.float64)
        zs_3d = np.zeros(shape, dtype=np.float64)


        for i, centerline in enumerate(cell_centerlines_renorm):
            xs = centerline[0,:]
            ys = centerline[1,:]
            preval=round((xs[0]-xs_min)/step)
            postval=preval+len(xs)
            xs_3d[i, :preval] = xs[0]
            xs_3d[i, postval:] = xs[-1]
            xs_3d[i, preval:postval]=xs
            ys_3d[i, :] = cell_timestamps[i]
            zs_3d[i, :preval] = ys[0]
            zs_3d[i, postval:] = ys[-1]
            zs_3d[i, preval:postval]=ys
        ax.plot_surface(xs_3d, ys_3d, zs_3d, cmap="viridis", lw=0.5, rstride=1,
                        cstride=1, alpha=0.5, edgecolor='none',
                        norm=mplc.PowerNorm(gamma=1.5))
        

        for centerline, peaks, troughs,timestamp in zip(cell_centerlines, cell_peaks,
                                            cell_troughs,cell_timestamps):
            xs = centerline[0, :]
            ys = centerline[1, :] 
            zs = timestamp * np.ones(len(xs))
            if peaks.size:
                peaks_x.extend(xs[peaks])
                peaks_y.extend(ys[peaks])
                peaks_z.extend([timestamp]*len(peaks))
            if troughs.size:
                troughs_x.extend(xs[troughs])
                troughs_y.extend(ys[troughs])
                troughs_z.extend([timestamp]*len(troughs))
            ax.plot3D(xs, zs, ys,c="k")
        ax.scatter(peaks_x,  peaks_z, peaks_y,c="red")
        ax.scatter(troughs_x, troughs_z, troughs_y, c="green")
        
        
        pnt_list, pnt_ROI = get_peak_troughs_lineage_lists(dataset, roi_id)
        for key in pnt_ROI :
            coord_x = []
            coord_y = []
            coord_z = []
            for elem in pnt_ROI[key]:
                coord_x.append(pnt_list[elem,3])
                coord_y.append(pnt_list[elem,4] )
                coord_z.append(pnt_list[elem,5]-base_time)
            ax.plot(coord_x, coord_z, coord_y, color = 'b')
    ax.set_zlabel(r'height ($nm$)')
    ax.set_ylabel(r'time ($min$)')
    ax.set_xlabel(r' centerline lenght ($\mu m$)')

    plt.title(roi_id)
    plt.show()

def main():
    dataset = os.path.join("WT_mc2_55", "30-03-2015")
    for roi_id, cell in load_dataset(dataset, False):
        if len(cell)>1:
            kymograph((cell, roi_id), dataset=dataset)
        # plot_cell_centerlines((cell, roi_id), dataset=dataset)
    plt.show()



if __name__ == "__main__":
    main()
