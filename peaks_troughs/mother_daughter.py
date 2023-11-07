import os

import numpy as np

# from align import align_centerlines
from group_by_cell import load_cell
# from plots import plot_kymograph
# from preprocess import get_scaled_parameters, preprocess_centerline


"""
ROI 1   ROI 2   ROI 3
ROI 1   ROI 2   ROI 4
ROI 1   ROI 5   ROI 15
ROI 2   ROI 3   ROI 28
ROI 2   ROI 4   ROI 23
ROI 2   ROI 4   ROI 26
ROI 3   ROI 28  ROI 36
ROI 4   ROI 23  ROI 24
ROI 4   ROI 23  ROI 25
ROI 4   ROI 26  ROI 39
ROI 23  ROI 25  ROI 46
"""

"""
ROI 1   ['ROI 2', 'ROI 5']
ROI 2   ['ROI 3', 'ROI 4']
ROI 4   ['ROI 23', 'ROI 26']
ROI 23  ['ROI 24', 'ROI 25']
ROI 31  ['ROI 32', 'ROI 33']
"""

import matplotlib.pyplot as plt

def main():
    dataset = os.path.join("WT_mc2_55", "30-03-2015")

    # path = os.path.join("data", "datasets", dataset, "Height", "Dic_dir")
    # roi_dict = np.load(os.path.join(path, "ROI_dict.npy"),
    #                    allow_pickle=True).item()
    # for gm_roi_name, gm_roi in roi_dict.items():
    #     for m_roi_name in gm_roi["Children"]:
    #         m_roi = roi_dict[m_roi_name]
    #         for d_roi_name in m_roi["Children"]:
    #             print(gm_roi_name, m_roi_name, d_roi_name, sep="\t")

    # path = os.path.join("data", "datasets", dataset, "Height", "Dic_dir")
    # roi_dict = np.load(os.path.join(path, "ROI_dict.npy"),
    #                    allow_pickle=True).item()
    # for m_roi_name, m_roi in roi_dict.items():
    #     if len(m_roi["Children"]) == 2:
    #         print(m_roi_name, m_roi["Children"], sep="\t")

    # for cell_names in [[1, 2]]:#, [1, 5]]:
    #     centerlines = []
    #     for cell_name in cell_names:
    #         cell = load_cell(cell_name, dataset, True)
    #         for frame_data in cell:
    #             xs = frame_data["xs"]
    #             ys = frame_data["ys"]
    #             pixel_size = frame_data["pixel_size"]
    #             params = get_scaled_parameters(pixel_size, misc=True)
    #             max_translation = params.pop("max_translation")
    #             del params["v_offset"]
    #             centerline = preprocess_centerline(xs, ys, **params)
    #             centerlines.append(centerline)
    #     centerlines = align_centerlines(*centerlines,
    #                                     max_translation=max_translation)
    #     centerlines = [(centerline[:, 0], centerline[:, 1])
    #                 for centerline in centerlines]
    #     plot_kymograph(centerlines, scale=pixel_size)
    j=0
    for i in range(62):
        
        try:
            cell = load_cell(i, dataset, True)
            plt.figure()
            plt.title(f'ROI {i}')
            offset = 0
            for frame_data in cell:
                xs = frame_data["xs"]
                ys = frame_data["ys"]
                ys += offset - np.median(ys)
                if frame_data["no_defect"]:
                    plt.plot(xs, ys)
                else:
                    
                    j+=1
                offset += 30
        except:
            continue
    plt.show()
    

if __name__ == '__main__':
    main()
