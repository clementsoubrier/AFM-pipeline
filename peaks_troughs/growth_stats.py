import itertools
import os
import sys
import statistics

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy import stats

package_path = '/home/c.soubrier/Documents/UBC_Vancouver/Projets_recherche/AFM/afm_pipeline'
if not package_path in sys.path:
    sys.path.append(package_path)
from scaled_parameters import get_scaled_parameters
from peaks_troughs.group_by_cell import Orientation, load_cell, load_dataset



def extract_growth(roi):
    t_0 = roi[0]["timestamp"]
    roi = iter(roi)
    while True:
        try:
            frame_data = next(roi)
        except StopIteration:
            return [], [], [], []
        orientation = Orientation(frame_data["orientation"])
        xs = frame_data["xs"]
        match orientation:
            case Orientation.UNKNOWN:
                continue
            case Orientation.NEW_POLE_OLD_POLE:
                old_pole_0 = xs[-1]
                new_pole_0 = xs[0]
            case Orientation.OLD_POLE_NEW_POLE:
                old_pole_0 = xs[0]
                new_pole_0 = xs[-1]
        break
    
    
    
    timestamps = []
    old_pole_growth = []
    new_pole_growth = []
    overall_growth = []
    for frame_data in roi:
        timestamp = frame_data["timestamp"]
        orientation = Orientation(frame_data["orientation"])
        xs = frame_data["xs"]
        match orientation:
            case Orientation.UNKNOWN:
                continue
            case Orientation.NEW_POLE_OLD_POLE:
                old_pole_var = xs[-1] - old_pole_0
                new_pole_var = new_pole_0 - xs[0]
            case Orientation.OLD_POLE_NEW_POLE:
                old_pole_var = old_pole_0 - xs[0]
                new_pole_var = xs[-1] - new_pole_0
        timestamps.append(timestamp - t_0)
        old_pole_growth.append(old_pole_var)
        new_pole_growth.append(new_pole_var)
        overall_growth.append(abs(xs[-1]-xs[0]))


    return np.array(timestamps), np.array(old_pole_growth), np.array(new_pole_growth), np.array(overall_growth)

def str_orient(orientation):
    match orientation:
        case Orientation.UNKNOWN:
            ori_str = "unknown orientation"
        case Orientation.NEW_POLE_OLD_POLE:
            ori_str = "new to old orientation"
        case Orientation.OLD_POLE_NEW_POLE:
            ori_str = "old new orientation"
    return ori_str



def plot_piecewise_linear_reg_old_new(timestamps, old_pole_growth, new_pole_growth, roi_name, outlier_detection=False):
    plt.figure()

    new_times = timestamps
    old_times = timestamps
    
    
    if outlier_detection:
        new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
        old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
        
    a_1, a_2, b, t = piecewise_pointwise_linear_regression(new_times, new_pole_growth)
    t_0 = new_times[0]
    t_f = new_times[-1]
    piecewise_times = [t_0, t, t_f]
    p_0 = b - a_1 * t
    p_f = b + a_2 * (t_f - t)
    piecewise_y = [p_0, b, p_f]
    
    plt.scatter(new_times, new_pole_growth, label="new pole length", color="C2")
    plt.plot(
        piecewise_times, piecewise_y, color="C2"
    )
    
    plt.scatter(old_times, old_pole_growth, label="old pole length", color="C0")
    slope, intercept = statistics.linear_regression(old_times, old_pole_growth)
    lin_int = slope*old_times+intercept
    plt.plot(old_times, lin_int,color="C0")
    
    
    y_min = min(min(new_pole_growth), min(piecewise_y), min(old_pole_growth), min(lin_int))
    y_max = max(max(new_pole_growth), max(piecewise_y), max(old_pole_growth), max(lin_int))
    plt.plot(
        [t, t],
        [y_min, y_max],
        linestyle="dashed",
        label=f"NETO: {int(t)} minutes",
        color="C1",
    )
    
    
    plt.plot([], [], " ", label=f"Slope ratio: {a_2 / a_1:.2f}")
    plt.annotate(
        f"{1000 * a_1:.1f} nm / minute", (t_0 + (0.05 * (t_f - t_0)), p_0)
    )
    plt.annotate(
        f"{1000 * a_2:.1f} nm / minute",
        (t_f - (0.05 * (t_f - t_0)), p_f),
        ha="right",
        va="top",
    )
    
    
    plt.legend()
    plt.xlabel("Time since cell birth (minutes)")
    plt.ylabel("Pole length (µm)")
    plt.title(f"Growth of {roi_name}")
    plt.show()


def plot_piecewise_linear_reg(timestamps,old_pole_growth, new_pole_growth, overall_growth, roi_name, outlier_detection=False):
    plt.figure()
    overall_growth = np.array(overall_growth)
    overall_times = np.array(timestamps)
    
    
    if outlier_detection:
        overall_times, overall_growth= remove_length_outliers(overall_times , overall_growth)
        # new_times, new_pole_growth = remove_length_outliers(overall_times , new_pole_growth)
        # old_times, old_pole_growth = remove_length_outliers(overall_times , old_pole_growth)
        
    a_1, a_2, b, t = piecewise_pointwise_linear_regression(overall_times, overall_growth)
    t_0 = overall_times[0]
    t_f = overall_times[-1]
    piecewise_times = [t_0, t, t_f]
    p_0 = b - a_1 * t
    p_f = b + a_2 * (t_f - t)
    piecewise_y = [p_0, b, p_f]
    y_min = min(min(overall_growth), min(piecewise_y))
    y_max = max(max(overall_growth), max(piecewise_y))
    plt.scatter(overall_times, overall_growth, label="cell length variation", color="C2")
    plt.plot(
        piecewise_times, piecewise_y, label="piecewise linear fit", color="C0"
    )
    plt.plot(
        [t, t],
        [y_min, y_max],
        linestyle="dashed",
        label=f"NETO: {int(t)} minutes",
        color="C1",
    )
    plt.plot([], [], " ", label=f"Slope ratio: {a_2 / a_1:.2f}")
    plt.annotate(
        f"{1000 * a_1:.1f} nm / minute", (t_0 + (0.05 * (t_f - t_0)), p_0)
    )
    plt.annotate(
        f"{1000 * a_2:.1f} nm / minute",
        (t_f - (0.05 * (t_f - t_0)), p_f),
        ha="right",
        va="top",
    )
    plt.legend()
    
    plt.xlabel("Time since cell birth (minutes)")
    plt.ylabel("Length variation (µm)")
    plt.title(f"Growth of {roi_name}")
    plt.show()
    

def remove_worst_length_outliers(times, y):
    dy = y[1:] - y[:-1]
    dt = times[1:] - times[:-1]
    dydt = dy / dt
    d2ydt = dydt[1:] - dydt[:-1]
    dtbis = (times[2:] - times[:-2]) / 2
    d2ydt2 = d2ydt / dtbis
    q1, q3 = np.percentile(d2ydt2, [25, 75])
    clipped = np.clip(d2ydt2, q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    errors = np.abs(d2ydt2 - clipped)
    i = np.argmax(errors)
    if errors[i] > 0:
        times = np.concatenate([times[: i + 1], times[i + 2 :]])
        y = np.concatenate([y[: i + 1], y[i + 2 :]])
    return times, y


def remove_length_outliers(times, y):
    while len(times) > 0:
        new_times, new_y = remove_worst_length_outliers(times, y)
        if len(new_times) == len(times):
            return np.array(times), np.array(y)
        times = new_times
        y = new_y
    raise RuntimeError


def piecewise_pointwise_linear_regression(times, y):
    assert times[-1] - times[0] >= 75
    best_err = np.inf
    best_a_1 = None
    best_a_2 = None
    best_b = None
    best_t = None
    errs = []
    for i, (t_1, t_2) in enumerate(itertools.pairwise(times), 1):
        if t_2 < 45 or t_1 > times[-1] - 30:
            continue
        t_1 = max(t_1, 45)
        t_2 = min(t_2, times[-1] - 30)
        for t in np.linspace(t_1, t_2, round(t_2 - t_1) + 1):
            mat = np.zeros((len(times), 3))
            mat[:i, 0] = times[:i] - t
            mat[i:, 1] = times[i:] - t
            mat[:, 2] = 1
            coefs = np.linalg.lstsq(mat, y, None)[0]
            a_1, a_2, b = coefs
            err = np.sum((y - np.dot(mat, coefs)) ** 2)
            errs.append(err)
            if err < best_err:
                best_err = err
                best_a_1 = a_1
                best_a_2 = a_2
                best_b = b
                best_t = t
    return best_a_1, best_a_2, best_b, best_t

def compute_slopes(times, y, t):
    mat = np.zeros((len(times), 3))
    i = np.nonzero(times >=t)[0][0]
    mat[:i, 0] = times[:i] - t
    mat[i:, 1] = times[i:] - t
    mat[:, 2] = 1
    coefs = np.linalg.lstsq(mat, y, None)[0]
    a_1, a_2, b = coefs
    return a_1, a_2, b
    


def plot_growth_all_cent(dataset, old_new=True, overall=True,outlier_detection=False):
    
    
    for roi_name, roi in load_dataset(dataset, False):
        timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
        if len(timestamps)>=5 and timestamps[-1]-timestamps[0] > 75:
            if old_new:
                plot_piecewise_linear_reg_old_new(timestamps, old_pole_growth, new_pole_growth, roi_name, outlier_detection=outlier_detection)
            if  overall:
                plot_piecewise_linear_reg(timestamps, old_pole_growth, new_pole_growth, overall_growth, roi_name, outlier_detection=outlier_detection)



def compute_growth_stats(datasetnames, outlier_detection=False):
    
    tot_growth = []
    overall_neto = []
    old_new_neto = []
    overall_slopes = []  # first, second
    old_new_slopes = []  # first new, second new, old
    params = get_scaled_parameters(data_set=True)
    
    if datasetnames in params.keys():
        datasets = params[datasetnames]
    elif isinstance(datasetnames, str): 
        raise NameError('This directory does not exist')
    else :
        datasets = datasetnames 
    
    for dataset in datasets:
        for _, roi in load_dataset(dataset, False):
            timestamps, old_pole_growth, new_pole_growth, overall_growth = extract_growth(roi)
            if len(timestamps)>=5 and timestamps[-1]-timestamps[0] > 75:
                tot_growth.append((overall_growth[-1]-overall_growth[0])/(timestamps[-1]-timestamps[0]))
                new_times = old_times = overall_times = timestamps
                
                if outlier_detection:
                    new_times, new_pole_growth = remove_length_outliers(new_times, new_pole_growth)
                    old_times, old_pole_growth = remove_length_outliers(old_times, old_pole_growth)
                    overall_times, overall_growth= remove_length_outliers(overall_times, overall_growth)
                    
                a_1, a_2, _, t = piecewise_pointwise_linear_regression(new_times, new_pole_growth)
                
                if 0<a_1<=a_2:
                    
                    old_new_neto.append(t)
                
                a_1, a_2, _, t = piecewise_pointwise_linear_regression(overall_times, overall_growth)
                overall_slopes.append([a_1,a_2])
                if 0<a_1<=a_2:
                    
                    overall_neto.append(t)
                # the neto is computed from the overall length because it is more stable
                new_a_1, new_a_2, _, = compute_slopes(new_times, new_pole_growth, t)
                a_3, a_4, _ = compute_slopes(old_times, old_pole_growth, t)
                mat = np.zeros((len(old_times), 2))
                mat[:, 0] = old_times
                mat[:, 1] = 1
                a_5, _ = np.linalg.lstsq(mat, old_pole_growth, None)[0]
                # a_5 = stats.linregress(old_times, old_pole_growth).slope
                
                if new_a_1>0 and new_a_2>0 and a_3>0 and a_4>0  and a_5>0:
                    old_new_slopes.append([new_a_1, new_a_2, a_3, a_4, a_5])
                
                
                
    overall_slopes = np.array(overall_slopes)      
    old_new_slopes = np.array(old_new_slopes)
    
    
    # plt.figure()
    # plt.hist(old_new_neto, 40, color='C0', alpha=0.5, label='Neto new pole', density=True)
    # plt.hist(overall_neto, 40, color='C1', alpha=0.5, label='overall Neto', density=True)
    # plt.title(f"Comparision Neto computation with dataset {datasetnames}")
    # plt.legend()
    _, ax = plt.subplots()
    ax.boxplot([old_new_neto,overall_neto], showfliers=False)
    ax.set_xticklabels(["new pole elongation", "overall elongation"])
    ax.set_ylabel(r"Time $(min)$")
    ax.set_title(f"Neto statistics with 2 methods and dataset {datasetnames}")
    pvalue = stats.ttest_ind(old_new_neto,overall_neto).pvalue
    print(pvalue)
    ax.annotate(f"P={pvalue:.2e}",(1.3,120))
    
    # plt.figure()
    # plt.hist(overall_slopes[:,0], 20, alpha=0.5, label='overall first slope', density=True)
    # plt.hist(overall_slopes[:,1], 20, alpha=0.5, label='overall second slope', density=True)
    # plt.title(f"Overall growth before / after neto with dataset {datasetnames}")
    # plt.legend()
    
    _, ax = plt.subplots()
    ax.boxplot([overall_slopes[:,0],overall_slopes[:,1],overall_slopes[:,1]-overall_slopes[:,0]], showfliers=False)
    ax.set_title(f"Overall cell elongation speed with dataset {datasetnames}")
    ax.set_ylabel(r"elongation speed ($\mu m (min)^{-1}$)")
    ax.set_xticklabels(["Before NETO", "After NETO","Difference before/after"])
    pvalue = stats.ttest_ind(overall_slopes[:,0],overall_slopes[:,1]).pvalue
    print(pvalue)
    ax.annotate(f"P={pvalue:.2e}",(1.3,0.02))

    
    
    _, ax = plt.subplots()
    ax.boxplot([old_new_slopes[:,1], old_new_slopes[:,4]], showfliers=False)
    ax.set_xticklabels(["new pole elongation after NETO", "old pole overall elongation"])
    ax.set_ylabel(r"elongation speed ($\mu m (min)^{-1}$)")
    ax.set_title(f"Comparision of elongation speed for dataset {datasetnames}")
    pvalue = stats.ttest_ind(old_new_slopes[:,1], old_new_slopes[:,4]).pvalue
    print(pvalue)
    x1 = 1
    x2 = 2 
    y = 0.015
    h=0.001
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue:.2e}", ha='center', va='bottom')
    ax.set_ylim(top = 0.017)
    
    
    
    _, ax = plt.subplots()
    ax.boxplot([old_new_slopes[:,0], old_new_slopes[:,1], old_new_slopes[:,2], old_new_slopes[:,3]], showfliers=False)
    ax.set_title(f"Pole elongation speed with dataset {datasetnames}")
    ax.set_ylabel(r"elongation speed ($\mu m (min)^{-1}$)")
    ax.set_xticklabels(["New pole \n before NETO", "New pole \n after NETO","Old pole \n before NETO","Old pole \n after NETO"])
    pvalue = stats.ttest_ind(old_new_slopes[:,0],old_new_slopes[:,1]).pvalue
    pvalue2 = stats.ttest_ind(old_new_slopes[:,2],old_new_slopes[:,3]).pvalue
    print(f"p value new pole {pvalue}, p value old pole {pvalue2}")
    
    x1 = 1
    x2 = 2 
    y = 0.011
    h=0.001
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue:.2e} *", ha='center', va='bottom')
    
    x1 = 3
    x2 = 4
    y = 0.015
    h=0.001
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],  color = 'r')
    ax.text((x1+x2)*.5, y+h, f"P={pvalue2:.2e}", ha='center', va='bottom')
    ax.set_ylim(top = 0.017)
    
    plt.show()   
            





if __name__ == "__main__":
    # plot_growth_all_cent(os.path.join("WT_mc2_55", "30-03-2015")) #, outlier_detection=True
    compute_growth_stats(os.path.join("WT_mc2_55", "30-03-2015"), outlier_detection=True)#'all' 'delta_lamA_03-08-2018' 'WT_no_drug'
