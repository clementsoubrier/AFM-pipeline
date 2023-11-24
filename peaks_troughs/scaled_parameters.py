KERNEL_SIZE = 0.1575  # micrometers
SLOPE_STD_CUT = 2.5  # dimensionless
WINDOW_SIZE = 0.1575  # micrometers

SMOOTH_KERNEL_STD = 0.118  # micrometers
MIN_FEATURE_WIDTH = 0.315  # micrometers
MIN_FEATURE_DEPTH = 11.75  # nanometers

MIN_CENTERLINE_LEN = 2.5  # micrometers
MIN_PREPROCESSED_CENTERLINE_LEN = 1.5  # micrometers
MAX_SLOPE_STD = 3  # dimensionless 5
MAX_SLOPE = 1500 # dimensionless (nanometers per micrometers) 1500 
MAX_VAR_SLOPE = 4000 # dimensionless (nanometers per micrometers)

MAX_TRANSLATION = 1.18  # micrometers
MAX_RELATIVE_TRANSLATION=0.5
ALIGNMENT_PENALTY = 10000 # dimensionless (nanometers per micrometers) 50 
WINDOW_RELATIVE_SIZE = 0.1
if MAX_RELATIVE_TRANSLATION+WINDOW_RELATIVE_SIZE>1:
    raise ValueError("MAX_RELATIVE_TRANSLATION+WINDOW_RELATIVE_SIZE<=1 is required") 
QUANTILE_SIZE = 0.3 # or 0.3

PLOT_V_OFFSET = 78.45  # nanometers

REF_PIXEL_SIZE= 10 # nanometers, for the kymograph
REF_VERTI_SCALE= 100 # nanometers, for the kymograph


def get_scaled_parameters(
    pixel_size,
    preprocessing=True,
    peaks_troughs=False,
    filtering=False,
    aligning=False,
    misc=False,
):
    params = {}
    if preprocessing:
        params["kernel_len"] = 1 + round(KERNEL_SIZE / pixel_size)
        params["std_cut"] = SLOPE_STD_CUT
        params["window"] = 1 + round(WINDOW_SIZE / pixel_size)
    if peaks_troughs:
        params["smooth_std"] = SMOOTH_KERNEL_STD / pixel_size
        params["min_width"] = MIN_FEATURE_WIDTH
        params["min_depth"] = MIN_FEATURE_DEPTH
    if filtering:
        params["min_len"] = MIN_CENTERLINE_LEN
        params["min_prep_len"] = MIN_PREPROCESSED_CENTERLINE_LEN
        params["max_der_std"] = MAX_SLOPE_STD
        params["max_der"] = MAX_SLOPE
        params["max_var_der"] = MAX_VAR_SLOPE
    if aligning:
        params["max_translation"] = max(1, round(MAX_TRANSLATION / pixel_size))
        params["max_relative_translation"] = MAX_RELATIVE_TRANSLATION
        params["penalty"] = ALIGNMENT_PENALTY
        params["window_relative_size"] = WINDOW_RELATIVE_SIZE
        params["quantile_size"] = QUANTILE_SIZE
        params["smooth_std"] = SMOOTH_KERNEL_STD / pixel_size
    if misc:
        params["v_offset"] = PLOT_V_OFFSET
    return params
