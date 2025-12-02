#!/usr/bin/env python

# Copyright 2025 David Krach

# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated documentation files (the “Software”), to deal in 
# the Software without restriction, including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
# Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

# author: dk, david.krach@mib.uni-stuttgart.de


# -- Header --------------------------------- #
import os, sys
import numpy as np
import scipy
from skimage import io, filters, measure, morphology
from scipy import ndimage
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import warnings
# ------------------------------------------- #



# --- Create circular mask ---

def create_mask(radius, img, c_x, c_y):
    """
    Creates a circular mask of given radius on an image.

    Args:
        radius: Radius of the circle.
        img: Image on which the mask is to be created.
        c_x, c_y: Center offsets for the circle.

    Returns:
        circular_mask: Binary mask (1 inside the circle, 0 outside).
    """
    m, n = img.shape
    circular_mask = np.zeros((m, n), dtype=np.int8)
    y, x = np.ogrid[-c_x:m-c_x, -c_y:n-c_y]
    mask = x*x + y*y <= radius*radius
    circular_mask[mask] = 1
    return circular_mask


# --- Save results to XML ---

def writestruct(data, filename):
    """
    Saves a dictionary to an XML file.

    Args:
        data: Dictionary to save.
        filename: Path to the output XML file.
    """
    root = ET.Element("results")
    for key, value in data.items():
        child = ET.SubElement(root, key)
        child.text = str(value)
    tree = ET.ElementTree(root)
    
    tree.write(filename)


# --- Porosity of 3d numpy array ---

def porosity(array, fluid):
    """
    

    Parameters
    ----------
    array : ndarray
        DESCRIPTION.
    fluid : int 
        DESCRIPTION.

    Returns
    -------
    float, porosity of the sample. simple computation not caring 
            about disconnected pores 

    """
    
    return (np.count_nonzero(array == fluid))/(array.shape[0]*array.shape[1]*array.shape[2])


# --- Export raw function for poremaps data

def export_raw(filename, array):
    """
    

    Parameters
    ----------
    filename : str
        DESCRIPTION.
    array : numpy.ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    # ‘F’ means to flatten in column-major (Fortran- style) order
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    
    
    array = array.flatten('F')
    
    
    # This tests every element of an array if it is boolean or not
    for i in range(array.shape[0]):
        if array[i].dtype != bool:
            raise ValueError(f'{array[i].dtype} Not a boolean array! Check Datatype')
    
    file = open(filename, mode = 'w')
    
    array.tofile(file)
    file.close()
    
    print(f'Writing raw file {filename} done.')



# --- Mirror raw domain in 3 directions --- 

def mirrorall3axes(array):
    """

    Parameters
    ----------
    array : numpy ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """

    print(f'Mirror axes: Initial size: {array.shape}')
    g2 = np.flip(array, 0)
    array = np.concatenate((array, g2), axis = 0)
    g3 = np.flip(array, 1)
    array = np.concatenate((array, g3), axis = 1)
    g4 = np.flip(array, 2)
    array = np.concatenate((array, g4), axis = 2)
    print(f'Mirror axes: Returned size: {array.shape}')
    
    return array


# --- Mirror raw domain in main direction --- 


def mirror_flow_dir(array):
    """
    

    Parameters
    ----------
    array : numpy ndarray
        DESCRIPTION.

    Returns
    -------
    None.

    """
    print(f'Mirror main direction: Initial size: {array.shape}')
    g4 = np.flip(array, 2)
    array = np.concatenate((array, g4), axis = 2)
    print(f'Mirror main direction: Returned size: {array.shape}')
    
    return array

# --- Swap data, 0 -> 1, 1 -> 0 --- 

def swap_0_1(array):
    """
    
    Parameters
    ----------
    array : numpy ndarray
        DESCRIPTION.

    Returns
    -------
    numpy ndarray.

    """
    
    array[array == 0]  = 99
    array[array == 1]  = 0
    array[array == 99] = 1
    
    return array


# --- Remove porespace not connected to any flow path --- 

def eliminate_unconnected_porespace(array):
    """
    

    Parameters
    ----------
    array : TYPE
        Important!!!
        Data input: solid must be zero 0 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
        An array-like object to be labeled. Any non-zero values in input 
        are counted as features and zero values are considered the background.

    Returns
    -------
    None.

    Since 0 is considered the background swap first!

    """
    
    array = swap_0_1(array) # one is now fluid
    
    warnings.warn(f'THIS IS NOT WORKING WITH SOLID FRAMES!')
    
    labeled_array, num_features = scipy.ndimage.label(array)

    print(f'Number of features: {num_features}')

    # get list of features in first and last slide
    features_first_slide = np.unique(labeled_array[:, :, 0])
    features_last_slide  = np.unique(labeled_array[:, :, -1])

    # common features, remove 0 since solid anyhow
    common_features = np.intersect1d(features_first_slide, features_last_slide)
    if np.all(common_features) == False:
        common_features = np.delete(common_features, 0)
    print(f'Number of common features: {common_features.shape[0]}')

    labeled_array[labeled_array == 0] = 255
    
    
    # all voxels on percolating paths set to 0
    for i in common_features:
        labeled_array[labeled_array == i] = 0

    # all disconnected pores are set to 1 if not part of the non connected features
    labeled_array[labeled_array != 0] = 1
    labeled_array[labeled_array == 255] = 1
    
    
    return labeled_array



# --- Create input file for poremaps solver ---

def create_input_file(  fn, 
                        size, 
                        vs, 
                        max_iter = 100000001, it_eval = 100, it_write = 100, 
                        solving_algorithm = 2, 
                        eps = 1.0e-6,
                        dom_decomposition = [0,0,0],
                        dom_of_interest = [0, 0, 0, 0, 0, 0],
                        write_output = [1, 1, 0, 0],
                        porosity = -1.0,
                        boundary_method = 0):
    """
    

    Parameters
    ----------
    fn : TYPE
        DESCRIPTION.
    size : TYPE
        DESCRIPTION. Should look like [100, 100, 100].
    vs : TYPE
        DESCRIPTION.
    max_iter : TYPE, optional
        DESCRIPTION. The default is 100000001.
    it_eval : TYPE, optional
        DESCRIPTION. The default is 100.
    it_write : TYPE, optional
        DESCRIPTION. The default is 100.
    solving_algorithm : TYPE, optional
        DESCRIPTION. The default is 2.
    eps : TYPE, optional
        DESCRIPTION. The default is 1.0e-6.
    dom_decomposition : TYPE, optional
        DESCRIPTION. The default is [0,0,0].
    dom_of_interest : TYPE, optional
        DESCRIPTION. The default is [0, 0, 0, 0, 0, 0].
    dom_of_interest : TYPE, optional
        DESCRIPTION. The default is [1, 1, 0, 0].
    porosity : TYPE, optional
        DESCRIPTION. The default is -1.0.
    boundary_method : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.
    """
    
    permeability_fn = 'permeability_{0}.log'.format(str(fn.rsplit('.raw')[0]))
    input_fn = 'input_{0}.inp'.format(str(fn.rsplit('.raw')[0]))
    
    filestring = f"""dom_decomposition {dom_decomposition[0]} {dom_decomposition[1]} {dom_decomposition[2]}
boundary_method {boundary_method}
geometry_file_name {fn}
size_x_y_z  {size[0]} {size[1]} {size[2]}
voxel_size  {vs}
max_iter    {max_iter}
it_eval    {it_eval}
it_write   {it_write}
log_file_name {permeability_fn}
solving_algorithm {solving_algorithm}
eps {eps}
porosity {porosity}
dom_interest {dom_of_interest[0]} {dom_of_interest[1]} {dom_of_interest[2]} {dom_of_interest[3]} {dom_of_interest[4]} {dom_of_interest[5]}
write_output {write_output[0]} {write_output[1]} {write_output[2]} {write_output[3]}\n"""

    
    
    textfile = open(input_fn, "w")
    a = textfile.write(filestring)
    textfile.close()
    
    print(f'Textfile {input_fn} saved!')
    
    return input_fn


# --- Ploting made shorter ---

def _plot_comp(a, b):
    """
    

    """

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(a)
    bar1 = plt.colorbar(im1)
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(b)
    bar2 = plt.colorbar(im2)
    plt.show()
    plt.clf()
    plt.close()



def _plot_single(a):
    """
    """

    im1 = plt.imshow(a)
    bar1 = plt.colorbar(im1)
    plt.show()
    plt.clf()
    plt.close()


# -------------------------------------------------------------------


# --- Configuration ---
MATERIAL = 'B1_15' # 'CT_32' or 'B1_15'
METHOD   = 'CORE' # 'CORE' or 'ANISO'
BINARIZE = False
DEBUG   = False
PLOT    = False
MASKING = True
MEDIAN_FILTER = True
POREMAPS_DATA = True
RESOLUTION = 'high' # 'low', 'med', 'high'
SLICE = 2 if DEBUG else 300
VOL_THRESHOLD = 4*4*4


# --- Define material specific parameters
# --- Adjust if required 

if MATERIAL == 'B1_15':
    res_dirs = {
            'low':  '00_entire_domain_low',    
            'med':  '01_entire_domain_med',
            'high': '02_entire_domain_high',
               }
    res_vs  =  { # in millimeter
            'low':  0.036150,
            'med':  0.024930, 
            'high': 0.010970, 
               }
    res_size = {
            'low':  [628, 647],
            'med':  [881, 910],
            'high': [968, 968],
               }
    res_bound = {
            'low': [[22,602], [22,602], [112,628]],
            'med': [[26,890], [8,872], [152,902]],
            'high': [[4,950], [4,950], [0,659]],
               }
    fname_prefix =  {
            'low': 'B1_15_LowRes',
            'med': 'B1_15_MidRes',
            'high': 'B1_15_HighRes',
                    }
    pm_aniso_lim = {
            'low': [[90,490], [90,490]],
            'med': [[131,731], [131,731]],
            'high': [[147,817], [147,817]],
    }


elif MATERIAL == 'CT_32':
    res_dirs = {
            'low':  '00_entire_domain_low',    
            'med':  '01_entire_domain_med',
            'high': '02_entire_domain_high',
               }
    res_vs  =  { # in millimeter
            'low':   0.033820,
            'med':  0.023990, 
            'high': 0.011260, 
               }
    res_size = {
            'low':  [663, 657],
            'med':  [934, 934],
            'high': [968, 968],
               }
    res_bound = {
            'low': [[14,650], [6,642], [58,650]],
            'med': [[18,918], [4,904], [84,842]],
            'high': [[6,960], [6,960], [0,646]],
               }
    fname_prefix =  {
            'low': 'CT_32_LowRes',
            'med': 'CT_32_MidRes',
            'high': 'CT_32_HighRes',
                    }
    pm_aniso_lim = {
            'low': [[100,536], [100,536]],
            'med': [[140,760], [140,760]],
            'high': [[145,809], [145,809]],
    }
else:
    raise ValueError('MATERIAL not avail.')


folder = f'{MATERIAL}/{res_dirs[RESOLUTION]}' 
save_folder = f'binarized/{folder}'
voxel_size = res_vs[RESOLUTION]  # millimeter
rows, columns = res_size[RESOLUTION][1], res_size[RESOLUTION][0] 

# --- Create directories for saving results ---
os.makedirs(os.path.join(save_folder, 'binarized'), exist_ok=True)
os.makedirs(os.path.join(save_folder, 'binarized_masked'), exist_ok=True)
# os.makedirs(os.path.join(save_folder, 'binarized_masked_labeled'), exist_ok=True)
os.makedirs(os.path.join(save_folder, 'binarized_core'), exist_ok=True)
os.makedirs(os.path.join(save_folder, 'binarized_aniso'), exist_ok=True)

if BINARIZE:
    # --- Load images ---
    # List all .tif files in the 'reconstructed' subfolder
    fdata = [f for f in os.listdir(folder) if f.endswith('.tif')]
    fdata.sort()  # Ensure files are in order
    num_slices = len(fdata)

    print(f'Number of slices found: {num_slices}')

    # Initialize 3D array to store image data
    imp_data_3d = np.zeros((rows, columns, num_slices), dtype=np.uint16)

    # Read each image into the 3D array
    for i, filename in enumerate(fdata):
        imp_data_3d[:, :, i] = io.imread(os.path.join(folder, filename))

    _inital_size = imp_data_3d.shape
    # Cut the stack in 2 lateral directions
    imp_data_3d = imp_data_3d[res_bound[RESOLUTION][1][0]:res_bound[RESOLUTION][1][1], 
                              res_bound[RESOLUTION][0][0]:res_bound[RESOLUTION][0][1],
                              res_bound[RESOLUTION][2][0]:res_bound[RESOLUTION][2][1]] 
    print(f'Size of array reduced from {_inital_size} to {imp_data_3d.shape}.')

    if PLOT: _plot_single(imp_data_3d[:, :, SLICE])


    if DEBUG:
        imp_data_3d = imp_data_3d[:, :, 300:303]
        print(f'Debuging image size: {imp_data_3d.shape}')

    # --- Noise reduction using 3D median filter ---
    if MEDIAN_FILTER:
        imp_data_median_3d = ndimage.median_filter(imp_data_3d, size=2)
    else:
        imp_data_median_3d = np.copy(imp_data_3d)

    if PLOT: _plot_comp(imp_data_3d[:, :, SLICE], imp_data_median_3d[:, :, SLICE])

    # Create mask for the first slice (assuming all slices have the same dimensions)
    mask = create_mask(int(imp_data_median_3d.shape[0]/2), imp_data_median_3d[:, :, 0], int(imp_data_median_3d.shape[0]/2), int(imp_data_median_3d.shape[1]/2))

    # --- Thresholding ---
    # Initialize 3D array for binarized data
    bdata_3d = np.zeros((imp_data_median_3d.shape), dtype=np.uint8)

    # Apply multi-Otsu thresholding to each slice
    for i in range(imp_data_median_3d.shape[2]):
        print(f'Masking and Multi-Otsu Thresholding image {i+1} of {imp_data_median_3d.shape[2]}')
        # Apply mask if required
        if MASKING:
            imp_data_3d_masked = imp_data_median_3d[:, :, i] * mask
        else:
            imp_data_3d_masked = imp_data_median_3d[:, :, i]

        pixels_for_threshold = imp_data_3d_masked[imp_data_3d_masked > 0]
        if len(pixels_for_threshold) > 0:
            thresh = filters.threshold_multiotsu(pixels_for_threshold, classes=3)
            bdata_3d[:, :, i] = np.digitize(imp_data_median_3d[:, :, i], bins=thresh)

    if PLOT: _plot_comp(imp_data_3d[:, :, SLICE], bdata_3d[:, :, SLICE])

    # --- Skeletonize: Identify pore and skeleton regions ---
    bdata_3d_skeleton = np.logical_or(bdata_3d == 1, bdata_3d == 2)
    # print(type(bdata_3d_skeleton[0,0, 0]))
    # bdata_3d_skeleton = np.zeros(bdata_3d.shape, dtype = bool)
    # bdata_3d_skeleton[bdata_3d == 2] = True

    if PLOT: _plot_comp(imp_data_3d[:, :, SLICE], bdata_3d_skeleton[:, :, SLICE])

    # --- Remove small objects (islands) and holes ---
    # Invert the skeleton to remove small objects (islands)
    bdata_3d_skeleton_cleaned = morphology.remove_small_objects(~bdata_3d_skeleton, 
                                                                min_size = VOL_THRESHOLD, 
                                                                connectivity = 1)
    # Remove small holes
    bdata_3d_skeleton_cleaned = morphology.remove_small_holes(bdata_3d_skeleton_cleaned, 
                                                              area_threshold = VOL_THRESHOLD, 
                                                              connectivity = 1)

    if PLOT: _plot_comp(imp_data_3d[:, :, SLICE], bdata_3d_skeleton_cleaned[:, :, SLICE])

    # --- Apply circular mask to cleaned data ---
    bdata_cleaned_3d_masked = np.zeros(bdata_3d_skeleton_cleaned.shape, dtype = bool)
    for i in range(bdata_cleaned_3d_masked.shape[2]):
        bdata_cleaned_3d_masked[:, :, i] = np.logical_or(~bdata_3d_skeleton_cleaned[:, :, i], ~mask.astype(bool))

    bdata_cleaned_3d_masked = ~bdata_cleaned_3d_masked
    if PLOT: _plot_comp(imp_data_3d[:, :, SLICE], bdata_cleaned_3d_masked[:, :, SLICE])

    # # --- Save binarized and masked images ---
    for i in range(bdata_3d_skeleton_cleaned.shape[2]):
        print(f'Writing image {i+1} of {bdata_3d_skeleton_cleaned.shape[2]}')
        io.imsave(os.path.join(save_folder, 'binarized', f'{fname_prefix[RESOLUTION]}_{i}.tif'),
                  bdata_3d_skeleton_cleaned[:, :, i].astype(np.uint8) * 255)
        io.imsave(os.path.join(save_folder, 'binarized_masked', f'{fname_prefix[RESOLUTION]}_{i}.tif'),
                  bdata_cleaned_3d_masked[:, :, i].astype(np.uint8) * 255)

    # --- Evaluate data: Label connected components and compute pore volumes ---
    print(f'Evaluate clusters')
    bdata_cleaned_3d_masked_eval = ~bdata_cleaned_3d_masked
    if PLOT: _plot_comp(imp_data_3d[:, :, SLICE], bdata_cleaned_3d_masked_eval[:, :, SLICE])

    connected_clusters = measure.label(bdata_cleaned_3d_masked_eval)
    pores = measure.regionprops(connected_clusters)

    num_slices = bdata_cleaned_3d_masked.shape[2]

    # --- Quantifications ---
    results = {
        'data_set': folder,
        'voxel_size': voxel_size,
        'units': 'voxel',
        'mask_diameter': 2 * int(imp_data_median_3d.shape[0]/2),
        'number_of_slices': num_slices,
        'mask_inner_volume': num_slices * np.sum(mask),
        'mask_outer_volume': num_slices * rows * columns - num_slices * np.sum(mask),
        'bulk_volume': num_slices * np.sum(mask),
        'solid_volume': np.sum(bdata_cleaned_3d_masked),
        'pore_volume_total': num_slices * np.sum(mask) - np.sum(bdata_cleaned_3d_masked),
        'pore_volume_dead_end': sum(p.area for p in pores) - max(p.area for p in pores) if pores else 0,
        'porosity_total': (num_slices * np.sum(mask) - np.sum(bdata_cleaned_3d_masked)) / (num_slices * np.sum(mask)),
        'porosity_dead_ends': (sum(p.area for p in pores) - max(p.area for p in pores)) / (num_slices * np.sum(mask)) if pores else 0,
        'porosity_effective': ((num_slices * np.sum(mask) - np.sum(bdata_cleaned_3d_masked)) / (num_slices * np.sum(mask))) -
                              ((sum(p.area for p in pores) - max(p.area for p in pores)) / (num_slices * np.sum(mask))) if pores else 0,
        }



    # --- Write struct to xml and print results 
    writestruct(results, os.path.join(save_folder, 'results.xml'))

    print(f'\n{results}\n')


if POREMAPS_DATA:
    # --- For poremaps data load the binarized data
    # --- Check if the folder exists
    if METHOD == 'ANISO':
        in_subfolder = 'binarized'
    elif METHOD == 'CORE':
        in_subfolder = 'binarized_masked'
    else:
        raise ValueError(f'METHOD is {METHOD}. Not defined properly.')
    
    tif_input_folder = os.path.join(save_folder, in_subfolder)
    if not os.path.isdir(tif_input_folder):
        raise FileNotFoundError(f'No folder exists at {tif_input_folder}')

    # --- Load data. Check for max RAM and run both parts in 2 steps if req.
    tif_data = [f for f in os.listdir(tif_input_folder) if f.endswith('.tif')]
    tif_data.sort()
    
    # --- Load one slice do determine the size of the stack
    number_of_slices = len(tif_data)
    lateral_size     = io.imread(os.path.join(tif_input_folder, tif_data[0])).shape
    print(f'Size of the loaded tif stack: [{lateral_size[0]}, {lateral_size[1]}, {number_of_slices}]')

    # Initialize 3D array to store image data
    imp_data_3d = np.zeros((lateral_size[0], lateral_size[1], number_of_slices), dtype=np.uint16)

    # Read each image into the 3D array
    for i, filename in enumerate(tif_data):
        imp_data_3d[:, :, i] = io.imread(os.path.join(tif_input_folder, filename))

    # Swap 
    imp_data_3d = imp_data_3d/255
    imp_data_3d = swap_0_1(imp_data_3d)
    imp_data_3d = np.uint16(imp_data_3d)

    print(f'Loaded data has following specs: Values {np.unique(imp_data_3d)}, dtype: {type(imp_data_3d[0, 0, 0])}, porosity {porosity(imp_data_3d, 0)}')

    if METHOD == 'ANISO':
        # --- Cut to the biggest possible cube, limits defined by hand
        imp_data_3d = imp_data_3d[pm_aniso_lim[RESOLUTION][1][0]:pm_aniso_lim[RESOLUTION][1][1], 
                              pm_aniso_lim[RESOLUTION][0][0]:pm_aniso_lim[RESOLUTION][0][1],
                              :]
                              # pm_aniso_lim[RESOLUTION][2][0]:pm_aniso_lim[RESOLUTION][2][1]]
        imp_data_3d = mirrorall3axes(imp_data_3d)
        
        r1 = np.copy(imp_data_3d)
        r2 = np.copy(imp_data_3d)
        all_imp_data = [imp_data_3d, np.transpose(r1, (2, 0, 1)), np.transpose(r2, (1, 2, 0))]

        # --- Loop over all 3 directions for the cube dataset
        for i, _imp_data_3d in enumerate(all_imp_data):

            overall_porosity = porosity(_imp_data_3d, fluid = 0)
            print(f'overall_porosity: {overall_porosity}')
            _imp_data_3d = eliminate_unconnected_porespace(_imp_data_3d)
            effective_porosity = porosity(_imp_data_3d, fluid = 0)
            print(f'effective_porosity: {effective_porosity}')
        
            if effective_porosity == 0.0:
                print('No file written since effective porosity is zero.')
            else:
                _imp_data_3d = np.array(_imp_data_3d, dtype=np.bool_)
                filename    = f'{MATERIAL}_{METHOD}_{RESOLUTION}_dir{i}_{_imp_data_3d.shape[0]}_{_imp_data_3d.shape[1]}_{_imp_data_3d.shape[2]}_vs{res_vs[RESOLUTION]}_porosity{effective_porosity}.raw'
                export_raw(filename, _imp_data_3d)
                input_fn    = create_input_file(filename, _imp_data_3d.shape, res_vs[RESOLUTION], eps = 1.0e-6, porosity = effective_porosity)


    elif METHOD == 'CORE':
        imp_data_3d = mirror_flow_dir(imp_data_3d)
        imp_data_3d = eliminate_unconnected_porespace(imp_data_3d)
        imp_data_3d = np.array(imp_data_3d, dtype=np.bool_)
        
        filename    = f'{MATERIAL}_{METHOD}_{imp_data_3d.shape[0]}_{imp_data_3d.shape[1]}_{imp_data_3d.shape[2]}_vs{res_vs[RESOLUTION]}.raw'
        export_raw(filename, imp_data_3d)
        
        # Use effective porosity from xml file
        tree = ET.parse(os.path.join(save_folder, 'results.xml'))
        root = tree.getroot()
        effective_porosity = float(root.find('porosity_effective').text)
        
        input_fn    = create_input_file(filename, imp_data_3d.shape, res_vs[RESOLUTION], eps = 1.0e-6, porosity = effective_porosity)











