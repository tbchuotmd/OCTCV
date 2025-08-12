import os
from pathlib import Path
import math, sympy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import magic
import cv2

class GridDimensionOptimizer:
    def __init__(self):
        pass

    def optimal_grid(self, n, preferred_ncols=3, max_cols=None):
        """ 
        Calculate the optimal number of rows and columns for a grid of subplots in a figure, aiming to:

        1. Minimize Empty Subplots
        2. Keep number of columns close to preferred_ncols (but can adjust if needed to better accomodate the other criteria)
        3. Try to keep aspect ratio close to 1 (i.e., try to create more square subplots / avoid vertically elongated figures, taking into account preferred_ncols and max_cols)

        PARAMETERS
        ----------
        n: total number of subplots
        preferred_ncols: preferred number of columns (default: 3)
        max_cols: maximum number of columns (default: None)

        RETURNS
        -------
        tuple: (nrows, ncols)

        EXAMPLES
        --------

        |   N | r x c | aspect_ratio | total_subplots | empty_subplots | N_is_prime   |
        |-----|-------|--------------|----------------|----------------|--------------|
        |   1 | 1 x 1 |       1      |              1 |              0 | False        |
        |   2 | 1 x 2 |       0.5    |              2 |              0 | True         |
        |   3 | 2 x 2 |       1      |              4 |              1 | True         |
        |   4 | 2 x 2 |       1      |              4 |              0 | False        |
        |   5 | 2 x 3 |       0.6667 |              6 |              1 | True         |
        |   6 | 2 x 3 |       0.6667 |              6 |              0 | False        |
        |   7 | 4 x 2 |       2      |              8 |              1 | True         |
        |   8 | 4 x 2 |       2      |              8 |              0 | False        |
        |   9 | 3 x 3 |       1      |              9 |              0 | False        | 


        To see more examples, run `generate_sample_table(number_of_examples)`

        """
        best = None
        best_score = None

        for ncols in range(1, n + 1):
            if max_cols and ncols > max_cols:
                break

            nrows = math.ceil(n / ncols)
            total = nrows * ncols
            empty = total - n
            aspect = ncols / nrows
            aspect_penalty = abs(math.log(aspect))

            empty_weight = empty + (aspect_penalty > 1.0)*3

            score = (
                empty_weight,                  
                aspect_penalty,                
                abs(ncols - preferred_ncols)*2,  
                total                          
            )

            if best_score is None or score < best_score:
                best = (nrows, ncols)
                best_score = score

        return best


#### PLACEHOLDER -- FUTURE PLAN: 

    # class tickmarkOptimizer():


    # if possible, might need to look into ways to query window geometry to figure out the optimal spaceing between tick marks automatically. 
    

    def sample_output_table(self,nexamples=20, markdown_print=False):

        """
        Generates a table (DataFrame) summarizing the optimal grid dimensions calculated using the `optimal_grid` method for the first nexamples.

        Parameters
        ----------
        nexamples : int, optional
            The number of examples to generate, default is 20.

        markdown_print : bool, optional
            If True, prints the resulting DataFrame in markdown table format, default is False.

        Returns
        -------
        DataFrame
            A pandas DataFrame containing columns:

            - 'N': Total number of sets of data points to plot - i.e., the minimum number of subplots required to plot all data points.

            - 'nrows x ncols': Optimal grid dimensions as a string.
            
            - 'aspect_ratio': The aspect ratio of the grid (i.e., nrows / ncols).
            
            - 'total_subplots': The total number of subplots in the grid.
            
            - 'empty_subplots': The number of empty subplots - Example: if N = 11, a grid of 4 rows and 3 columns will have 2 empty subplots.
            
            - 'N_is_prime': Boolean indicating if the number of subplots is prime -- prime numbers tend to result in more empty subplots, while composite numbers tend to more neatly be arranged into a rectangular grid.
        """

        totals = []
        optrows = []
        optcols = []
        spcounts = []

        for i in range(1,nexamples+1):
            # print(f"{i}: {optimal_grid(i)}, {optimal_grid(i)[0] * optimal_grid(i)[1]}")
            totals.append(i)
            nrows,ncols = self.optimal_grid(i)
            optrows.append(nrows)
            optcols.append(ncols)
            spcounts.append(self.optimal_grid(i)[0] * self.optimal_grid(i)[1])

        optdimtbl =pd.DataFrame({'N':totals,'nrows':optrows,'ncols':optcols,'total_subplots':spcounts})

        optdimtbl['nrows x ncols'] = optdimtbl.nrows.astype(str) + ' x ' + optdimtbl.ncols.astype(str)

        optdimtbl['aspect_ratio'] = (optdimtbl.nrows / optdimtbl.ncols).round(4)

        optdimtbl = optdimtbl[['N','nrows x ncols','aspect_ratio','total_subplots']]

        optdimtbl['empty_subplots'] = optdimtbl.total_subplots - optdimtbl.N

        optdimtbl['N_is_prime'] = optdimtbl.N.apply(lambda x: sympy.isprime(x))

        if markdown_print:
            print(optdimtbl.to_markdown(index=False,tablefmt='pipe'))
        
        return optdimtbl


def pathToArray(path):
    """
    Reads a file at the given path into a numpy array.

    Parameters
    ----------
    path : str
        Path to the file to be read.

    Returns
    -------
    arr : numpy array
        The contents of the file, loaded into a numpy array.

    Raises
    ------
    ValueError
        If the file does not exist, or is not a file, or if the file is not a supported image format.

    Supported formats include:
        - numpy (.npy)
        - JPEG (.jpg, .jpeg)
        - PNG (.png)
        - TIFF (.tif, .tiff)
        - WebP (.webp)
        - AVIF (.avif)
        - GIF (.gif)
        - BMP (.bmp)

    Notes
    -----
    Uses the `magic` library to determine the file type based on the file contents, and then uses either `numpy.load` for numpy files or `cv2.imread` for image files to load the contents of the file into a numpy array.
    """
    if not os.path.isfile(path):
        raise ValueError(f"Path {path} does not exist or is not a file.")
    fileinfo = [magic.from_file(path).lower(), magic.from_file(path, mime=True).lower()]

    # Check if the file is a supported image
    accepted_formats = ['numpy', 
                        'jpeg', 'jpg', 
                        'png', 
                        'tiff', 'tif', 
                        'webp', 'web/p',
                        'avif', 
                        'gif', 
                        'bmp','bitmap']
        
    if not any(ext in info for ext in accepted_formats for info in fileinfo):
        raise ValueError(f"File \033[7m{path}\033[0m is not a supported image format.\n\nAccepted formats: {', '.join([f"\033[7m{ext}\033[0m" for ext in accepted_formats])}")
    
    if fileinfo[0].lower().startswith('numpy'):
        return np.load(path)
    else:
        return cv2.imread(path)
    
def arrayDimParser(array):
    if array.ndim == 2:
        return "2D grayscale"
    elif array.ndim == 3:
        if array.shape[2] in (3,4):
            return "2D color"
        else:
            return "3D grayscale"
    elif array.ndim == 4:
        if array.shape[3] in (3,4):
            return "3D color"
        else:
            return "4D"
    else:
        return f"{array.ndim}D"
    
def pathByIndex(index,df,path_col='filepath'):
    return df.iloc[index][path_col]
    
def vizInputParser(vizInput):
    # Case 1: already an array
    """
    Ensures a variety of visual data inputs (vizInput) -- either an array or a path to a file containing one of two types of visual data (i.e., either a numpy file [.npy] or an image file [.jpg, .jpeg, .png, .tif, .tiff, .webp, .avif, .gif, .bmp]) -- are loaded as a numpy array.

    Parameters
    ----------
    vizInput : array_like or path_like
        The input to be converted.  This can be either an array (numpy
        ndarray, list, or tuple), or a path-like (str, bytes, Path).

    Returns
    -------
    array : numpy ndarray
        The input converted to a numpy array.

    Raises
    ------
    ValueError
        If the input is not a supported type.
    FileNotFoundError
        If the input is a path-like but the file does not exist.
    IsADirectoryError
        If the input is a path-like but the path is a directory.
    PermissionError
        If the input is a path-like but the file is not readable.
    TypeError
        If the input is not a supported type.
    """
    if isinstance(vizInput, np.ndarray):
        return vizInput

    # Case 2: path-like (str, bytes, Path)
    if isinstance(vizInput, (str, bytes, os.PathLike)):
        p = Path(vizInput)
        try:
            if p.is_dir():
                raise IsADirectoryError(f"The provided path is a directory, not a file: \033[7m{p}\033[0m")
            # Let pathToArray raise if the file is missing/locked/corrupt, etc.
            return pathToArray(p)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The provided path for vizInput does not exist or is not a file:\n\n\t\033[7m{p}\033[0m"
            )
        except IsADirectoryError:
            raise
        except PermissionError:
            raise
        except Exception as e:
            # Normalize everything else to a ValueError that preserves the root cause.
            raise ValueError(f"Failed to load array from {p}: {e}") from e

    # Case 3: unsupported type
    raise TypeError("vizInput must be a numpy ndarray or a path-like (str/Path).")
        


def crossSection(vizInput, axis_norm=0,slice_depth=0.5):
    """Extracts cross-section/slice from 3D arrays or otherwise returns 2D array formatted for matplotlib.pytplot.imshow (RGB format) if 2D image provided.
    
    PARAMETERS
    ----------
    vizInput : array_like or path_like
        This can either by a numpy array already loaded, a path to a numpy file, or a path to an image file.  Regardless of type, it will be loaded as a numpy array from which the cross-section/slice will be extracted.
    axis_norm : int, optional
        Axis along which to extract cross-section/slice, default is 0; i.e., this axis is orthogonal/normal to the cross-section/slice being extracted.
    slice_depth : float, optional
        Fraction of 1 corresponding to depth of slice to extract, default is 0.5 (the middle slice along axis_norm).  This allows for specification of depth without knowing the exact array dimensions ahead of time.

    RETURNS
    -------
    slice : array
        Cross-section/slice extracted from array.
    """

    # load file as array
    visArr = vizInputParser(vizInput)
    dimClass = arrayDimParser(visArr)

    if dimClass == "3D grayscale":
        vol = visArr

        # get cross section based on axis_norm & slice_depth
        if axis_norm == 1:
            slice_index = vol.shape[1] * slice_depth
            slice_index = int(slice_index)
            slice = vol[:,slice_index,:]
            other_axes = (0,2)
        elif axis_norm == 0:
            slice_index = vol.shape[0] * slice_depth
            slice_index = int(slice_index)
            slice = vol[slice_index,:,:]
            other_axes = (1,2)
        elif axis_norm == 2:
            slice_index = vol.shape[2] * slice_depth
            slice_index = int(slice_index)
            slice = vol[:,:,slice_index]
            other_axes=(0,1)
        
        y,x = other_axes
        h = vol.shape[y]
        w = vol.shape[x]
        d = vol.shape[axis_norm]
       
    elif dimClass.startswith('2D'):
        y,x = (0,1)
        h = visArr.shape[y]
        w = visArr.shape[x]
        d = 1
        slice_index = 0
        if dimClass.endswith("grayscale"):
            slice = visArr
        elif dimClass.endswith("color"):
            slice = cv2.cvtColor(visArr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Array dimension {dimClass} not supported.")
    else:
        raise ValueError(f"Array dimension {dimClass} not supported.")
    
    return (slice, {'slice_index':slice_index,
                    'slice_depth':slice_depth,
                    'axis_norm':axis_norm,
                    'h':h,
                    'w':w,
                    'd':d}
            )


def tickIntervalCalc(max_value,denom,spacing_tier=2):
    """
    Calculates a number b * 10^c that most closely approximates max_value / 10,
    with c being an integer and b being either 1 or 5.  Essentially, this results in tickmark intervals that end in 1, 5, or 0 (e.g., 0.5, 0.1, 1, 10, 50, 100).
    """
    if max_value == 0:
        return "0"

    target_value = max_value / denom

    sign = 1 if target_value >= 0 else -1
    abs_target_value = abs(target_value)

    if abs_target_value == 0:
        return "0"

    c1 = math.floor(math.log10(abs_target_value))
    c2 = c1 + 1

    candidates = [
        (1, c1),
        (5, c1),
        (1, c2),
        (5, c2)
    ]

    min_diff = float('inf')
    best_b = 0
    best_c = 0

    for b, c in candidates:
        approximation = sign * b * (10 ** c)
        diff = abs(target_value - approximation)
        if diff < min_diff:
            min_diff = diff
            best_b = b
            best_c = c

    int(sign*best_b*10**best_c)
    
    result = sign*best_b*10**best_c

    if spacing_tier == 0:
        return result / 2
    elif spacing_tier == 1:
        return result
    elif spacing_tier == 2:
        return result * 2


def plotCrossSection(filepath, axis_norm=1,slice_depth=0.5, figsize=(6,5), ax=None):
    
    dimClass = arrayDimParser(pathToArray(filepath))

    # Turn off LaTeX
    if plt.rcParams['text.usetex'] == True:
        plt.rcParams['text.usetex']=False

    # Get cross section and slice info
    slice_arr,slice_info = crossSection(filepath, axis_norm, slice_depth)
    slice_index = slice_info['slice_index']
    slice_depth = slice_info['slice_depth']
    axis_norm = slice_info['axis_norm']
    h = slice_info['h']
    w = slice_info['w']
    d = slice_info['d']

    # Plot cross section
    filename = os.path.basename(filepath)
    title = f"{os.path.splitext(filename)[0]}\n"

    if not ax:
        _,ax = plt.subplots(1,1,figsize=figsize)
        plt.title(title, fontweight='bold')
    else:
        ax.set_title(title, fontweight='bold')

    ax.imshow(slice_arr)

    # Formatting & Annotations
    
    AR = w/h
    if AR > 1:
        xdenom = 10 * int(AR)
        ydenom = 10 
    elif AR < 1:
        xdenom = 10 
        ydenom = 10 * int(1/AR)
    else:
        xdenom = ydenom = 10
    
    xtint = tickIntervalCalc(w,xdenom)
    ytint = tickIntervalCalc(h,ydenom)
    
    ax.set_xticks(range(0,w,xtint))
    ax.set_yticks(range(0,h,ytint))

    mx = 1.07
    
    if dimClass.startswith('3D'):
        ax.text(mx,.90,f'Slice Index', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold')
    
        ax.text(mx,.85,f'{slice_index} of {d}', 
                transform=ax.transAxes,
                fontsize=12)
        
        ax.text(mx,.50,f'Normal Axis', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold')
    
        ax.text(mx,.45,f'ndarray axis = {axis_norm}', 
                transform=ax.transAxes,
                fontsize=12)
        
        ax.text(mx,.30,f'Slice Depth', 
                transform=ax.transAxes, 
                fontsize=10, fontweight='bold')
    
        ax.text(mx,.25,f'{slice_depth}', 
                transform=ax.transAxes,
                fontsize=12)
        
    ax.text(mx,.70,f'Image\nDimensions', 
            transform=ax.transAxes, 
            fontsize=10, fontweight='bold')
    
    ax.text(mx,.65,f'{w} x {h} px', 
            transform=ax.transAxes,
            fontsize=12)
    
    return ax

    
def viewCrossSection(filepath, axis_norm=1,slice_depth=0.5, figsize=(6,5), ax=None):
    pltax = plotCrossSection(filepath,axis_norm,slice_depth,figsize,ax)
    plt.show()



# def orthoPlanes(filepath,ortho_depth=(.5,.5,.5)):

#     visArr = pathToArray(filepath)
#     dimClass = arrayDimParser(visArr)
#     if dimClass != "3D grayscale":
#         raise ValueError("Array must be 3D grayscale.")

#     vol = visArr
#     voldims = vol.shape

#     # Simply copy the float across three axes
#     if isinstance(ortho_depth,float):
#         ortho_depth = (ortho_depth,ortho_depth,ortho_depth)
#     # Otherwise, make sure it's an iterable with 1-3 items that's not a string
    
#     elif not isinstance(ortho_depth,str)\
#     and len(ortho_depth) in range(1,4):
#         try:
#             iter(ortho_depth)
#             # Then replace the defaults from left to right / leave behind what's not specified.
#             rem = 3 - len(ortho_depth)
#             if rem > 0:
#                 odls = list(ortho_depth)
#                 for i in range(rem):
#                     odls.append(0.5)
#                 ortho_depth = tuple(odls)
#             else:
#                 pass           
            
#         except:
#             raise TypeError('ortho_depth must either be [1] a float or [2] a non-string iterable with 1-3 items')

        
#     orthoslices = []
#     for axis in range(3):
#         slice, _ = crossSection(filepath, slice_depth=ortho_depth[axis], axis_norm=axis)
#         orthoslices.append(slice)

    
#     fH = np.array([ m.shape[0] for m in orthoslices]).max().astype(int)
#     fW = np.array([ m.shape[1] for m in orthoslices]).mean().astype(int)
    
#     fH,fW = round(fH/fW,1).as_integer_ratio()
#     fH,fW = fH*3,fW*3
    
#     fig,ax = plt.subplots(1,3,figsize=(fH,fW))
    
#     filename = os.path.basename(filepath)
#     fig.suptitle(f'{os.path.splitext(filename)[0]}\n', fontweight='bold')
    
#     for i,slice in enumerate(orthoslices):
#         vertax,horizax = [ a for a in range(3) if a != i ]
#         h,w = orthoslices[i].shape
#         hh,ww = voldims[vertax],voldims[horizax]
#         d = voldims[i]
#         # print(hh,ww,d,'\n')
#         slice_index = int(ortho_depth[i]*d)
#         ax[i].imshow(slice)
#         ax[i].set_title(f'Axis {i}\n')
#         ax[i].text(.5,1.02,f'Slice #{slice_index} of {d}', transform=ax[i].transAxes, 
#                    fontsize=9, fontweight=None,
#                   horizontalalignment='center')


#         ax[i].set_xticks(range(0,w,15))
#         ax[i].tick_params(axis='x',labelsize=7)
#         ax[i].set_yticks(range(0,h,15))
#         ax[i].tick_params(axis='y',labelsize=7)
        
#         if i == 0:
#             d1 = ortho_depth[1]*voldims[1]
#             ax[i].axhline(d1,c='white')
#         if i == 2:
#             d1 = ortho_depth[1]*voldims[1]
#             ax[i].axvline(d1,c='white')
#         if i == 1:
#             d02 = 0.5 * voldims[0]
#             ax[i].axvline(d02,c='yellow')
#             ax[i].axhline(d02,c='yellow')

#     plt.tight_layout()
#     plt.show()

  


def orthoPlanes(
    filepath,
    ortho_depth=(0.5, 0.5, 0.5),
    axlines=False,
    anatomy_axes=False,
    cmap=('red', 'magenta', 'yellow'),
    fig_facecolor='gray',
    figsize=None
):
    """
    Plot three orthogonal slices from a 3D grayscale volume.

    Parameters
    ----------
    filepath : str | os.PathLike
        Path to a file loadable by `pathToArray`, which returns a numpy array.
    ortho_depth : float | iterable of 1–3 floats in [0,1]
        Depth(s) along (Z, Y, X) as fractions of dimension length.
        - single float -> replicated to (d,d,d)
        - 1–3 length iterable -> fills missing with 0.5 on the right.
    axlines : bool
        If True, draw dashed crosshair/registration lines indicating the other
        planes' positions in each view.
    anatomy_axes : bool
        If True, draw simple anterior arrows/labels for orientation (demo).
    cmap : tuple[str, str, str]
        Colors for per-axis titles/lines. Defaults ('red','magenta','yellow').
    fig_facecolor : str
        Figure background color. Default 'gray'.

    Returns
    -------
    (fig, ax) : (matplotlib.figure.Figure, np.ndarray[Axes])
    """

    # --- Load & validate array
    visArr = pathToArray(filepath)
    dimClass = arrayDimParser(visArr)
    if dimClass != "3D grayscale":
        raise ValueError(f"Array must be 3D grayscale, got: {dimClass}")

    vol = visArr
    voldims = vol.shape  # (Z, Y, X) assumed by crossSection(axis_norm)

    # --- Normalize ortho_depth to a 3-tuple
    if isinstance(ortho_depth, (int, float)):
        ortho_depth = (float(ortho_depth),) * 3
    else:
        # non-string iterable with length 1–3
        if isinstance(ortho_depth, (str, bytes)):
            raise TypeError("ortho_depth must not be a string.")
        try:
            n = len(ortho_depth)
        except Exception:
            raise TypeError("ortho_depth must be a float or an iterable of length 1–3.")
        if n not in (1, 2, 3):
            raise ValueError("ortho_depth iterable must have length 1–3.")
        od = list(map(float, ortho_depth))
        od += [0.5] * (3 - n)
        ortho_depth = tuple(od)

    # (optional) keep depths inside [0,1] but don’t hard-fail if slightly out
    ortho_depth = tuple(max(0.0, min(1.0, d)) for d in ortho_depth)

    # --- Extract orthogonal slices via crossSection 
    # crossSection is assumed to return (slice_array, meta_dict_or_none
    orthoslices = []
    for axis in range(3):
        slc, _ = crossSection(filepath, slice_depth=ortho_depth[axis], axis_norm=axis)
        orthoslices.append(slc)

    if not figsize:
        # --- Figure size heuristics (preserve your ratio)
        fH = int(max(m.shape[0] for m in orthoslices))
        fW = int(np.mean([m.shape[1] for m in orthoslices]))
        # avoid division by zero
        fW = max(fW, 1)

        ratio = round(fH / fW, 1)
        num, den = (1, 1)
        try:
            num, den = ratio.as_integer_ratio()
        except Exception:
            pass
        # scale up a bit so it’s readable
        figsize = (den * 3, num * 3)

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    fig.set_facecolor(fig_facecolor)

    # Title from filename (no extension)
    filename = os.path.basename(str(filepath))
    fig.suptitle(os.path.splitext(filename)[0] + "\n", fontweight='bold')

    # --- Draw each view
    for i, slc in enumerate(orthoslices):
        h, w = slc.shape
        dZ, dY, dX = voldims
        dims = (dZ, dY, dX)
        d = dims[i]
        slice_index = int(ortho_depth[i] * d)

        ax[i].imshow(slc)
        ax[i].set_title(f'Normal Axis: {i}\n', color=cmap[i], fontweight='bold')

        ax[i].text(
            0.5, 1.02, f'Slice #{slice_index} of {d}',
            transform=ax[i].transAxes,
            fontsize=9,
            ha='center'
        )

        # modest ticks for readability
        ax[i].set_xticks(range(0, w, max(1, w // 15)))
        ax[i].tick_params(axis='x', labelsize=7)
        ax[i].set_yticks(range(0, h, max(1, h // 15)))
        ax[i].tick_params(axis='y', labelsize=7)

        # Optional crosshair-style reference lines
        if axlines:
            alpha = 0.75
            ls = '--'
            lw = 1.75

            if i == 0:  # Z-view (show Y and X positions)
                y_pos = ortho_depth[1] * dY
                x_pos = ortho_depth[2] * dX
                ax[i].axhline(y_pos, c=cmap[1], linewidth=lw, alpha=alpha, linestyle=ls)
                ax[i].axvline(x_pos, c=cmap[2], linewidth=lw, alpha=alpha, linestyle=ls)
                # ax[i].annotate(
                #     '1', xy=(x_pos, y_pos), xytext=(x_pos+x_pos/1.5, y_pos-y_pos/15),
                #     transform=ax[i].transData,
                #     color=cmap[1], fontsize=10, fontweight='bold', ha='center', va='center'
                # )
                # ax[i].annotate(
                #     '2', xy=(x_pos, y_pos), xytext=(x_pos+x_pos/10, y_pos-y_pos/1.5),
                #     transform=ax[i].transData,
                #     color=cmap[2], fontsize=10, fontweight='bold', ha='center', va='center'
                # )

                if anatomy_axes:
                    # simple anterior arrow (axes coords: 0..1)
                    ax[i].annotate(
                        '', xy=(0.5, 0.30), xytext=(0.5, 0.05),
                        transform=ax[i].transAxes,
                        arrowprops=dict(arrowstyle='->', color='white', mutation_scale=20, linewidth=2.5)
                    )
                    ax[i].text(0.48, 0.33, 'anterior', transform=ax[i].transAxes,
                               ha='right', va='center', color='white', fontweight='bold', rotation=90)

            if i == 2:  # X-view (show Y and Z positions)
                x_pos = ortho_depth[1] * dY
                y_pos = ortho_depth[0] * dZ
                ax[i].axvline(x_pos, c=cmap[1], linewidth=lw, alpha=alpha, linestyle=ls)
                ax[i].axhline(y_pos, c=cmap[0], linewidth=lw, alpha=alpha, linestyle=ls)
                # ax[i].annotate(
                #     '0', xy=(x_pos, y_pos), xytext=(x_pos+.75, y_pos-.1),
                #     transform=ax[i].transData,
                #     color=cmap[0], fontsize=10, fontweight='bold', ha='center', va='center'
                # )
                # ax[i].annotate(
                #     '1', xy=(x_pos, y_pos), xytext=(x_pos+.1, y_pos-y_pos/1.5),
                #     transform=ax[i].transData,
                #     color=cmap[1], fontsize=10, fontweight='bold', ha='center', va='center'
                # )

                if anatomy_axes:
                    ax[i].annotate(
                        '', xy=(0.30, 0.5), xytext=(0.05, 0.5),
                        transform=ax[i].transAxes,
                        arrowprops=dict(arrowstyle='->', color='white', mutation_scale=14, linewidth=2)
                    )
                    ax[i].text(0.35, 0.44, 'anterior', transform=ax[i].transAxes,
                               ha='left', va='center', color='white', fontweight='bold')

            if i == 1:  # Y-view (show X and Z positions)
                x_pos = ortho_depth[2] * dX
                y_pos = ortho_depth[0] * dZ
                ax[i].axvline(x_pos, c=cmap[2], linewidth=lw, alpha=alpha, linestyle=ls)
                ax[i].axhline(y_pos, c=cmap[0], linewidth=lw, alpha=alpha, linestyle=ls)
                # ax[i].annotate(
                #     '0', xy=(x_pos, y_pos), xytext=(x_pos+x_pos/1.5, y_pos-y_pos/10),
                #     transform=ax[i].transData,
                #     color=cmap[0], fontsize=10, fontweight='bold', ha='center', va='center'
                # )
                # ax[i].annotate(
                #     '2', xy=(x_pos, y_pos), xytext=(x_pos+x_pos/10, y_pos-y_pos/1.5),
                #     transform=ax[i].transData,
                #     color=cmap[2], fontsize=10, fontweight='bold', ha='center', va='center'
                # )


    plt.tight_layout()
    return fig, ax

def viewImagesByPatient(df, patient_id, figdims=None, figsize=(12,12), patient_id_colname='patient_id', filepath_colname='filepath'):
    ptdf = df[df[patient_id_colname] == patient_id]
    total = len(ptdf)

    if figdims:
        nrows,ncols = figdims
    else:
        gdo = GridDimensionOptimizer()
        nrows, ncols = gdo.optimal_grid(total)

    _,ax = plt.subplots(nrows,ncols,figsize=figsize)
    
    axf = ax.flat

    

    for i in range(total):
        ptrecord = ptdf.iloc[i]
        imgArr,_ = crossSection(ptrecord[filepath_colname])
        axf[i].imshow(imgArr)
        axf[i].set_title(f"{ptrecord.image_type} | {ptrecord.dx_class} | {ptrecord.laterality}\n{os.path.basename(ptrecord.filepath)}",fontsize=10)
    
    plt.suptitle(f"Patient {patient_id}",fontsize=12)
    plt.tight_layout()


