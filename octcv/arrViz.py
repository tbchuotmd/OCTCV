import os
from pathlib import Path
import math, sympy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import magic
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_sample_volume(
    index:int=None, name:str=None, 
    volume_dirpath:str=None
):
    if volume_dirpath is None:
        arrViz_dirpath = os.path.dirname(__file__)
        # if os.path.basename(arrViz_dirpath) != 'octcv':
            # print(arrViz_dirpath,'\n')
        volume_dirpath = os.path.join(arrViz_dirpath,'../datasrc/volumesOCT')
        
    if index is not None:
        volume_filenames = os.listdir(volume_dirpath)
        volume_filepath = os.path.join(volume_dirpath,volume_filenames[index])
    elif name is not None:
        volume_filepath = os.path.join(volume_dirpath,name)
    else:
        # print(f"getting list of files within {os.path.abspath(volume_dirpath)}")
        volume_filenames = os.listdir(volume_dirpath)
        # print(f"volume_filenames is a {type(volume_filenames)} of len {len(volume_filenames)}")
        index = np.random.randint(0,len(volume_filenames))
        # print(f"will apply random index of {index}, which is a {type(index)}")
        filename = volume_filenames[index]
        # print(f"the file at index {index} is {filename}")
        volume_filepath = os.path.join(volume_dirpath,volume_filenames[index])
        # print(f"the final volume filepath is {volume_filepath}")
        
    if not os.path.isfile(volume_filepath):
        raise ValueError(f"The path \033[45m{volume_filepath}\033[0m does not point to an existing file.")

    return np.load(volume_filepath)
    
class volumeViewer:
    def __init__(self, 
                 volume_data:np.ndarray = np.random.randint(0,255,(9,9,9)) 
                ):
        
        if volume_data.ndim not in [3,4]:
            raise ValueError(f"Provided volume_data has ndim = {volume_data.ndim}.  Data must have ndim of either 3 (grayscale) or 4 (color).")
            
        self.data = volume_data

    def sliceViewer(self, data=None, axis=0,plotscale=4):
        data = self.data if data is None else data
        
        # 1. Create the base figure
        fig = go.Figure()
    
        # 2. Parse Logic for Axis
        def zdata(slice_index,data=data,axis=axis):
            slices = [slice(None)] * data.ndim
            slices[axis] = slice_index
            return data[tuple(slices)][::-1,::-1]
        
        # 3. Add each slice as a 'trace' (initially all hidden except the first)
        for i in range(data.shape[axis]):
            fig.add_trace(
                go.Heatmap(
                    visible=False,
                    z=zdata(i),
                    colorscale='Viridis',
                    showscale=False
                )
            )

        middle_slice_index = data.shape[axis]//2
        
        # Make the first trace visible
        fig.data[middle_slice_index].visible = True
        
        # 4. Create the slider steps
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": f"Slice: {i}"}], 
                label=str(i)
            )
            step["args"][0]["visible"][i] = True  # Make only the current selection visible
            steps.append(step)
        
        # 5. Add slider to layout (Vertical on the right)
        sliders = [dict(
            active=middle_slice_index,
            currentvalue={
                "prefix": "Slice #: ", 
                "visible": True, 
                "xanchor": "center", # Centers the text relative to the slider bar
                "offset": 10         # Space between text and slider
            },
            pad={"l": 20, "r": 20, "t": 10, "b": 10},
            x=1.05,               # Positioned just to the right of the plot
            y=0,                  # Anchored at the bottom
            len=1,                # Span the full height (0 to 1)
            xanchor="left",
            yanchor="bottom",
            steps=steps
        )]
        
        fig.update_layout(
            sliders=sliders,
            margin=dict(r=120, l=50, t=50, b=50), # Increased right margin for the slider
            yaxis=dict(
                scaleanchor="x", 
                scaleratio=1,
                constrain='domain'
            ),
            xaxis=dict(constrain='domain')
        )

        fig.show()

    def surfacePlot(self,data=None):
        data = self.data if data is None else data

        h,w,d = data.shape

        # Need way to map hwd to whatever j in mgrid...
        X, Y, Z = np.mgrid[0:255:64j,0:255:128j,0:255:64j,]
        values = data
        
        # 2. Define binary threshold
        threshold = 100
        
        # 3. Create Isosurface
        fig = go.Figure(data=go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=threshold,
            isomax=threshold,
            surface_count=3,
            opacity=0.3,
            colorscale='Viridis'
        ))
        
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        fig.show()

    def orthoSlicesViewer(self,data=None):
        
        if data is None:
            data = self.data
            
        d0, d1, d2 = data.shape
        
        # 1. Aspect Ratio Handling (Scale factor f=4 for visibility)
        f = 4
        # Subplot widths are proportional to the 'horizontal' dimension of the slice
        column_widths = [d2, d2, d1] 
        
        fig = make_subplots(
            rows=1, cols=3, 
            subplot_titles=("Sagittal (Ax 0)", "Coronal (Ax 1)", "Axial (Ax 2)"),
            column_widths=column_widths,
            horizontal_spacing=0.05
        )
    
        # Track trace start indices
        offsets = [0, d0, d0 + d1]
    
        # 2. Add Heatmap Traces
        for i in range(d0):
            fig.add_trace(go.Heatmap(z=data[i,:,:], visible=(i==0), colorscale='Viridis', showscale=False), row=1, col=1)
        for i in range(d1):
            fig.add_trace(go.Heatmap(z=data[:,i,:], visible=(i==0), colorscale='Viridis', showscale=False), row=1, col=2)
        for i in range(d2):
            fig.add_trace(go.Heatmap(z=data[:,:,i], visible=(i==0), colorscale='Viridis', showscale=False), row=1, col=3)
    
        # 3. Initialize Crosshair Shapes
        # Use a dictionary for the line sub-properties
        line_props = dict(color="red", width=5, dash="dot")
        
        # Placeholder lines at index 0
        # Shapes are added in order: 0,1 (Plot 1) | 2,3 (Plot 2) | 4,5 (Plot 3)
        for col, x_max, y_max in [(1, d2, d1), (2, d2, d0), (3, d1, d0)]:
            # Horizontal Line
            line_props['color']='yellow'
            fig.add_shape(type="line", x0=0, x1=x_max, y0=0, y1=0, 
                          xref=f"x{col}", yref=f"y{col}", line=line_props) 
            # Vertical Line
            line_props['color']='magenta'
            fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=y_max, 
                          xref=f"x{col}", yref=f"y{col}", line=line_props)
    
        def create_steps(ax_idx):
            steps = []
            n_slices = data.shape[ax_idx]
            total_traces = d0 + d1 + d2
            
            for i in range(n_slices):
                # (2) Don't reset other subplots: Use 'None' for indices not in this axis
                visibility = [None] * total_traces
                for j in range(n_slices):
                    visibility[offsets[ax_idx] + j] = (i == j)
                
                # (3) Crosshair Logic
                # Shapes are indexed 0-5. 
                # If we move Axis 0: Update H-line of Plot 2 and H-line of Plot 3
                shape_updates = {}
                if ax_idx == 0:
                    shape_updates.update({"shapes[2].y0": i, "shapes[2].y1": i, "shapes[4].y0": i, "shapes[4].y1": i})
                elif ax_idx == 1:
                    shape_updates.update({"shapes[0].y0": i, "shapes[0].y1": i, "shapes[5].x0": i, "shapes[5].x1": i})
                elif ax_idx == 2:
                    shape_updates.update({"shapes[1].x0": i, "shapes[1].x1": i, "shapes[3].x0": i, "shapes[3].x1": i})
    
                steps.append(dict(
                    method="update",
                    args=[{"visible": visibility}, shape_updates],
                    label=str(i)
                ))
            return steps
    
        # 4. Sliders and Layout
        fig.update_layout(
            sliders=[
                dict(active=0, steps=create_steps(0), len=0.3, x=0, currentvalue={"prefix": "Ax0: "}),
                dict(active=0, steps=create_steps(1), len=0.3, x=0.35, currentvalue={"prefix": "Ax1: "}),
                dict(active=0, steps=create_steps(2), len=0.3, x=0.7, currentvalue={"prefix": "Ax2: "})
            ],
            height=max(d1, d0) * f + 150, # Dynamic height based on data
            width=sum(column_widths) * f,
            margin=dict(l=20, r=20, t=50, b=100)
        )
        
        # Reverse Y-axes for image convention
        fig.update_yaxes(autorange="reversed")
        fig.show()


#---------------------------------------------------------------------------------------------------------


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

        # In the case that a single row of plots is desired
        if preferred_ncols == n:
            best = (1,n)
            return best
            
        # Otherwise, will prioritize more square-looking figures (i.e., even if n=3 and preferred_cols=3, will recomend (2,2)).  Note: if n=3,preferred_cols=1 --> the following will recommend a single column of plots (3,1), hence only situation where preferred_cols == n must be explicitly written in the if statement above.
        
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
        
def getHyperplane(numpy_array, axis=0, slice_index=None, slice_depth=0.5):
    """
    Takes an N-dimensional NumPy array as input and returns a "sub-array" with N-1 dimensions found at a specified depth along a given axis of the original array.

    ----------------------------------------------------

        For grayscale 3D volume arrays, this means taking a 2D slice along the specified axis (e.g., x,y,z)

        For grayscale 2D image arrays, this means taking a linear array (row or column).

        For color volumes/images, the same applies, except in the case of specifying the last axis (axis=2 for color 2D images, axis=3 for color 3D volumes), in which case the result is a single color channel of the whole array (e.g., for RGB 2D Image -  a 3D array of shape (h,w,3), passing axis=2 and slice_index=2 would return a 2D array of shape (h,w) with values corresponding to the "Blue" channel.)
        
    ----------------------------------------------------
    
    """
    
    slices = [slice(None)] * numpy_array.ndim
    if slice_index is None:
        slice_index = int(numpy_array.shape[axis] * slice_depth)
    slices[axis] = slice_index
    return numpy_array[tuple(slices)]
    

def plotSlice(volume, axis=0, slice_index=None, slice_depth=0.5,figsize=(5,5),ax=None,hide_plot_axes=True):
    """
    Plots image of a slice along a given axis of a volume.
    
    PARAMS
    ------
    **for getHyperplane() function**:
    
        volume: a 3D Volume in the form of either a path to a .npy file (str) or NumPy array (np.ndarray) - can either be grayscale (h,w,d) or color (h,w,d,c) [ where c=3 in the case of RGB/BGR, or c=4 in the case of RGBA ]
    
        axis: the NumPy axis orthogonal to the slice to be extracted from the volume.  Default is 0 (the first axis).
    
        slice_index: specific index of the slice along the axis - if not specified, slice_index will be automatically calculated from slice_depth.  Default is None.
    
        slice_depth: value within [0,1] specifying the position along the axis to extract the slice; if slice_index specified, this parameter is ignored.  Otherwise, the slice_index is calculated from slice_depth (e.g., slice_depth=0.25 means get the slice_index that is ~1/4 the way from the first slice to the last slice along the specified axis).  Default is 0.5, meaning the middle slice along the specified axis.

    +++

    **for matplotlib.pyplot subplots**:
        
        figsize (Default = (5,5))
        ax (Default = None)
        hide_plot_axes (Default = True)

    """
    numpy_3D_array = vizInputParser(volume)
    shape = numpy_3D_array.shape
    
    if len(shape) == 3 or (len(shape) == 4 and shape[-1] in [3,4]):
        volume_slice = getHyperplane(numpy_array=numpy_3D_array,
                      axis=axis,
                      slice_index=slice_index,
                      slice_depth=slice_depth)
    else:
        raise ValueError(f"numpy_3D_array must be a 3D volume - i.e., must be either a 3D array in the case of pure grayscale, or 4D array in which the length of the last/4th axis (axis=3) is either 3 (RGB/BGR) or 4 (RGBA).  Provided array has shape: {numpy_3D_array.shape}.")

    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)

    if hide_plot_axes:
        ax.set_axis_off()
    else:
        ax.set_axis_on()
    ax.imshow(volume_slice)



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


def plotCrossSection(filepath, axis_norm=1,slice_depth=0.5, figsize=(6,5), title=None, ax=None, show_image_stats=True):
    
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
    if title is None:
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
    
    if type(show_image_stats) == bool:
        if not show_image_stats:
            return ax
        else:
            show_image_stats = ['slice_index','axis_norm','slice_depth','image_dims']

    if type(show_image_stats) == list:
        if dimClass.startswith('3D'):
            if 'slice_index' in show_image_stats:
                ax.text(mx,.90,f'Slice Index', 
                        transform=ax.transAxes, 
                        fontsize=10, fontweight='bold')
            
                ax.text(mx,.85,f'{slice_index} of {d}', 
                        transform=ax.transAxes,
                        fontsize=12)
                
            if 'axis_norm' in show_image_stats:
                ax.text(mx,.50,f'Normal Axis', 
                        transform=ax.transAxes, 
                        fontsize=10, fontweight='bold')
            
                ax.text(mx,.45,f'ndarray axis = {axis_norm}', 
                        transform=ax.transAxes,
                        fontsize=12)
                
            if 'slice_depth' in show_image_stats:
                ax.text(mx,.30,f'Slice Depth', 
                        transform=ax.transAxes, 
                        fontsize=10, fontweight='bold')
            
                ax.text(mx,.25,f'{slice_depth}', 
                        transform=ax.transAxes,
                        fontsize=12)
            
        if 'image_dims' in show_image_stats:
            ax.text(mx,.70,f'Image\nDimensions', 
                    transform=ax.transAxes, 
                    fontsize=10, fontweight='bold')
            
            ax.text(mx,.60,f'{w} x {h} px', 
                    transform=ax.transAxes,
                    fontsize=12)

    else:
        raise ValueError('show_image_stats must be a boolean or list of strings')
    
    return ax

    
def viewCrossSection(filepath, axis_norm=1,slice_depth=0.5, figsize=(6,5), ax=None):
    pltax = plotCrossSection(filepath,axis_norm,slice_depth,figsize,ax)
    plt.show()

def plot_from_df(df, dfRowIndices, filepath_colname='filepath', 
                 axis_norm=1,slice_depth=0.5, 
                 figsize=(12,12), figdims:tuple=None,
                 show_image_stats=False,
                 title_features=None, title=None):
    
    if figdims:
        if len(figdims) == 2:
            total = len(dfRowIndices)
            if figdims[0] and figdims[1]:
                nrows,ncols = figdims
            elif figdims[0]:
                nrows = figdims[0]
                ncols = int(np.ceil(total/nrows))
            elif figdims[1]:
                ncols = figdims[1]
                nrows = int(np.ceil(total/ncols))
            else:
                raise ValueError("at least one of figdims[0] and figdims[1] must be specified.")
        else:
            raise ValueError("figdims must be a two-tuple.")
    else:
        gdo = GridDimensionOptimizer()
        nrows, ncols = gdo.optimal_grid(len(dfRowIndices))


    _,ax = plt.subplots(nrows,ncols,figsize=figsize,constrained_layout=True)

    for i,ri in enumerate(dfRowIndices):
        filepath = df.iloc[ri][filepath_colname]
        filename = os.path.basename(filepath)
        
        if title is None and title_features is None:
            title = os.path.splitext(filename)[0]
    
        if title_features:
            title = ' | '.join([ str(df.iloc[ri][feature]) for feature in title_features ])

        plotCrossSection(filepath, 
                         axis_norm, 
                         slice_depth, 
                         figsize, 
                         title, 
                         show_image_stats=show_image_stats,
                         ax=ax.flat[i])



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

def plotImagesByPatient(df, 
                        patient_id, 
                        patient_id_colname='patient_id',
                        filepath_colname='filepath',
                        title_features=['image_type','dx_class','laterality'],
                        feature_headers=False,
                        figdims=None, figsize=(12,12) 
                        ):
    ptdf = df[df[patient_id_colname] == patient_id]
    pt_bscans = ptdf[ptdf['image_type'].str.contains('scan')]
    # for row in pt_bscans.iterrows
    
    total = len(ptdf)
    n_bscans = len()

    if figdims:
        nrows,ncols = figdims
    else:
        gdo = GridDimensionOptimizer()
        nrows, ncols = gdo.optimal_grid(total)

    fig,ax = plt.subplots(nrows,ncols,figsize=figsize)
    
    axf = ax.flat

    for i in range(total):
        ptrecord = ptdf.iloc[i]
        filepath = ptrecord[filepath_colname]
        filename = os.path.basename(filepath)
        imgArr,_ = crossSection(filepath)
        axf[i].imshow(imgArr)
        try:
            if feature_headers:
                title = " | ".join([f"{k}: {v}" for k,v in ptrecord[title_features].items()])
            else:
                title = " | ".join([f"{v}" for k,v in ptrecord[title_features].items()])
        except:
            title = os.path.splitext(filename)[0]

        axf[i].set_title(title,fontsize=10)

    plt.suptitle(f"Patient {patient_id}",fontsize=12)
    plt.tight_layout()

    return fig

def plotbybscans(df,patient_id,
                 pid_colname='patient_id',
                 filepath_colname='filepath',
                 title_features = ['image_type','dx_class','laterality'],
                 figdims=None,
                 figsize=(12,10),
                 display_filenames=False,
                 savefig=None
                ):
    
    ptdf = df[df[pid_colname]==patient_id]
    ptdf = ptdf.reset_index().drop(columns='index')
    
    image_indices = []
    bscan_indices = ptdf[ptdf['image_type'].str.lower().str.match(r'b.scan')].index
    for bidx in bscan_indices:
        try:
            # Check if there is a mask image immediately below the b-scan;
            # if not, then it has no masks
            itype_rowbelow = ptdf.iloc[bidx+1]['image_type'].lower()
            if re.match(r"cup|disc",itype_rowbelow):
                # get the two rows below the b-scan row (since the max is two masks per b-scan) 
                masks = ptdf.iloc[bidx+1:bidx+3,:]
                
                # and filter out only the ones that have masks
                # i.e., could result in either 2 or 1 total masks
                masks = masks[masks['image_type'].str.lower().str.match(r'cup|disc')]

                # extract the row indices the masks
                mask_indices = list(masks.index)
            else:
                # Even if the second row down from the b-scan is a mask, 
                # but the one immediately below is not (i.e. it's a b-scan), 
                # then that second row down that is a mask is technicaly a mask 
                # for the b-scan directly below the one of interest.
                # Here, in all other cases, we simple set the index for both
                # cup and disc masks to be None:
                mask_indices = [None,None]
        except:
            # Try-Except block does the same thing, accounts for the case
            # when the b-scan is the last row in the df and has no masks
            # in which case attempting to index df.iloc[bidx+3 ] would be
            # out of bounds / return an error
            mask_indices = [None,None]

        # This way, len(image_indices) will always be equal to
        # len(bscan_indices) * 3, corresponding to n_bscans rows
        # and 3 columns (bscan and its maximum of two masks)

        # Extend image_indices in 3-value batches, in the order
        # (bscan, cup, disc) with each match corresponding with
        # a resulting row within the `plt.subplots` plot
        image_indices.extend([bidx]+mask_indices)
    
    # Iterate through image_indices to plot the images, skipping 
    # subplots when the image index is None (i.e., no mask)
    # Result should be that every row starts with the b-scan image
    # and includes any existing masks to the right of the b-scan.
    nrows = len(bscan_indices)
    ncols = 3
    fig,ax = plt.subplots(nrows,ncols,figsize=figsize)
    axf = ax.flat
    fig.suptitle(f"B-Scans & Masks for Patient #{patient_id}", 
                 fontweight='bold',
                 fontsize=12
                )
    for i,idx in enumerate(image_indices):
        if idx is not None:
            row = ptdf.iloc[idx]
            imgarr = cv2.imread(row['filepath'])
            h,w,c = imgarr.shape
            axf[i].imshow(imgarr)
            title = " | ".join([row[feat] for feat in title_features])
            filename = os.path.basename(row['filepath'])
            if display_filenames:
                axf[i].text(x=int(0.15*w),
                            y=int(0.1*h),
                            s=filename,
                            fontsize=8,
                            color='lime',
                            rotation=0
                           )
            axf[i].set_title(title)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        print(f"Figure saved as {savefig}")


def viewImagesByPatient(df, 
                        patient_id, 
                        patient_id_colname='patient_id',
                        filepath_colname='filepath',
                        title_features=['image_type','dx_class','laterality'],
                        figdims=None, figsize=(12,12) 
                        ):
    ptdf = df[df[patient_id_colname] == patient_id]
    
    _ = plotImagesByPatient(ptdf, patient_id, patient_id_colname, filepath_colname, title_features, figdims, figsize)
    
    plt.show()

    



