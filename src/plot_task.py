"""
Osdag Screening Task: Xarray and Plotly/PyPlot
================================================

Task 1: 2D SFD & BMD for central longitudinal girder 
        (elements: [15,24,33,42,51,60,69,78,83])

Task 2: 3D SFD & BMD for all girders (5 girders total)
        Similar to MIDAS visualization style

Author: Kislay Anand
Date: 03 February 2026

Requirements:
 - screening_task.nc dataset in ../data/ directory
 - node.py and element.py in same folder (geometry definitions)
 - xarray, numpy, plotly, scipy packages
"""

import os
import sys
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

# Import geometry dictionaries from uploaded files
# node.py contains `nodes` dict: node_id -> [x, y, z]
# element.py contains `members` dict: element_id -> [start_node_id, end_node_id]
try:
    from node import nodes
    from element import members
except ImportError:
    print("ERROR: Cannot import node.py or element.py")
    print("Please ensure these files are in the same directory as this script")
    sys.exit(1)


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "screening_task.nc")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
HTML_DIR = os.path.join(OUTPUT_DIR, "interactive")

# Create output directories if they don't exist
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# Central girder (Girder 3) - as specified in task
CENTRAL_GIRDER_ELEMENTS = [15, 24, 33, 42, 51, 60, 69, 78, 83]
CENTRAL_GIRDER_NODES = [3, 13, 18, 23, 28, 33, 38, 43, 48, 8]  # 10 nodes for 9 elements

# All 5 girders' node and element tags (Task 2)
GIRDERS = {
    "Girder 1": {
        "elements": [13, 22, 31, 40, 49, 58, 67, 76, 81], 
        "nodes": [1, 11, 16, 21, 26, 31, 36, 41, 46, 6]
    },
    "Girder 2": {
        "elements": [14, 23, 32, 41, 50, 59, 68, 77, 82], 
        "nodes": [2, 12, 17, 22, 27, 32, 37, 42, 47, 7]
    },
    "Girder 3": {
        "elements": [15, 24, 33, 42, 51, 60, 69, 78, 83], 
        "nodes": [3, 13, 18, 23, 28, 33, 38, 43, 48, 8]
    },
    "Girder 4": {
        "elements": [16, 25, 34, 43, 52, 61, 70, 79, 84], 
        "nodes": [4, 14, 19, 24, 29, 34, 39, 44, 49, 9]
    },
    "Girder 5": {
        "elements": [17, 26, 35, 44, 53, 62, 71, 80, 85], 
        "nodes": [5, 15, 20, 25, 30, 35, 40, 45, 50, 10]
    },
}


# ============================================================================
# DATA LOADING AND EXTRACTION FUNCTIONS
# ============================================================================

def load_dataset(path):
    """
    Load the Xarray NetCDF dataset containing structural analysis results.
    
    Expected dataset structure:
        - Coordinates: Element (element IDs), Component (force/moment names)
        - Data variable: 'forces' containing all force and moment values
        
    Returns:
        xr.Dataset: Loaded dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    ds = xr.open_dataset(path)
    print(f"✓ Dataset loaded successfully")
    print(f"  Total elements in dataset: {len(ds.coords['Element'])}")
    print(f"  Available components: {list(ds.coords['Component'].values)}")
    return ds


def get_component_for_element(ds, element_id, component_name):
    """
    Extract a specific component value for a given element from the dataset.
    
    Args:
        ds: xarray Dataset
        element_id: int - Element ID to query
        component_name: str - Component name (e.g., 'Mz_i', 'Mz_j', 'Vy_i', 'Vy_j')
        
    Returns:
        float: Component value
        
    Raises:
        KeyError: If element_id or component_name not found in dataset
    """
    # Find index of element in dataset coordinates
    elems = ds.coords['Element'].values
    
    # Handle potential dtype issues in element coordinate
    try:
        idx = int(np.where(elems == element_id)[0][0])
    except (IndexError, ValueError) as e:
        # Fallback: search by converting to int
        idx = None
        for i, val in enumerate(elems):
            if int(val) == int(element_id):
                idx = i
                break
        if idx is None:
            raise KeyError(
                f"Element {element_id} not found in dataset. "
                f"Available elements: {elems[:10]}..."
            ) from e
    
    # Check if component exists
    comp_names = list(ds.coords['Component'].values)
    if component_name not in comp_names:
        raise KeyError(
            f"Component '{component_name}' not in dataset. "
            f"Available: {comp_names}"
        )
    
    # Extract and return value
    value = float(ds['forces'].isel(Element=idx).sel(Component=component_name).values)
    return value


def build_continuous_girder_values(ds, element_sequence, node_sequence,
                                    comp_i_name, comp_j_name, node_positions):
    """
    Build continuous nodal values along a girder from element-end values.
    
    For structural analysis, each element has values at its i-end (start) and 
    j-end (finish). For adjacent elements in a girder, the j-end of element k 
    should approximately equal the i-end of element k+1 (at their shared node).
    
    This function stitches element values into continuous nodal values by:
    - Using element_0.comp_i for node 0
    - Averaging element_k.comp_j and element_(k+1).comp_i for interior nodes
    - Using element_n.comp_j for the last node
    
    Args:
        ds: xarray Dataset
        element_sequence: list of element IDs forming the girder
        node_sequence: list of node IDs (length = len(element_sequence) + 1)
        comp_i_name: str - Component name for i-end (e.g., 'Mz_i')
        comp_j_name: str - Component name for j-end (e.g., 'Mz_j')
        node_positions: dict mapping node_id -> [x, y, z]
        
    Returns:
        tuple: (positions, values)
            positions: np.array of z-coordinates (longitudinal positions)
            values: np.array of component values at each node
    """
    n_elements = len(element_sequence)
    n_nodes = len(node_sequence)
    
    # Validate input
    assert n_nodes == n_elements + 1, \
        f"Node sequence length ({n_nodes}) must be elements + 1 ({n_elements + 1})"
    
    # Extract component values from dataset for all elements
    comp_i_vals = []
    comp_j_vals = []
    for eid in element_sequence:
        vi = get_component_for_element(ds, eid, comp_i_name)
        vj = get_component_for_element(ds, eid, comp_j_name)
        comp_i_vals.append(vi)
        comp_j_vals.append(vj)
    
    comp_i_vals = np.array(comp_i_vals, dtype=float)
    comp_j_vals = np.array(comp_j_vals, dtype=float)
    
    # Get longitudinal positions (z-coordinate) for each node
    positions = np.array([node_positions[nid][2] for nid in node_sequence], dtype=float)
    
    # Build nodal values by stitching element values
    values = np.zeros(n_nodes, dtype=float)
    
    for k in range(n_nodes):
        if k == 0:
            # First node: use i-end of first element
            values[k] = comp_i_vals[0]
        elif k == n_nodes - 1:
            # Last node: use j-end of last element
            values[k] = comp_j_vals[-1]
        else:
            # Interior nodes: average j-end of element (k-1) and i-end of element k
            # These should be approximately equal at the shared node
            left = comp_j_vals[k - 1]
            right = comp_i_vals[k]
            
            # Handle potential NaN values
            if np.isnan(left) and not np.isnan(right):
                values[k] = right
            elif np.isnan(right) and not np.isnan(left):
                values[k] = left
            elif np.isnan(left) and np.isnan(right):
                values[k] = np.nan
            else:
                # Average the two values
                values[k] = 0.5 * (left + right)
    
    return positions, values


def fill_nan_linear(x, y):
    """
    Fill NaN values in y using linear interpolation based on x positions.
    
    Args:
        x: np.array of positions
        y: np.array of values (may contain NaN)
        
    Returns:
        np.array: y with NaN values filled by linear interpolation
    """
    if np.all(np.isnan(y)):
        return y
    
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        # Not enough points to interpolate; replace NaNs with zeros
        return np.nan_to_num(y, nan=0.0)
    
    # Create interpolation function from non-NaN points
    f = interp1d(x[mask], y[mask], kind='linear', 
                 bounds_error=False, fill_value='extrapolate')
    y_filled = f(x)
    
    return y_filled


# ============================================================================
# TASK 1: 2D PLOTTING FUNCTIONS (BMD & SFD)
# ============================================================================

def plot_2d_bmd_sfd(positions, mz_values, vy_values, title_prefix="Central Girder", 
                    save_html=None, save_png=True):
    """
    Create 2D plots for Bending Moment Diagram (BMD) and Shear Force Diagram (SFD).
    
    The plots are stacked vertically with:
    - Top panel: Bending Moment (Mz)
    - Bottom panel: Shear Force (Vy)
    
    Both plots include:
    - Line plot with markers at nodal positions
    - Zero reference line (dashed black)
    - Proper axis labels and titles
    
    Args:
        positions: np.array of longitudinal positions (z-coordinates)
        mz_values: np.array of bending moment values
        vy_values: np.array of shear force values
        title_prefix: str - Title prefix for the plot
        save_html: str - Filename for interactive HTML output (optional)
        save_png: bool - Whether to save static PNG image
        
    Returns:
        plotly.graph_objects.Figure: The created figure
    """
    # Create figure with 2 subplots (stacked vertically)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f"{title_prefix} — Bending Moment Diagram (Mz)", 
            f"{title_prefix} — Shear Force Diagram (Vy)"
        )
    )
    
    # ===== TOP PANEL: BENDING MOMENT DIAGRAM =====
    # Main BMD line with markers
    fig.add_trace(
        go.Scatter(
            x=positions, 
            y=mz_values, 
            mode='lines+markers',
            name='Bending Moment (Mz)', 
            line=dict(color='#d62728', width=3),
            marker=dict(size=6, color='#d62728'),
            hovertemplate='Position: %{x:.2f}m<br>Mz: %{y:.2e}<extra></extra>'
        ), 
        row=1, col=1
    )
    
    # Zero reference line for BMD
    fig.add_trace(
        go.Scatter(
            x=[positions.min(), positions.max()], 
            y=[0, 0],
            mode='lines', 
            line=dict(color='black', width=1, dash='dash'), 
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    
    # ===== BOTTOM PANEL: SHEAR FORCE DIAGRAM =====
    # Main SFD line with markers
    fig.add_trace(
        go.Scatter(
            x=positions, 
            y=vy_values, 
            mode='lines+markers',
            name='Shear Force (Vy)', 
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6, color='#1f77b4'),
            hovertemplate='Position: %{x:.2f}m<br>Vy: %{y:.2e}<extra></extra>'
        ), 
        row=2, col=1
    )
    
    # Zero reference line for SFD
    fig.add_trace(
        go.Scatter(
            x=[positions.min(), positions.max()], 
            y=[0, 0],
            mode='lines', 
            line=dict(color='black', width=1, dash='dash'), 
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # ===== LAYOUT AND STYLING =====
    fig.update_xaxes(title_text='Longitudinal Position Z (m)', row=2, col=1)
    fig.update_yaxes(title_text='Bending Moment (units)', row=1, col=1)
    fig.update_yaxes(title_text='Shear Force (units)', row=2, col=1)
    
    fig.update_layout(
        title_text=f"{title_prefix} — SFD & BMD Analysis",
        showlegend=True,
        height=800,
        width=1200,
        hovermode='x unified',
        template='plotly_white'
    )
    
    # ===== SAVE OUTPUTS =====
    # Save interactive HTML
    if save_html:
        outpath = os.path.join(HTML_DIR, save_html)
        fig.write_html(outpath)
        print(f"✓ Saved interactive HTML: {outpath}")
    
    # Save static PNG (requires kaleido)
    if save_png:
        try:
            png_filename = f"{title_prefix.replace(' ', '_')}_2D.png"
            png_path = os.path.join(FIG_DIR, png_filename)
            fig.write_image(png_path, scale=2)
            print(f"✓ Saved static PNG: {png_path}")
        except Exception as e:
            print(f"⚠ Could not save PNG (kaleido may not be installed): {e}")
    
    return fig


# ============================================================================
# TASK 2: 3D PLOTTING FUNCTIONS
# ============================================================================

def build_3d_grid_for_girders(ds, girders_dict, component, node_coords):
    """
    Build 3D grid data for plotting all girders' force/moment distributions.
    
    This function creates a structured grid where:
    - Rows represent different girders (5 girders total)
    - Columns represent nodal positions along each girder
    - Values represent the force/moment component at each position
    
    Args:
        ds: xarray Dataset
        girders_dict: dict of girder definitions {name: {elements: [...], nodes: [...]}}
        component: str - 'Mz' for bending moment or 'Vy' for shear force
        node_coords: dict mapping node_id -> [x, y, z]
        
    Returns:
        tuple: (X, Z, Yvals, girder_names)
            X: 2D array of x-coordinates (transverse positions)
            Z: 2D array of z-coordinates (longitudinal positions)
            Yvals: 2D array of component values
            girder_names: list of girder names
    """
    girder_names = list(girders_dict.keys())
    ny = len(girder_names)  # Number of girders (rows)
    
    # All girders should have same number of nodes
    sample_nodes = girders_dict[girder_names[0]]['nodes']
    nx = len(sample_nodes)  # Number of nodes per girder (columns)
    
    # Initialize output arrays
    X = np.zeros((ny, nx), dtype=float)      # Transverse positions (x)
    Z = np.zeros((ny, nx), dtype=float)      # Longitudinal positions (z)
    Yvals = np.zeros((ny, nx), dtype=float)  # Force/moment values
    
    # Determine component names based on input
    if component == 'Mz':
        comp_i, comp_j = 'Mz_i', 'Mz_j'
    elif component == 'Vy':
        comp_i, comp_j = 'Vy_i', 'Vy_j'
    else:
        raise ValueError(f"Component must be 'Mz' or 'Vy', got: {component}")
    
    # Process each girder
    for i, gname in enumerate(girder_names):
        g = girders_dict[gname]
        node_seq = g['nodes']      # List of node IDs
        elem_seq = g['elements']   # List of element IDs
        
        # Build continuous nodal values for this girder
        positions, values = build_continuous_girder_values(
            ds, elem_seq, node_seq,
            comp_i, comp_j,
            node_positions=node_coords
        )
        
        # Fill the grid arrays for this girder (row i)
        for j, nid in enumerate(node_seq):
            X[i, j] = float(node_coords[nid][0])  # x-coordinate (transverse)
            Z[i, j] = float(node_coords[nid][2])  # z-coordinate (longitudinal)
            Yvals[i, j] = values[j]               # Component value
    
    return X, Z, Yvals, girder_names


def plot_3d_surface(X, Z, Yvals, girder_names, title, save_html=None, 
                    save_png=True, vertical_scale=None):
    """
    Create 3D surface plot showing force/moment distribution across all girders.
    
    This visualization is similar to MIDAS post-processing style where:
    - X-axis: Transverse direction (bridge width)
    - Y-axis: Vertical direction (force/moment magnitude, scaled for visibility)
    - Z-axis: Longitudinal direction (bridge length)
    
    The plot includes:
    - Colored mesh surface representing force/moment magnitudes
    - Deck gridlines showing the structure layout
    - Interactive 3D rotation and zoom
    
    Args:
        X: 2D array of x-coordinates (shape: n_girders × n_nodes)
        Z: 2D array of z-coordinates
        Yvals: 2D array of force/moment values
        girder_names: list of girder names
        title: str - Plot title
        save_html: str - Filename for HTML output (optional)
        save_png: bool - Whether to save PNG
        vertical_scale: float - Vertical scaling factor (auto-calculated if None)
        
    Returns:
        plotly.graph_objects.Figure: The created 3D figure
    """
    # ===== AUTO-CALCULATE VERTICAL SCALE =====
    if vertical_scale is None:
        # Scale so vertical range is ~8-10% of longitudinal span
        zspan = Z.max() - Z.min()
        max_abs_val = np.nanmax(np.abs(Yvals))
        
        if max_abs_val > 1e-9:
            vertical_scale = 0.08 * zspan / max_abs_val
        else:
            vertical_scale = 1.0
    
    # Apply vertical scaling for visualization
    Y_plot = Yvals * vertical_scale
    
    # Calculate magnitude for coloring
    mag = np.abs(Yvals)
    
    # ===== CREATE FIGURE =====
    fig = go.Figure()
    
    # ===== ADD DECK GRIDLINES =====
    # Draw longitudinal lines for each girder
    for i in range(X.shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=X[i, :], 
                y=Y_plot[i, :], 
                z=Z[i, :],
                mode='lines', 
                line=dict(width=3, color='gray'),
                name=f"{girder_names[i]} centerline",
                showlegend=(i == 0),
                hovertemplate=f'{girder_names[i]}<br>Position: %{{z:.2f}}m<br>Value: %{{customdata:.2e}}<extra></extra>',
                customdata=Yvals[i, :]
            )
        )
    
    # ===== CREATE 3D MESH SURFACE =====
    # Build triangular mesh from grid points
    ny, nx = X.shape
    
    # Flatten vertex coordinates
    verts_x = X.flatten()
    verts_y = Y_plot.flatten()
    verts_z = Z.flatten()
    
    # Build face connectivity (two triangles per quad)
    faces_i = []
    faces_j = []
    faces_k = []
    
    def idx(i, j):
        """Convert 2D grid index to 1D vertex index"""
        return i * nx + j
    
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Quad vertices:
            #   a --- b
            #   |     |
            #   c --- d
            a = idx(i, j)
            b = idx(i, j + 1)
            c = idx(i + 1, j)
            d = idx(i + 1, j + 1)
            
            # Triangle 1: a-b-c
            faces_i.append(a)
            faces_j.append(b)
            faces_k.append(c)
            
            # Triangle 2: b-d-c
            faces_i.append(b)
            faces_j.append(d)
            faces_k.append(c)
    
    # Add mesh surface with color based on magnitude
    mesh = go.Mesh3d(
        x=verts_x, 
        y=verts_y, 
        z=verts_z,
        i=faces_i, 
        j=faces_j, 
        k=faces_k,
        intensity=mag.flatten(),
        colorscale='Viridis',
        colorbar=dict(
            title='|Value|',
            x=1.02
        ),
        opacity=0.85,
        name='Force/Moment Surface',
        hoverinfo='skip'
    )
    fig.add_trace(mesh)
    
    # ===== LAYOUT AND STYLING =====
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='Transverse X (m)',
            yaxis_title=f'{component} (scaled × {vertical_scale:.2g})',
            zaxis_title='Longitudinal Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=1400,
        height=800,
        showlegend=True
    )
    
    # Extract component name for filename
    component = title.split(':')[0] if ':' in title else 'component'
    
    # ===== SAVE OUTPUTS =====
    if save_html:
        outpath = os.path.join(HTML_DIR, save_html)
        fig.write_html(outpath)
        print(f"✓ Saved interactive 3D HTML: {outpath}")
    
    if save_png:
        try:
            png_filename = f"3D_{component.replace(' ', '_')}.png"
            png_path = os.path.join(FIG_DIR, png_filename)
            fig.write_image(png_path, scale=2, width=1400, height=800)
            print(f"✓ Saved static 3D PNG: {png_path}")
        except Exception as e:
            print(f"⚠ Could not save PNG: {e}")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for the screening task.
    
    Workflow:
    1. Load Xarray dataset
    2. Task 1: Create 2D BMD & SFD for central girder
    3. Task 2: Create 3D BMD & SFD for all girders
    4. Save all outputs (HTML and PNG files)
    """
    print("=" * 70)
    print("OSDAG SCREENING TASK: Xarray and Plotly/PyPlot")
    print("=" * 70)
    print()
    
    # ===== LOAD DATASET =====
    print("Step 1: Loading Xarray dataset...")
    try:
        ds = load_dataset(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please ensure screening_task.nc is in the ../data/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        sys.exit(1)
    print()
    
    # ===== TASK 1: 2D BMD & SFD FOR CENTRAL GIRDER =====
    print("=" * 70)
    print("TASK 1: 2D Bending Moment & Shear Force Diagrams")
    print("=" * 70)
    print(f"Central girder elements: {CENTRAL_GIRDER_ELEMENTS}")
    print(f"Central girder nodes: {CENTRAL_GIRDER_NODES}")
    print()
    
    try:
        # Build continuous values for bending moment (Mz)
        print("Extracting Mz (Bending Moment) data...")
        pos_central, mz_vals = build_continuous_girder_values(
            ds,
            CENTRAL_GIRDER_ELEMENTS,
            CENTRAL_GIRDER_NODES,
            comp_i_name='Mz_i', 
            comp_j_name='Mz_j',
            node_positions=nodes
        )
        
        # Build continuous values for shear force (Vy)
        print("Extracting Vy (Shear Force) data...")
        _, vy_vals = build_continuous_girder_values(
            ds,
            CENTRAL_GIRDER_ELEMENTS,
            CENTRAL_GIRDER_NODES,
            comp_i_name='Vy_i', 
            comp_j_name='Vy_j',
            node_positions=nodes
        )
        
        # Fill any NaN values using linear interpolation
        print("Interpolating any missing values...")
        mz_vals_filled = fill_nan_linear(pos_central, mz_vals)
        vy_vals_filled = fill_nan_linear(pos_central, vy_vals)
        
        # Create and save 2D plots
        print("Creating 2D plots...")
        fig2d = plot_2d_bmd_sfd(
            pos_central, 
            mz_vals_filled, 
            vy_vals_filled, 
            title_prefix="Central Girder (Girder 3)",
            save_html="task1_central_girder_2d.html",
            save_png=True
        )
        
        print("✓ Task 1 completed successfully!")
        print()
        
    except Exception as e:
        print(f"ERROR in Task 1: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ===== TASK 2: 3D BMD & SFD FOR ALL GIRDERS =====
    print("=" * 70)
    print("TASK 2: 3D Force & Moment Diagrams (All Girders)")
    print("=" * 70)
    print(f"Processing {len(GIRDERS)} girders...")
    print()
    
    try:
        # Build 3D grid for shear force (Vy)
        print("Building 3D grid for Shear Force (Vy)...")
        X_vy, Z_vy, Y_vy, gnames = build_3d_grid_for_girders(
            ds, GIRDERS, 
            component='Vy', 
            node_coords=nodes
        )
        
        # Build 3D grid for bending moment (Mz)
        print("Building 3D grid for Bending Moment (Mz)...")
        X_mz, Z_mz, Y_mz, _ = build_3d_grid_for_girders(
            ds, GIRDERS, 
            component='Mz', 
            node_coords=nodes
        )
        
        # Calculate appropriate vertical scaling factors
        zspan = Z_vy.max() - Z_vy.min()
        scale_vy = 0.08 * zspan / (np.nanmax(np.abs(Y_vy)) + 1e-9)
        scale_mz = 0.08 * zspan / (np.nanmax(np.abs(Y_mz)) + 1e-9)
        
        print(f"Calculated vertical scales: Vy={scale_vy:.4g}, Mz={scale_mz:.4g}")
        print()
        
        # Create 3D plot for shear force
        print("Creating 3D Shear Force plot...")
        fig3d_vy = plot_3d_surface(
            X_vy, Z_vy, Y_vy, gnames,
            title="Vy: 3D Shear Force Distribution (MIDAS-style)",
            save_html="task2_3d_shear_force.html",
            save_png=True,
            vertical_scale=scale_vy
        )
        
        # Create 3D plot for bending moment
        print("Creating 3D Bending Moment plot...")
        fig3d_mz = plot_3d_surface(
            X_mz, Z_mz, Y_mz, gnames,
            title="Mz: 3D Bending Moment Distribution (MIDAS-style)",
            save_html="task2_3d_bending_moment.html",
            save_png=True,
            vertical_scale=scale_mz
        )
        
        print("✓ Task 2 completed successfully!")
        print()
        
    except Exception as e:
        print(f"ERROR in Task 2: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ===== SUMMARY =====
    print("=" * 70)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print()
    print("Output files created:")
    print(f"  Interactive HTML files: {HTML_DIR}/")
    print(f"    - task1_central_girder_2d.html")
    print(f"    - task2_3d_shear_force.html")
    print(f"    - task2_3d_bending_moment.html")
    print()
    print(f"  Static PNG images: {FIG_DIR}/")
    print(f"    - Central_Girder_(Girder_3)_2D.png")
    print(f"    - 3D_Vy_3D.png")
    print(f"    - 3D_Mz_3D.png")
    print()
    print("You can now:")
    print("  1. Open the HTML files in a web browser for interactive viewing")
    print("  2. Use the PNG files for your report documentation")
    print("  3. Create a video demonstration showing the interactive features")
    print()


if __name__ == "__main__":
    main()
