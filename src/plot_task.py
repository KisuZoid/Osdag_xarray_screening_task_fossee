# src/plot_task.py
"""
Plotting script for Screening Task:
- Task 1: 2D SFD & BMD for central longitudinal girder (elements: [15,24,33,42,51,60,69,78,83])
- Task 2: 3D SFD & BMD for all girders (5 girders, nodes/elements provided)

Requires:
 - screening_task.nc in ../data/
 - node.py and element.py in same folder (imports nodes, members)
"""

import os
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d

# Import geometry dictionaries (these are the uploaded files)
# node.py contains `nodes` dict: node_id -> [x, y, z]
# element.py contains `members` dict: element_id -> [start_node_id, end_node_id]
from node import nodes           # :contentReference[oaicite:4]{index=4}
from element import members     # :contentReference[oaicite:5]{index=5}

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "screening_task.nc")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
HTML_DIR = os.path.join(OUTPUT_DIR, "interactive")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

# Central girder elements & nodes per problem statement:
CENTRAL_GIRDER_ELEMENTS = [15, 24, 33, 42, 51, 60, 69, 78, 83]
CENTRAL_GIRDER_NODES = [3, 13, 18, 23, 28, 33, 38, 43, 48, 8]  # 10 nodes -> 9 elements

# All girders' node and element tags per problem statement (used for Task-2)
GIRDERS = {
    "girder1": {"elements": [13,22,31,40,49,58,67,76,81], "nodes": [1,11,16,21,26,31,36,41,46,6]},
    "girder2": {"elements": [14,23,32,41,50,59,68,77,82], "nodes": [2,12,17,22,27,32,37,42,47,7]},
    "girder3": {"elements": [15,24,33,42,51,60,69,78,83], "nodes": [3,13,18,23,28,33,38,43,48,8]},
    "girder4": {"elements": [16,25,34,43,52,61,70,79,84], "nodes": [4,14,19,24,29,34,39,44,49,9]},
    "girder5": {"elements": [17,26,35,44,53,62,71,80,85], "nodes": [5,15,20,25,30,35,40,45,50,10]},
}

# Utility: ensure element index map from dataset to element id
def load_dataset(path):
    ds = xr.open_dataset(path)
    # dataset expected structure: coords Element, Component ; data variable 'forces'
    # components include 'Mz_i', 'Mz_j', 'Vy_i', 'Vy_j', 'x','y','z' ...
    return ds

def get_component_for_element(ds, element_id, component_name):
    """
    Return component value for element_id for the given component_name (string).
    ds: xarray dataset
    """
    # Find index of element in ds coordinates
    elems = ds.coords['Element'].values
    # If Element coordinate contains bytes / dtype oddities, ensure direct match
    try:
        idx = int(np.where(elems == element_id)[0][0])
    except Exception as e:
        # fallback - search by python int match
        idx = None
        for i, val in enumerate(elems):
            if int(val) == int(element_id):
                idx = i
                break
        if idx is None:
            raise KeyError(f"Element id {element_id} not found in dataset Element coord") from e

    # Component names are strings in ds.coords['Component']
    comp_names = list(ds.coords['Component'].values)
    if component_name not in comp_names:
        raise KeyError(f"Component '{component_name}' not in dataset. Available: {comp_names}")

    return float(ds['forces'].isel(Element=idx).sel(Component=component_name).values)

def build_continuous_girder_values(ds, element_sequence, node_sequence,
                                  comp_i_name, comp_j_name, node_positions):
    """
    For a girder defined by element_sequence (len n) and node_sequence (len n+1),
    build arrays of positions (longitudinal) and values at each node by taking
    element_i and element_j values and stitching them at shared nodes.

    node_positions: dict node_id -> scalar longitudinal coordinate (we'll use z)
    Returns: (positions [n+1], values [n+1]) where values correspond to nodes.
    """

    n_e = len(element_sequence)
    n_nodes = len(node_sequence)
    assert n_nodes == n_e + 1, "node sequence length must be elements + 1"

    # For each element fetch comp_i and comp_j values.
    comp_i_vals = []
    comp_j_vals = []
    for eid in element_sequence:
        vi = get_component_for_element(ds, eid, comp_i_name)
        vj = get_component_for_element(ds, eid, comp_j_name)
        comp_i_vals.append(vi)
        comp_j_vals.append(vj)
    comp_i_vals = np.array(comp_i_vals, dtype=float)
    comp_j_vals = np.array(comp_j_vals, dtype=float)

    # Build nodal list by taking i of first element, then j of each element (ensures continuity)
    # But we were given node_sequence explictly; use that for coordinates ordering.
    positions = np.array([node_positions[nid][2] for nid in node_sequence], dtype=float)  # z coordinate
    # Now build nodal values:
    # For node k (0..n): if k==0 -> use element 0's Mz_i ; if k==n -> use element n-1's Mz_j
    # else average element_(k-1).Mz_j and element_k.Mz_i (they should be approximately equal)
    values = np.zeros(n_nodes, dtype=float)
    for k in range(n_nodes):
        if k == 0:
            values[k] = comp_i_vals[0]
        elif k == n_nodes - 1:
            values[k] = comp_j_vals[-1]
        else:
            left = comp_j_vals[k - 1]
            right = comp_i_vals[k]
            # handle NaNs by linear logic
            if np.isnan(left) and not np.isnan(right):
                values[k] = right
            elif np.isnan(right) and not np.isnan(left):
                values[k] = left
            elif np.isnan(left) and np.isnan(right):
                values[k] = np.nan
            else:
                values[k] = 0.5 * (left + right)
    return positions, values

# Plot helpers
def plot_2d_bmd_sfd(positions, mz_values, vy_values, title_prefix="Central Girder", save_html=None):
    """
    Create 2 subplots (BMD & SFD) stacked vertically using Plotly.
    positions: array of longitudinal positions (z)
    mz_values: array of bending moment values (at nodes)
    vy_values: array of shear values (at nodes)
    """
    fig = make_2panel_plot(positions, mz_values, vy_values, title_prefix)
    if save_html:
        outpath = os.path.join(HTML_DIR, save_html)
        fig.write_html(outpath)
        print(f"Saved interactive 2D plot: {outpath}")
    # also optionally save PNG
    fig.write_image(os.path.join(FIG_DIR, f"{title_prefix.replace(' ', '_')}_2D.png"), scale=2)
    return fig

def make_2panel_plot(positions, mz_values, vy_values, title_prefix):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=(f"{title_prefix} — Bending Moment (Mz)", f"{title_prefix} — Shear Force (Vy)"))

    # BMD
    fig.add_trace(go.Scatter(x=positions, y=mz_values, mode='lines+markers',
                             name='Mz', line=dict(width=3)), row=1, col=1)
    # add zero reference
    fig.add_trace(go.Scatter(x=[positions.min(), positions.max()], y=[0, 0],
                             mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False),
                  row=1, col=1)

    # SFD
    fig.add_trace(go.Scatter(x=positions, y=vy_values, mode='lines+markers',
                             name='Vy', line=dict(width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[positions.min(), positions.max()], y=[0, 0],
                             mode='lines', line=dict(color='black', width=1, dash='dash'), showlegend=False),
                  row=2, col=1)

    fig.update_xaxes(title_text='Longitudinal position (z)', row=2, col=1)
    fig.update_yaxes(title_text='Mz (units per dataset)', row=1, col=1)
    fig.update_yaxes(title_text='Vy (units per dataset)', row=2, col=1)
    fig.update_layout(title_text=f"{title_prefix} — SFD & BMD (plotted using values from dataset)",
                      height=700, width=1000)
    return fig

def build_3d_grid_for_girders(ds, girders_dict, component='Vy', node_coords=nodes):
    """
    Build a 2D grid (girders across width, nodes along length) of component values.
    component: either 'Vy' or 'Mz' (we will use node values built from element i/j)
    Returns:
      X (ny x nx) grid of x-coordinates
      Z (ny x nx) grid of z-coordinates (longitudinal)
      Yvals (ny x nx) grid of vertical magnitudes (component values)
      girder_labels (list)
    ny = number of girders, nx = number of nodes along each girder
    """
    girder_names = list(girders_dict.keys())
    ny = len(girder_names)
    # assume all girders share equal number of nodes
    sample_nodes = girders_dict[girder_names[0]]['nodes']
    nx = len(sample_nodes)

    X = np.zeros((ny, nx), dtype=float)
    Z = np.zeros((ny, nx), dtype=float)
    Yvals = np.zeros((ny, nx), dtype=float)
    for i, gname in enumerate(girder_names):
        g = girders_dict[gname]
        node_seq = g['nodes']  # list of node ids (length nx)
        elem_seq = g['elements']
        # We need nodal values: use element Mz_i / Mz_j or Vy_i / Vy_j to build node values
        # For Mz use component names 'Mz_i' and 'Mz_j', for Vy use 'Vy_i' and 'Vy_j'
        if component == 'Mz':
            comp_i, comp_j = 'Mz_i', 'Mz_j'
        elif component == 'Vy':
            comp_i, comp_j = 'Vy_i', 'Vy_j'
        else:
            raise ValueError("component must be 'Mz' or 'Vy'")

        # create positions and values per girder
        # node_coords is dict node_id->[x,y,z], we need x,z
        positions, values = build_continuous_girder_values(ds, elem_seq, node_seq,
                                                           comp_i, comp_j,
                                                           node_positions=node_coords)
        # Fill arrays row i
        for j, nid in enumerate(node_seq):
            X[i, j] = float(node_coords[nid][0])  # x
            Z[i, j] = float(node_coords[nid][2])  # z (longitudinal)
            Yvals[i, j] = values[j]  # raw component value (sign preserved)

    return X, Z, Yvals, girder_names

def plot_3d_surface(X, Z, Yvals, girder_names, title, save_html=None, vertical_scale=None):
    """
    Plot a surface representing Yvals over X,Z (both same shape), using Plotly.
    vertical_scale: if provided, multiply Yvals by this factor to enhance visibility.
    """
    if vertical_scale is None:
        # choose a scale so vertical range occupies ~10% of max span in Z
        zspan = Z.max() - Z.min()
        yspan = np.nanmax(np.abs(Yvals)) - np.nanmin(np.abs(Yvals)) if np.nanmax(np.abs(Yvals))!=0 else 1.0
        # crude scaling factor:
        vertical_scale = max(1.0, 0.1 * zspan / (np.nanmax(np.abs(Yvals)) + 1e-9))

    Y_plot = Yvals * vertical_scale

    # For nice coloring use magnitude colormap
    mag = np.abs(Yvals)
    fig = go.Figure()

    # Plot deck gridlines (wireframe)
    for i in range(X.shape[0]):
        fig.add_trace(go.Scatter3d(x=X[i, :], y=Y_plot[i, :], z=Z[i, :],
                                   mode='lines', line=dict(width=3, color='gray'),
                                   name=f"{girder_names[i]} deck line", showlegend=(i==0)))

    # Create surface by using X (columns -> girder index order) -> need meshgrid shaped accordingly
    # Plotly surface expects X, Y, Z as 2D arrays; we're supplying z as 'height' but we'll map to y-axis as vertical
    # We'll map axes: x->X, y->Y_plot (vertical), z->Z
    # We'll construct the surface as a mesh via Mesh3d for correct 3d appearance.

    # Create vertices
    ny, nx = X.shape
    verts_x = X.flatten()
    verts_y = Y_plot.flatten()
    verts_z = Z.flatten()
    # Build triangular faces (two triangles per quad)
    faces_i = []
    faces_j = []
    faces_k = []
    def idx(i, j):
        return i * nx + j
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = idx(i, j)
            b = idx(i, j + 1)
            c = idx(i + 1, j)
            d = idx(i + 1, j + 1)
            # triangle a-b-c and b-d-c
            faces_i += [a, b]
            faces_j += [b, d]
            faces_k += [c, c]

    mesh = go.Mesh3d(
        x=verts_x, y=verts_y, z=verts_z,
        i=faces_i, j=faces_j, k=faces_k,
        intensity=mag.flatten(),
        colorscale='Viridis',
        colorbar=dict(title='|value|'),
        opacity=0.9,
        name='magnitude surface'
    )
    fig.add_trace(mesh)

    # Add zero-plane and axes
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (width) [m]',
            yaxis_title=f'{title.split(":")[0]} (vertical, scaled)',
            zaxis_title='Longitudinal Z [m]',
            aspectmode='auto'
        ),
        width=1200, height=700
    )

    if save_html:
        outpath = os.path.join(HTML_DIR, save_html)
        fig.write_html(outpath)
        print(f"Saved interactive 3D plot: {outpath}")

    # Save static image optionally
    fig.write_image(os.path.join(FIG_DIR, f"{title.replace(' ', '_')}.png"), scale=2)
    return fig

# --- Main execution ---
def main():
    print("Loading dataset...")
    ds = load_dataset(DATA_PATH)
    print("Dataset loaded. Available components:", list(ds.coords['Component'].values))

    # Task 1: central girder SFD & BMD
    node_positions = nodes  # nodes dict: node_id -> [x,y,z]
    print("Building central girder (elements:", CENTRAL_GIRDER_ELEMENTS, ")")
    pos_central, mz_vals = build_continuous_girder_values(ds,
                                                         CENTRAL_GIRDER_ELEMENTS,
                                                         CENTRAL_GIRDER_NODES,
                                                         comp_i_name='Mz_i', comp_j_name='Mz_j',
                                                         node_positions=node_positions)
    _, vy_vals = build_continuous_girder_values(ds,
                                                CENTRAL_GIRDER_ELEMENTS,
                                                CENTRAL_GIRDER_NODES,
                                                comp_i_name='Vy_i', comp_j_name='Vy_j',
                                                node_positions=node_positions)

    # Interpolate to fill any NaNs (linear along length) for plot continuity
    def fill_nan_linear(x, y):
        if np.all(np.isnan(y)):
            return y
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            # not enough points to interpolate; replace remaining NaNs with zeros
            return np.nan_to_num(y, nan=0.0)
        f = interp1d(x[mask], y[mask], kind='linear', bounds_error=False, fill_value='extrapolate')
        y_filled = f(x)
        return y_filled

    mz_vals_f = fill_nan_linear(pos_central, mz_vals)
    vy_vals_f = fill_nan_linear(pos_central, vy_vals)

    print("Plotting Task-1: 2D BMD & SFD for central girder...")
    fig2d = plot_2d_bmd_sfd(pos_central, mz_vals_f, vy_vals_f, title_prefix="Central Girder (elements: " + ",".join(map(str,CENTRAL_GIRDER_ELEMENTS))+")", save_html="central_girder_2d.html")
    print("Task-1 done. Saved figure & HTML.")

    # Task 2: 3D SFD & BMD for all girders
    print("Building 3D grids for all girders (Vy and Mz)...")
    X_vy, Z_vy, Y_vy, gnames = build_3d_grid_for_girders(ds, GIRDERS, component='Vy', node_coords=node_positions)
    X_mz, Z_mz, Y_mz, _ = build_3d_grid_for_girders(ds, GIRDERS, component='Mz', node_coords=node_positions)

    # Decide a vertical scaling factor visually sensible:
    # Option: set vertical_scale so that the max vertical amplitude equals 8% of longitudinal span
    zspan = Z_vy.max() - Z_vy.min()
    suggested_scale_vy = 0.08 * zspan / (np.nanmax(np.abs(Y_vy)) + 1e-9)
    suggested_scale_mz = 0.08 * zspan / (np.nanmax(np.abs(Y_mz)) + 1e-9)

    print(f"Plotting Task-2: 3D SFD (Vy) with vertical_scale={suggested_scale_vy:.4g} and 3D BMD (Mz) with vertical_scale={suggested_scale_mz:.4g} ...")
    fig3d_vy = plot_3d_surface(X_vy, Z_vy, Y_vy, gnames, title="Vy: 3D Shear Force Surface (raw units, scaled for visualization)", save_html="3d_vy.html", vertical_scale=suggested_scale_vy)
    fig3d_mz = plot_3d_surface(X_mz, Z_mz, Y_mz, gnames, title="Mz: 3D Bending Moment Surface (raw units, scaled for visualization)", save_html="3d_mz.html", vertical_scale=suggested_scale_mz)

    print("Task-2 done. Saved 3D interactive plots & PNGs in outputs/")

if __name__ == "__main__":
    main()
