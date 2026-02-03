# Osdag Screening Task: Xarray and Plotly/PyPlot

## 📋 Project Overview

This project implements the Osdag screening assignment for structural engineering visualization. It creates 2D and 3D visualizations of shear force diagrams (SFD) and bending moment diagrams (BMD) for a bridge grillage structure using Xarray for data handling and Plotly for interactive visualization.

**Osdag** is a free and open-source software for the design and detailing of steel structures.

---

## 🎯 Task Description

### Task 1: 2D SFD & BMD for Central Girder
Create shear force and bending moment diagrams for the central longitudinal girder (Girder 3) consisting of elements `[15, 24, 33, 42, 51, 60, 69, 78, 83]`.

**Requirements:**
- Extract `Mz` (bending moment) and `Vy` (shear force) from Xarray dataset
- Create visually pleasing 2D plots with proper labels and titles
- Use sign convention from dataset without manual flipping
- Display continuous diagrams with markers at nodal positions

### Task 2: 3D SFD & BMD for All Girders
Generate 3D visualizations similar to MIDAS post-processing style showing force/moment distributions across all 5 girders.

**Requirements:**
- Process all 5 girders with correct node coordinates and element connectivity
- Create 3D surface plots with magnitudes extruded vertically
- Ensure proper scaling, axes labels, and color coding
- Match MIDAS-style visualization aesthetic

---

## 📁 Project Structure

```
Osdag_xarray_screening/
├── data/
│   └── screening_task.nc          # Input Xarray dataset
├── src/
│   ├── plot_task.py               # Main plotting script (THIS FILE)
│   ├── element.py                 # Element connectivity data
│   └── node.py                    # Node coordinate data
├── outputs/
│   ├── figures/                   # Static PNG images
│   │   ├── Central_Girder_(Girder_3)_2D.png
│   │   ├── 3D_Vy_3D.png
│   │   └── 3D_Mz_3D.png
│   └── interactive/               # Interactive HTML plots
│       ├── task1_central_girder_2d.html
│       ├── task2_3d_shear_force.html
│       └── task2_3d_bending_moment.html
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── REPORT.pdf                     # Detailed technical report
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download Repository
```bash
git clone https://github.com/KisuZoid/Osdag_xarray_screening_task_fossee
cd Osdag_xarray_screening_task_fossee
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical computations
- `xarray` - Multi-dimensional dataset handling
- `netCDF4` - NetCDF file format support
- `plotly` - Interactive plotting
- `kaleido` - Static image export
- `scipy` - Scientific computing (interpolation)

### Step 3: Verify Data Files
Ensure the following files are present:
- `data/screening_task.nc` - Xarray dataset with structural analysis results
- `src/element.py` - Element connectivity definitions
- `src/node.py` - Node coordinate definitions

---

## ▶️ Usage

### Running the Script

Navigate to the `src/` directory and run:

```bash
cd src
python plot_task.py
```

### Expected Output

The script will:
1. Load the Xarray dataset
2. Extract force and moment data for all elements
3. Create 2D plots for the central girder (Task 1)
4. Create 3D surface plots for all girders (Task 2)
5. Save interactive HTML and static PNG files

**Console output:**
```
======================================================================
OSDAG SCREENING TASK: Xarray and Plotly/PyPlot
======================================================================

Step 1: Loading Xarray dataset...
✓ Dataset loaded successfully
  Total elements in dataset: 85
  Available components: ['Fx_i', 'Fx_j', 'Fy_i', 'Fy_j', 'Fz_i', 'Fz_j', 
                          'Mx_i', 'Mx_j', 'My_i', 'My_j', 'Mz_i', 'Mz_j', 
                          'Vx_i', 'Vx_j', 'Vy_i', 'Vy_j', 'Vz_i', 'Vz_j', ...]

======================================================================
TASK 1: 2D Bending Moment & Shear Force Diagrams
======================================================================
Central girder elements: [15, 24, 33, 42, 51, 60, 69, 78, 83]
Central girder nodes: [3, 13, 18, 23, 28, 33, 38, 43, 48, 8]

Extracting Mz (Bending Moment) data...
Extracting Vy (Shear Force) data...
Interpolating any missing values...
Creating 2D plots...
✓ Saved interactive HTML: ../outputs/interactive/task1_central_girder_2d.html
✓ Saved static PNG: ../outputs/figures/Central_Girder_(Girder_3)_2D.png
✓ Task 1 completed successfully!

======================================================================
TASK 2: 3D Force & Moment Diagrams (All Girders)
======================================================================
Processing 5 girders...

Building 3D grid for Shear Force (Vy)...
Building 3D grid for Bending Moment (Mz)...
Calculated vertical scales: Vy=0.0356, Mz=0.0412

Creating 3D Shear Force plot...
✓ Saved interactive 3D HTML: ../outputs/interactive/task2_3d_shear_force.html
✓ Saved static 3D PNG: ../outputs/figures/3D_Vy_3D.png

Creating 3D Bending Moment plot...
✓ Saved interactive 3D HTML: ../outputs/interactive/task2_3d_bending_moment.html
✓ Saved static 3D PNG: ../outputs/figures/3D_Mz_3D.png
✓ Task 2 completed successfully!

======================================================================
ALL TASKS COMPLETED SUCCESSFULLY!
======================================================================
```

---

## 📊 Output Files

### Task 1 Outputs (2D Diagrams)
- **`task1_central_girder_2d.html`** - Interactive 2-panel plot showing:
  - Top panel: Bending Moment Diagram (Mz)
  - Bottom panel: Shear Force Diagram (Vy)
  - Hover for exact values at each position
  - Pan, zoom, and download capabilities

- **`Central_Girder_(Girder_3)_2D.png`** - Static image for reports

### Task 2 Outputs (3D Diagrams)
- **`task2_3d_shear_force.html`** - Interactive 3D shear force visualization
  - Colored surface showing force magnitudes
  - Gridlines showing bridge structure
  - Rotate, pan, and zoom in 3D

- **`task2_3d_bending_moment.html`** - Interactive 3D bending moment visualization
  - Similar to shear force but for bending moments
  - MIDAS-style extruded visualization

- **`3D_Vy_3D.png`** and **`3D_Mz_3D.png`** - Static images

---

## 🔧 Technical Implementation

### Data Processing Workflow

1. **Dataset Loading**
   - Load NetCDF file using `xarray.open_dataset()`
   - Dataset contains `forces` variable with dimensions: (Element, Component)
   - Components include: `Mz_i`, `Mz_j`, `Vy_i`, `Vy_j`, etc.

2. **Value Extraction**
   - For each element, extract `_i` (start node) and `_j` (end node) values
   - Handle potential NaN values and data type conversions

3. **Continuous Girder Construction**
   - Stitch element values into continuous nodal values
   - For interior nodes: average `element_k.j_end` and `element_(k+1).i_end`
   - First node: use `element_0.i_end`
   - Last node: use `element_n.j_end`

4. **Interpolation**
   - Fill any remaining NaN values using linear interpolation
   - Ensures smooth, continuous plots

5. **3D Grid Building**
   - Create structured grid: rows = girders, columns = nodes
   - Extract x, y, z coordinates for each node
   - Map force/moment values to grid positions

6. **Visualization**
   - Use Plotly for interactive HTML outputs
   - Apply vertical scaling for 3D plots (8% of longitudinal span)
   - Color code by magnitude for intuitive interpretation

### Key Functions

#### `load_dataset(path)`
Loads the Xarray NetCDF dataset and validates structure.

#### `get_component_for_element(ds, element_id, component_name)`
Extracts a specific force/moment component for a given element.

#### `build_continuous_girder_values(ds, element_sequence, node_sequence, comp_i_name, comp_j_name, node_positions)`
Builds continuous nodal values along a girder from element-end values.

#### `plot_2d_bmd_sfd(positions, mz_values, vy_values, title_prefix, save_html, save_png)`
Creates 2-panel 2D plot for BMD and SFD.

#### `build_3d_grid_for_girders(ds, girders_dict, component, node_coords)`
Constructs 3D grid data for all girders.

#### `plot_3d_surface(X, Z, Yvals, girder_names, title, save_html, save_png, vertical_scale)`
Creates 3D surface plot with MIDAS-style visualization.

---

## 🎨 Visualization Features

### 2D Plots (Task 1)
- ✅ Clean, professional layout with two vertically stacked panels
- ✅ Clear axis labels and titles
- ✅ Zero reference lines (dashed black)
- ✅ Markers at nodal positions
- ✅ Hover tooltips showing exact values
- ✅ Interactive zoom and pan
- ✅ Download plot as PNG from browser

### 3D Plots (Task 2)
- ✅ MIDAS-style visualization with vertical extrusion
- ✅ Color-coded surface (Viridis colormap) based on magnitude
- ✅ Deck gridlines showing structure layout
- ✅ Interactive 3D rotation, pan, and zoom
- ✅ Proper axis labels (X: transverse, Y: vertical scaled, Z: longitudinal)
- ✅ Automatic vertical scaling (8% of span)
- ✅ Hover tooltips with girder names and values

---

## 📐 Girder Definitions

### Central Girder (Girder 3) - Used in Task 1
- **Elements:** `[15, 24, 33, 42, 51, 60, 69, 78, 83]`
- **Nodes:** `[3, 13, 18, 23, 28, 33, 38, 43, 48, 8]`

### All Girders - Used in Task 2

| Girder | Elements | Nodes |
|--------|----------|-------|
| **Girder 1** | `[13, 22, 31, 40, 49, 58, 67, 76, 81]` | `[1, 11, 16, 21, 26, 31, 36, 41, 46, 6]` |
| **Girder 2** | `[14, 23, 32, 41, 50, 59, 68, 77, 82]` | `[2, 12, 17, 22, 27, 32, 37, 42, 47, 7]` |
| **Girder 3** | `[15, 24, 33, 42, 51, 60, 69, 78, 83]` | `[3, 13, 18, 23, 28, 33, 38, 43, 48, 8]` |
| **Girder 4** | `[16, 25, 34, 43, 52, 61, 70, 79, 84]` | `[4, 14, 19, 24, 29, 34, 39, 44, 49, 9]` |
| **Girder 5** | `[17, 26, 35, 44, 53, 62, 71, 80, 85]` | `[5, 15, 20, 25, 30, 35, 40, 45, 50, 10]` |

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'xarray'"
**Solution:** Install dependencies using `pip install -r requirements.txt`

### Issue: "FileNotFoundError: Dataset not found"
**Solution:** 
- Verify `screening_task.nc` is in `../data/` relative to the script
- Check file permissions
- Ensure correct path structure

### Issue: "Cannot save PNG images"
**Solution:** 
- Install kaleido: `pip install kaleido`
- If kaleido fails, HTML outputs will still work
- Use browser's built-in download feature for images

### Issue: "Element X not found in dataset"
**Solution:**
- Verify element IDs in `element.py` match dataset
- Check dataset contents: `python -c "import xarray as xr; print(xr.open_dataset('data/screening_task.nc'))"`

### Issue: "ValueError: component must be 'Mz' or 'Vy'"
**Solution:**
- Ensure dataset contains 'Mz_i', 'Mz_j', 'Vy_i', 'Vy_j' components
- Check component names in dataset: look at `Component` coordinate

---

## 📝 Code Quality

### Features
- ✅ Comprehensive docstrings for all functions
- ✅ Type hints in function signatures (where applicable)
- ✅ Detailed inline comments explaining logic
- ✅ Error handling with informative messages
- ✅ Modular design with clear separation of concerns
- ✅ PEP 8 compliant code style
- ✅ Descriptive variable names
- ✅ Progress messages during execution

### Testing Recommendations
```python
# Test dataset loading
import xarray as xr
ds = xr.open_dataset('../data/screening_task.nc')
print(ds)

# Test element extraction
from plot_task import get_component_for_element
value = get_component_for_element(ds, 15, 'Mz_i')
print(f"Element 15 Mz_i: {value}")

# Test node coordinates
from node import nodes
print(f"Node 3 coordinates: {nodes[3]}")

# Test element connectivity
from element import members
print(f"Element 15 connects nodes: {members[15]}")
```

---

## 🎓 Academic Integrity

This screening task submission is licensed under **Creative Commons Attribution-ShareAlike 4.0 International License** by FOSSEE.

**Acknowledgments:**
- Osdag development team
- FOSSEE (Free/Libre and Open Source Software for Education)
- ospgrillage library for grillage modeling

## 📚 References

1. **Osdag Official Website:** https://osdag.fossee.in/
2. **Xarray Documentation:** https://docs.xarray.dev/
3. **Plotly Python Documentation:** https://plotly.com/python/
4. **MIDAS Civil:** For visualization style reference
5. **Structural Analysis Fundamentals:** McCormac & Csernak (2012)

---

## ✅ Submission Checklist

- [x] Code runs without errors
- [x] All required outputs generated (2D + 3D plots)
- [x] Interactive HTML files created
- [x] Static PNG images saved
- [x] Code well-commented and documented
- [x] README.md comprehensive
- [x] requirements.txt complete
- [x] REPORT.pdf with technical explanation
- [x] Video demonstration recorded
- [x] GitHub repository public with osdag-admin as collaborator
- [x] All files zipped for submission

---

## 📄 License

This project is submitted under the **Creative Commons Attribution-ShareAlike 4.0 International License** as required by FOSSEE for the Osdag screening task.

---

## 👤 Author

**Kislay Anand**  
Submission Date: 03 February 2026  
Osdag Screening Task: Xarray and Plotly/PyPlot

---
