# Quick Setup Guide

## Installation (5 minutes)

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

If you encounter issues, install individually:
```bash
pip install numpy xarray netCDF4 plotly scipy kaleido
```

### 2. Verify File Structure
Ensure your directory looks like this:
```
project/
├── data/
│   └── screening_task.nc
├── src/  (or current directory)
│   ├── plot_task.py
│   ├── element.py
│   ├── node.py
└── outputs/  (will be created automatically)
```

### 3. Run the Script
```bash
cd src
python plot_task.py
```

## Expected Output

### Console Output:
```
======================================================================
OSDAG SCREENING TASK: Xarray and Plotly/PyPlot
======================================================================

Step 1: Loading Xarray dataset...
✓ Dataset loaded successfully
  Total elements in dataset: 85
  Available components: [...]

======================================================================
TASK 1: 2D Bending Moment & Shear Force Diagrams
======================================================================
...
✓ Task 1 completed successfully!

======================================================================
TASK 2: 3D Force & Moment Diagrams (All Girders)
======================================================================
...
✓ Task 2 completed successfully!

======================================================================
ALL TASKS COMPLETED SUCCESSFULLY!
======================================================================
```

### Files Created:
- `outputs/interactive/task1_central_girder_2d.html`
- `outputs/interactive/task2_3d_shear_force.html`
- `outputs/interactive/task2_3d_bending_moment.html`
- `outputs/figures/Central_Girder_(Girder_3)_2D.png`
- `outputs/figures/3D_Vy_3D.png`
- `outputs/figures/3D_Mz_3D.png`

## Viewing Results

### Interactive HTML (Recommended)
1. Navigate to `outputs/interactive/`
2. Double-click any `.html` file to open in browser
3. Use mouse to interact:
   - **Drag**: Pan the plot
   - **Scroll**: Zoom in/out
   - **Right-drag** (3D only): Rotate view
   - **Hover**: See exact values

### Static PNG Images
- Open with any image viewer
- Use in reports, presentations, documents
- High resolution (1200-1400px wide)

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'xarray'"
**Solution:** Run `pip install xarray netCDF4`

### Error: "FileNotFoundError: Dataset not found"
**Solution:** 
- Verify `screening_task.nc` is in `../data/` directory relative to script
- Or adjust `DATA_PATH` variable in `plot_task.py`

### Error: "Cannot import node or element"
**Solution:** Ensure `node.py` and `element.py` are in same directory as `plot_task.py`

### Warning: "Could not save PNG"
**Solution:** 
- Install kaleido: `pip install kaleido`
- Or ignore (HTML outputs work fine without it)

### Issue: 3D plots look flat
**Solution:** This is normal - the script automatically applies vertical scaling. Check the y-axis label for the scale factor used.

## Next Steps

1. **Create video demonstration:**
   - Record screen showing interactive features
   - Upload to YouTube as unlisted
   - Include link in submission

2. **Prepare GitHub repository:**
   - Initialize: `git init`
   - Add files: `git add .`
   - Commit: `git commit -m "Osdag screening task"`
   - Push to GitHub
   - Add `osdag-admin` as collaborator

3. **Compile submission:**
   - ZIP all files: code, outputs, report
   - Include YouTube video link
   - Submit via specified channel

## Support

For questions:
- Join Osdag Discord: [Link in task description]
- Post in #screening-task-help channel
- Include error messages and screenshots

---

**Good luck with your submission! 🚀**
