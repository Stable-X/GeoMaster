# GeoMaster: Advanced Geometry Enhancement for High-Resolution 3D Modeling

GeoMaster is a state-of-the-art tool designed to enhance the geometric details of 3D models, providing high-resolution outputs suitable for various applications such as gaming, virtual reality, and 3D printing.

## Installation

To get started with GeoMaster, follow these steps:

### Clone the Repository

First, clone the GeoMaster repository to your local machine:
```bash
git clone https://github.com/hugoycj/GeoMaster.git --recursive
cd GeoMaster
```

### Step 1: Setup the environment

Create the geomaster environment using conda:

```bash
conda create -n geomaster python=3.9
conda activate geomaster
```

### Step 2: Install Requirements

First, navigate to the submodule and install dependencies for submodules:

```bash
cd submodule/PromptDA
pip install -r requirements.txt
pip install -e .

cd ../nvdiffrast
pip install -e .
```

Then, install the dependencies for this project (GeoMaster):

```bash
cd ../..
pip install -r requirements.txt
pip install -e .
```

### Step 3: Install Gaustudio

Finally, install `gaustudio` from its GitHub repository:
```bash
pip install git+https://github.com/GAP-LAB-CUHK-SZ/gaustudio
```

## Data Preparation

Before running GeoMaster, ensure that your data is organized in the following structure:
```
data_001
├── confidence  [optional]
├── depths      [required]
├── images      [required]
└── sparse      [required]
```
## Usage
### Reconstruct Mesh
#### Step 1: Prepare the data
First, you need to prepare the input data using the `gm-prepare-data` command:

```
gm-prepare-data -s ./examples/glt
```

This command will generate the necessary files and directories for the mesh reconstruction process.

#### Step 2: Reconstruct the mesh
Next, you can use the `gm-recon` command to reconstruct the mesh from the prepared data:

```
gm-recon -s ./examples/glt -m ./examples/glt/visual_hull.ply
```

### Enhancing Mesh

To enhance the geometric detail of your 3D model using `gm-refine`, use the following command:

```bash
gm-refine -s ${INPUT_DIR} -m ${MODEL_DIR}/${MODEL_NAME}.ply
```

Replace:
- `${INPUT_DIR}` with the path to your input data directory
- `${MODEL_DIR}` with the path to your model directory
- `${MODEL_NAME}` with the name of your model file

Example:
```bash
gm-refine -s examples/glt/ -m examples/glt.ply
```

The enhanced result will be saved as `${MODEL_DIR}/${MODEL_NAME}.refined.ply`.

### Making the Mesh Watertight

If you only have the mesh file and do not need to specify an input data directory, you can use the `gm-process-mesh` command to make the mesh watertight:

```bash
gm-process-mesh -m ${MODEL_DIR}/${MODEL_NAME}.ply
```

Replace:
- `${MODEL_DIR}` with the path to your model directory
- `${MODEL_NAME}` with the name of your model file

Example:
```bash
gm-process-mesh -m examples/glt.ply
```

The watertight result will be saved as `${MODEL_DIR}/${MODEL_NAME}.refined.ply`, overwriting the previous refined mesh if it exists.

### Cull invisible part

To remove invisible or rarely seen vertices from your model, use the following command:

```bash
gm-cull-mesh -s ${INPUT_DIR} -m ${MODEL_DIR}/${MODEL_NAME}.ply [--min_weights VALUE] [--use_mask]
```

Replace:
- `${INPUT_DIR}` with the path to your input data directory
- `${MODEL_DIR}` with the path to your model directory
- `${MODEL_NAME}` with the name of your model file

Optional parameters:
- `--min_weights VALUE`: Set the minimum number of views a vertex must be visible in (default is 5)
- `--use_mask`: Apply masking during processing

Examples:

1. Culling with minimum weights:
```bash
gm-cull-mesh -s examples/glt/ -m examples/glt.ply --min_weights 1
```

2. Culling with masking:
```bash
gm-cull-mesh -s examples/glt/ -m examples/glt.ply --use_mask
```

3. Culling with both minimum weights and masking:
```bash
gm-cull-mesh -s examples/glt/ -m examples/glt.ply --min_weights 5 --use_mask
```

The culled result will be saved as `${MODEL_DIR}/${MODEL_NAME}.clean.ply`.

## Contribution

Feel free to contribute to GeoMaster by submitting issues or pull requests on our [GitHub repository](https://github.com/hugoycj/GeoMaster).

Happy Modeling with GeoMaster!
