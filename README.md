# GeoMaster: Advanced Geometry Enhancement for High-Resolution 3D Modeling

GeoMaster is a state-of-the-art tool designed to enhance the geometric details of 3D models, providing high-resolution outputs suitable for various applications such as gaming, virtual reality, and 3D printing.

## Installation

To get started with GeoMaster, follow these steps:

### Clone the Repository

First, clone the GeoMaster repository to your local machine:
```bash
git clone https://github.com/hugoycj/GeoMaster.git
cd GeoMaster
```

### Step 1: Install Requirements

Install the necessary Python packages specified in the `requirements.txt` file:
```bash
pip install -r requirements.txt
cd geomaster/utils/libmesh
python setup.py build_ext --inplace
cd ../../../
```

### Step 2: Install PyTorch

Ensure that your PyTorch version is higher than 1.7.1. You can install the specified version using the following command:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Install nvdiffrast

Next, install `nvdiffrast` by cloning its repository and running the installation command:
```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
cd ..
```

### Step 4: Install Gaustudio

Finally, install `gaustudio` from its GitHub repository:
```bash
pip install git+https://github.com/GAP-LAB-CUHK-SZ/gaustudio
```

## Data Preparation

Before running GeoMaster, ensure that your data is organized in the following structure:
```
data_001/
    images/
    mask/
    sparse/
data_002/
    images/
    mask/
    sparse/
data_003/
    images/
    mask/
    sparse/
```
Additionally, you should have a reconstructed coarse mesh available.

## Run GeoMaster
### Using `gm-process`
To enhance the geometric detail of your 3D model using `gm-process`, use the following command:
```bash
gm-process -s ${INPUT_DIR} -m ${MODEL_DIR}/${MODEL_NAME}.ply
```
Replace `${INPUT_DIR}` with the path to your input data directory, `${MODEL_DIR}` with the path to your model directory, and `${MODEL_NAME}` with the name of your model file.

The enhanced result will be saved as `${MODEL_DIR}/${MODEL_NAME}.refined.ply`.

### Using `gm-process-mesh`
If you only have the mesh file and do not need to specify an input data directory, you can use the `gm-process-mesh` command:
```bash
gm-process-mesh -m ${MODEL_DIR}/${MODEL_NAME}.ply
```
Replace `${MODEL_DIR}` with the path to your model directory, and `${MODEL_NAME}` with the name of your model file.

The enhanced result will be saved as `${MODEL_DIR}/${MODEL_NAME}.refined.ply`.

## Example

Here is an example of how to run the pipeline:

### Data Preparation Example
```
examples/
    glt/
        images/
        mask/
        sparse/
    glt.ply
```

### Running GeoMaster
```bash
gm-process -s examples/glt/ -m examples/glt.ply 
# or
gm-process-mesh  -m examples/glt.ply --occ=True
```

After the command executes, the refined model will be available at:
```
examples/glt.refined.ply # watertight mesh
examples/glt.normalized.ply # normalize mesh to [-0.5, 0.5]
examples/glt.normalized.npz # sample points and occupancies
```

## Contribution

Feel free to contribute to GeoMaster by submitting issues or pull requests on our [GitHub repository](https://github.com/hugoycj/GeoMaster).

Happy Modeling with GeoMaster!
