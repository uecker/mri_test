---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="MIGYtH14DwXD" -->
# Introduction to the BART Toolbox

This tutorial introduces the BART command-line inteface (CLI).

**Authors**: [Martin Uecker](mailto:uecker@tugraz.at)$^{\dagger,*,+}$, [Nick Scholand](mailto:scholand@tugraz.at)$^{\dagger,*,+}$, [Moritz Blumenthal](mailto:moritz.blumenthal@med.uni-goettingen.de)$^*$, [Xiaoqing Wang](mailto:xiaoqing.wang@med.uni-goettingen.de)$^{\dagger,*,+}$

**Presenter**: [Martin Uecker](mailto:uecker@tugraz.at)

**Institutions**: $^{\dagger}$University of Technology Graz, $^*$University Medical Center Göttingen, $^+$German Centre for Cardiovascular Research  Partnerside Göttingen
<!-- #endregion -->

<!-- #region id="hBjVg_bNnHzj" -->
## Requirements



<!-- #endregion -->

<!-- #region id="e5RxFHCDnKGt" -->
### Local Usage
- Install bart from its [github repository](https://github.com/mrirecon/bart) (needed for machine learning part!)
- Set the `TOOLBOX_PATH` to the BART directory and add it to the `PATH`

```bash
export TOOLBOX_PATH=/path/to/bart  
export PATH=$TOOLBOX_PATH:$PATH
```

Although the simplest way to call the BART CLI tools is through a terminal, there are also wrapper functions that allow the tools to run through Matlab and Python. These are located under the `$TOOLBOX_PATH/matlab` and `$TOOLBOX_PATH/python` directories.
<!-- #endregion -->

<!-- #region id="f09HrbbxDwXJ" -->
### Online Usage
MyBinder and Google Colaboratory allow us to run a Jupyter instance through a browser. In the following we install and configure BART for both instances.
<!-- #endregion -->

<!-- #region id="MZ-L3VADnvy2" -->
#### Check for GPU

**Google Colaboratory** allows to use a GPU. To access it set:

- Go to Edit → Notebook Settings
- choose GPU from Hardware Accelerator drop-down menu

**MyBinder** does not allow GPU access. The following code will automatically detect which service you are using.
<!-- #endregion -->

```python id="pqkS51t1DwXK"
# Check if notebook runs on colab
import sys, os

os.environ['COLAB'] = 'true' if ('google.colab' in sys.modules) else 'false'
os.environ['CUDA'] = '1' if ('google.colab' in sys.modules) else '0'
```

```bash id="u5ylS2IpDwXM" colab={"base_uri": "https://localhost:8080/"} outputId="e0b7945f-321e-40b7-debe-d9293604aa63"

# Prepare GPUs if on Google Colab

if $COLAB;
then

    # Use CUDA 10.1 when on Tesla K80

    # Estimate GPU Type
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)

    echo "GPU Type:"
    echo $GPU_NAME

    if [ "Tesla K80" = "$GPU_NAME" ];
    then
        echo "GPU type Tesla K80 does not support CUDA 11. Set CUDA to version 10.1."

        # Change default CUDA to version 10.1
        cd /usr/local
        rm cuda
        ln -s cuda-10.1 cuda
    else
        echo "Current GPU supports default CUDA-11."
        echo "No further actions are necessary."
    fi

    echo "GPU Information:"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
    nvcc --version
fi
```

<!-- #region id="leAhIZJ9oWWR" -->
#### Install BART

Here we install BARTs dependencies, we download the current repository from github and compile it.
<!-- #endregion -->

```bash id="KYpxLsEEDwXN"

# Install BARTs dependencies
apt-get install -y make gcc libfftw3-dev liblapacke-dev libpng-dev libopenblas-dev &> /dev/null

# Clone Bart
[ -d /content/bart ] && rm -r /content/bart
git clone https://github.com/mrirecon/bart/ bart &> /dev/null
```

```bash id="tDWAtWn0DwXP" colab={"base_uri": "https://localhost:8080/"} outputId="221632f8-8f4a-4b77-d9f5-2921797e9d80"

# Choose a branch to work on
BRANCH=master

cd bart

# Switch to desired branch of the BART project
git checkout $BRANCH

# Define specifications 
COMPILE_SPECS=" PARALLEL=1
                CUDA=$CUDA
                CUDA_BASE=/usr/local/cuda
                CUDA_LIB=lib64
                OPENBLAS=1
                BLAS_THREADSAFE=1"

printf "%s\n" $COMPILE_SPECS > Makefiles/Makefile.local

# Compile BART
make &> /dev/null
```

<!-- #region id="GV-aQQg8DwXQ" -->
After downloading and compiling BART, we set the required `TOOLBOX_PATH` variable pointing to the BART instance:
<!-- #endregion -->

```python id="39Yu9PaEDwXR" colab={"base_uri": "https://localhost:8080/"} outputId="7fa57042-57ba-439e-f9aa-52f08b43a8f2"
%env TOOLBOX_PATH=bart
```

<!-- #region id="hb0N6uqDDwXR" -->
#### Set Environment for BART

After downloading and compiling BART, the next step simplifies the handling of BARTs command line interface inside of a ipyhton jupyter-notebook. We add the BART directory to the PATH variable and include the python wrapper for reading *.cfl files:
<!-- #endregion -->

```python id="sU6XPDEyDwXU"
os.environ['PATH'] = os.environ['TOOLBOX_PATH'] + ":" + os.environ['PATH']
sys.path.append(os.environ['TOOLBOX_PATH'] + "/python/")
```

<!-- #region id="skXvcEL4DwXW" -->
Check BART setup:
<!-- #endregion -->

```bash id="SQAZz46ODwXW" colab={"base_uri": "https://localhost:8080/"} outputId="8c5de119-d71c-4ca8-e739-fd482ee855e4"

echo "# The BART used in this notebook:"
which bart
echo "# BART version: "
bart version
```

<!-- #region id="-B54--FpFnPB" -->
### Visualization Helper

For this tutorial we will visualize some maps. Therefore, we need a little helper function and some python libraries.

<!-- #endregion -->

```python id="jjk7WRl5FqTh"
# More python libraries
import cfl
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image

def plot_map(dataname, colorbar, vmin, vmax, cbar_label):

    # Import data        
    data = np.abs(cfl.readcfl(dataname).squeeze())

    # Import colorbar type
    colorbartype =  colorbar

    # Set zero to a black color for a masking effect
    my_cmap = cm.get_cmap(colorbartype, 256)
    my_cmap.set_bad('black')

    data = np.ma.masked_equal(data, 0)

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(data, interpolation='nearest', cmap=my_cmap, vmin=vmin, vmax=vmax)

    # Style settings
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params()

    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax1.set_axis_off()

    plt.show()
```

<!-- #region id="SdvGf7xsjhpb" -->
### Download Supporting Materials
For this tutorial, we also need several supporting materials (figures, plotting scripts and compressed data for ML part). They are stored in the GitHub repository and need to be downloaded.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="ouqzdOLhjp1t" outputId="dd840852-f535-4c77-c694-5c1f926e450a"
#FIXME: Combine

# Download the required supporting material if it is not already there
[ ! -f bart_moba.zip ] && wget -q https://github.com/mrirecon/bart-workshop/raw/master/ismrm2021/model_based/bart_moba.zip
unzip -n bart_moba.zip

[ ! -f data_weights.zip ] && wget -q https://github.com/mrirecon/bart-workshop/raw/master/ismrm2021/neural_networks/data_weights.zip
unzip -n data_weights.zip

# Download Raw Data for IR FLASH parts
name=IR-FLASH

if [[ ! -f ${name} ]]; then
  echo Downloading ${name}
  wget -q https://zenodo.org/record/4060287/files/${name}.cfl
  wget -q https://zenodo.org/record/4060287/files/${name}.hdr
fi

# cat md5sum.txt | grep ${name} | md5sum -c --ignore-missing

head -n2 ${name}.hdr
```

<!-- #region id="5Gqo3SJqDwXZ" -->
## BARTs Command Line Tool
BART provides a number of tools for MRI image reconstruction and multi-dimensional array manipulation.
<!-- #endregion -->

```python id="dM-uW8dEDwXZ" colab={"base_uri": "https://localhost:8080/"} outputId="ba3d6db4-715d-4dd2-ee9e-2a9bdcc238c6"
# get a list of BART commands by running bart with no arguments:
! bart
```

<!-- #region id="_zfuPZNkDwXb" -->
### BART Command Structure

The command structure follows

> **`bart`** + **`command`** + **`options`** + **`input / output`**

Each BART command comtains a number of optional arguments, followed by input and output files.   
To see all options and requirements of a command, use the `-h` flag:

<!-- #endregion -->

```python id="YF-P7oMEDwXc" colab={"base_uri": "https://localhost:8080/"} outputId="fbee4e6f-0b5e-4678-80be-9a3e86a82b53"
# Obtain help for each command (-h)
! bart toimg -h
```

<!-- #region id="WM4t453bDwXc" -->
BART loosely follows the Linux CLI convention, where optional arguments are indicated with brackets, and files are indicated by carrot symbols.

As we can see, there are many CLI tools available. The full descriptions of each program can be found under `$TOOLBOX_PATH/doc/commands.txt`.
<!-- #endregion -->

<!-- #region id="3oXo4iuEDwXc" -->
### Using BART
 
As a first example, let's create a Shepp-Logan numerical phantom in k-space. We will use the `phantom` tool. Before using the tool, let's look at its options
<!-- #endregion -->

```python id="mdwlVvNMDwXe" colab={"base_uri": "https://localhost:8080/"} outputId="d0afa983-0938-4578-952a-391b256213bf"
! bart phantom -h
```

<!-- #region id="l65K0lkLDwXe" -->
The `phantom` tool includes the option `-k` to create it directly in k-space, and `-x` to specify the dimensions.
<!-- #endregion -->

```python id="G4IikpLXDwXe" colab={"base_uri": "https://localhost:8080/", "height": 275} outputId="bc41079f-0ff2-4ae1-e03a-391088acf06e"
# Create Cartesian k-space phantom (256 samples)
! bart phantom -x 256 -k shepp_logan

! echo "Shepp-Logan k-space phantom"
plot_map("shepp_logan", "viridis", 0, 0.02, '')
```

<!-- #region id="WJzGVMeuDwXe" -->
### Data File Format
All data files are multi-dimensional arrays. By running the `phantom` command, we made new files on disk, with the names  
`shepp_logan.cfl` and `shepp_logan.hdr`

The data files represent a multi-dimensional array. The `hdr`, or header file, contains the data dimensions and other information about the dataset. The `cfl`, or complex-float file, contains the single-precision raw data in column-major order and interleaved real and imaginary parts.  

The header file is a raw text file and can be opened with any text editor. The complex-float file is a binary file. Matlab and Python readers/writers are available under the `matlab` and `python` directories, respectively.

### View data dimensions
Because the header file is a text file, we can directly print it:
<!-- #endregion -->

```python id="DWONQd_zDwXf" colab={"base_uri": "https://localhost:8080/"} outputId="adf9f78d-465c-4fa8-e87d-739fcdee0bed"
! cat shepp_logan.hdr
```

<!-- #region id="741w3qC8DwXf" -->
Although not discussed here, BART can write to other file formats, including a simple H5 container, ISMRMRD format, and others. Therefore, not every format will include a header file. Thus, it is better to use the `show` command.
<!-- #endregion -->

```python id="cx6g7E2sDwXg" colab={"base_uri": "https://localhost:8080/"} outputId="abf89506-570d-4820-ae93-01d5cb0c02db"
! bart show -h
```

<!-- #region id="zGNvNGn4DwXg" -->
We can look at an individual dimension of the data with the `-d` command, or we can display all the meta data about the dataset. 

Next, we show the dimensions of the created Shepp-Logan phantom
<!-- #endregion -->

```python id="GQFw_AUGDwXh" colab={"base_uri": "https://localhost:8080/"} outputId="51d18286-5fdb-408c-e863-4f3bcd0dd596"
! bart show -m shepp_logan
```

<!-- #region id="jzz1q1nUDwXh" -->
Our dataset is 16-dimensional, but only the first two dimensions are non-singleton.

By convention, the dimensions are `[X, Y, Z, C, M, T, F, ...]`,
where `(X, Y, Z)` are the spatial matrix dimensions,  
`C` and `M` are the coil dimensions and ESPIRiT maps dimensions, respectively,  
`T` and `F` are used for echo times and coefficient dimensions,   
and other higher order dimensions such as phase, flow encode, etc.
<!-- #endregion -->

<!-- #region id="rkEKZc2wDwXh" -->
### Using Bitmasks
Let's reconstruct our k-space phantom. using a simple inverse Fourier transform.  
Therefore, we perform a Fast Fourier Transform (FFT).

BART has an `fft` tool for doing just that.
<!-- #endregion -->

```python id="7IinGE42DwXi" colab={"base_uri": "https://localhost:8080/", "height": 275} outputId="1876b694-0836-4490-94d8-b314a7b4e55e"
# Perform FFT reconstruction
! bart fft -u -i 3 shepp_logan shepp_logan_rec

! echo "IFFT of Shepp-Logan phantom"
plot_map("shepp_logan_rec", "viridis", 0, 0.005, '')
```

<!-- #region id="wy0HmM5gDwXj" -->
Let's see the fft-options.
<!-- #endregion -->

```python id="R7S8Vs6QDwXk" colab={"base_uri": "https://localhost:8080/"} outputId="7435409a-9a6d-4fdd-d008-ca61c319303c"
# Show help for fft command
! bart fft -h
```

<!-- #region id="E8Ja88-DDwXk" -->
Thus, we performed an inverse (`-i`) unitary (`-u`) Fast Fourier Transform on the image dimensions **`(0, 1)`** specified by the bitmask **`3`**.

<!-- #endregion -->

<!-- #region id="Q66iAkUMDwXk" -->
Instead of using for loops, BART data operates on bitmasks. To operate on a particular array dimension, a bitmask specifies the active dimensions. This is a powerful approach for perfoming multi-dimensional operations, as all the tools will work on arbitrarily chosen dimensions.   

In our case, we wanted to perform an IFFT along dimensions 0 and 1, and the bitmask is calculated as:  
<center>
$ \text{bitmask}=2^{~0} + 2^{~1} = 3$
</center> <br>
BART also provides a command-line tool to calculate the bitmasks for specific dimensions.
<!-- #endregion -->

```python id="w_Lgk8REDwXl" colab={"base_uri": "https://localhost:8080/"} outputId="c795592b-810d-4e6c-d8e9-64c9eff0f8d9"
# Calculate bitmask for active dimensions 0 and 1
! bart bitmask 0 1
```

<!-- #region id="tIKCcmsTDwXl" -->
## BART Examples

<!-- #endregion -->

<!-- #region id="ePpy5MiNtR7d" -->
### Subspace T1 Mapping

A specialized tutorial for subspace T1 mapping with BART can be found in the [3rd BART Webinar Materials](https://github.com/mrirecon/bart-webinars/tree/master/webinar3).
<!-- #endregion -->

<!-- #region id="yz7rE-BLKwAd" -->
We start by importing the characteristics of the downloaded IR-FLASH dataset following BARTs dimensionality definitions.
<!-- #endregion -->

```python id="HUvJkSd53Cg0"
dim = np.shape(cfl.readcfl("IR-FLASH"))

os.environ['READ'] = str(dim[0])
os.environ['SPOKES'] = str(dim[2])
os.environ['COILS'] = str(dim[3])
os.environ['REP'] = str(dim[10])
```

<!-- #region id="jxHx9Rkw4lJs" -->
Set the known sequence information.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="2DxodtdF4qRI" outputId="eb3526e1-6a2c-41c4-9e9c-5074b63aee4f"
%env TR=0.0041
```

<!-- #region id="xUbjrToOuTd7" -->
#### Dictionary Generation, SVD and Temporal Basis

Set the number of temporal subspace coefficients.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="suY_W6s25EqY" outputId="dc4a3e16-599f-45a0-c68c-e94f34612ede"
%env NUM_COE=4
```

<!-- #region id="KbFrSapm5Jmg" -->
Now we have everything to generate the dictionary
<!-- #endregion -->

```bash id="6fDErOkBuOrv"


# Dictionary characteristics
## R1s
NUM_R1S=1000
MIN_R1S=5e-3
MAX_R1S=5

## Mss
NUM_MSS=100
MIN_MSS=1e-2
MAX_MSS=1

# Simulate dictionary based on the `signal` tool
bart signal -F -I -n$REP -r$TR \
            -1 $MIN_R1S:$MAX_R1S:$NUM_R1S \
            -3 $MIN_MSS:$MAX_MSS:$NUM_MSS  dicc

# reshape the dicc 6th and 7th dimension to have all the elements 
# concentrated in the 6th dimension
bart reshape $(bart bitmask 6 7) $((NUM_R1S * NUM_MSS)) 1 dicc dicc_reshape

# squeeze the reshaped dictionary to remove non-zero dimensions
bart squeeze dicc_reshape dicc_squeeze
```

<!-- #region id="SGYd3Hvo5WCp" -->
And perform an svd to create our temporal basis
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="GsBB2Hv05VYl" outputId="a42c1151-74b9-4488-9f99-927507027b0b"

# Perform an SVD of the squeezed dictionary and output and
# economy-size decomposition of the resulting matrix
bart svd -e dicc_squeeze U S V

# Extract desired number of orthonormal columns from U
bart extract 1 0 $NUM_COE U basis0

# transpose the basis to have time in the 5th dimension 
# and coefficients in the 6th dimension
bart transpose 1 6 basis0 basis1
bart transpose 0 5 basis1 basis

# Print the transposed basis dimensions
echo "Temporal Basis"
head -n2 basis.hdr
```

<!-- #region id="KkLHKkmhukOC" -->
#### Coil Compression
To reduce the size of our dataset and therefore also decrease the computational complexity, we perform a coil compression with the `cc` command. By passing `-A` we choose to use all possible data and want to reduce the dataset to *NUM_VCOILS* virtual coils with `-p` to

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="kgkeo0kJ5_P3" outputId="f3444c97-37c9-4d21-db18-442324691944"
%env NUM_VCOILS=8
```

```bash colab={"base_uri": "https://localhost:8080/"} id="H7m5VK89uqqn" outputId="0f54947c-edd2-4b57-d554-ea3325d5f025"

# Transpose the 0th and 1st dimension of the downloaded data
# to ensure compatibility with BARTs non-Cartesian tools
bart transpose 0 1 IR-FLASH ksp

# Perform coil compression
bart cc -A -p $NUM_VCOILS ksp ksp_cc
```

<!-- #region id="_JUtkpkCu2XL" -->
#### Trajectory Generation
In the next step we generate a trajectory with the `traj` tool. To match the acquisition of the downloaded data, we need to specify a radial `-r`, centered `-c`, double-angle `-D`, 7th tiny golden-angle `-G -s7` sampling. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="CcXqraR32pDl" outputId="ec1e1cdb-03ee-4528-d635-c2c2ad864c2f"
%env NUM_TGA=7
```

<!-- #region id="wuWVblmT2oOE" -->
The timesteps are passed using `-t`, the spokes by `-y` and the samples are specified with `-x`.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="C5LkG-ZPvFl0" outputId="74b3a174-5148-4f3d-cff4-4c94e60e5ba4"

# Read file characteristics from downloaded dataset
READ=`bart show -d 0 IR-FLASH`
SPOKES=`bart show -d 2 IR-FLASH`
REP=`bart show -d 10 IR-FLASH`

# Create the trajectory using the `traj` tool
bart traj -r -c -D -G -x$READ -y$SPOKES -s$NUM_TGA -t$REP traj

# Print out its dimensions
echo "Trajectory"
head -n2 traj.hdr
```

<!-- #region id="5nwwcslAvcPs" -->
Here the 3 in the zeroth dimensions includes the coordinates in (x,y,z) and is the reasion for the previous transpose of the downloaded dataset!
<!-- #endregion -->

<!-- #region id="sp04vChGvejm" -->
#### Gradient Delay Correction
Because the dataset is following an IR FLASH signal relaxation, the gradient delay correction should be applied to data in the last repetitions which are in a steady-state. Therefore, we extract some final repetitions from the trajectory and the dataset using the `extract` command. Have in mind that the time dimension is the 10th here!
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="eiCjJxmzvsPl" outputId="9eb0b2e0-4957-4211-dee1-8227915d62aa"

# Define the number of timesteps the gradient delay correction should be 
# performed on (Steady-State)
FRAMES_FOR_GDC=100

# Extract the DATA_GDC last time frames from the trajectory and the dataset
bart extract 10 $((REP-FRAMES_FOR_GDC)) $REP traj traj_extract
bart extract 10 $((REP-FRAMES_FOR_GDC)) $REP ksp_cc ksp_extract

# Transpose the 2nd and 10th dimension for later use with the `estdelay` tool
bart transpose 10 2 traj_extract traj_extract1
bart transpose 10 2 ksp_extract ksp_extract1

# Estimate and store the gradient delays usign RING
GDELAY=$(bart estdelay -R traj_extract1 ksp_extract1)

echo "Gradient Delays: "$GDELAY

# Calculate the trajectory with known gradient delays
bart traj -r -c -D -G -x$READ -y$SPOKES -s$NUM_TGA -t$REP -q $GDELAY trajn
```

<!-- #region id="vMR9HvyLv7RN" -->
#### Coil Sensitivity Estimation

The coil profile estimation is similar to the gradient delay estimation performed on some of the last timesteps of the IR FLASH dataset. Therefore, *FRAMES_FOR_CSE* spokes from the last timesteps are extracted using the `extract` command.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="Rw2nBqkev-dV" outputId="a3cb1f4e-bc4d-407c-c993-ece0f99196a8"

# Define the number of timesteps at th end of the dataset, where the coil 
# sensitivity estimation should be performed on (Steady-State)
FRAMES_FOR_CSE=300

# Extract last time frames (10th dim) from trajectory and k-space data
bart extract 10 $((REP-FRAMES_FOR_CSE)) $REP trajn traj_ss
bart extract 10 $((REP-FRAMES_FOR_CSE)) $REP ksp_cc ksp_cc_ss

bart transpose 2 10 traj_ss traj_ss2
bart transpose 2 10 ksp_cc_ss ksp_cc_ss2

# apply an inverse nufft of the extracted steady-state dataset
bart nufft -i -d$READ:$READ:1 traj_ss2 ksp_cc_ss2 img

# transform reconstruction in image space back to k-space
# to create gridded k-space for the ESPIRiT implementation
bart fft -u $(bart bitmask 0 1 2) img ksp_grid

THRESHOLD=0.01
NUM_ESPIRIT_MAP=1

# Estimate coil sensitivities from gridded, steady-state k-space using `ecalib`
bart ecalib -S -t $THRESHOLD -m $NUM_ESPIRIT_MAP ksp_grid sens_invivo

```

<!-- #region id="3AxOykoMwTps" -->
#### Subspace-Constrained Reconstruction
To start the subspace-constrained reconstruction we need to verify the data and trajectory dimension again.
<!-- #endregion -->

```bash colab={"base_uri": "https://localhost:8080/"} id="N8zOL3vpwZOy" outputId="aa219f72-e079-4ef5-9ba6-aad75f298446"

# Transpose dimensions for working with PICS tool
bart transpose 5 10 trajn traj_final
bart transpose 5 10 ksp_cc ksp_final

DEBUG=4
ITER=100
REG=0.0015

[ $CUDA ] && GPU=-g

bart pics   $GPU -e -d $DEBUG -i$ITER \
            -RW:$(bart bitmask 0 1):$(bart bitmask 6):$REG \
            -t traj_final -B basis \
            ksp_final sens_invivo subspace_reco_invivo

# Print dimensions of reconstruction
echo "Reconstructed Coefficient"
head -n2 subspace_reco_invivo.hdr

# Resize reconstructions to remove 2-fold oversampling effects
bart resize -c 0 $((READ/2)) 1 $((READ/2)) subspace_reco_invivo coeff_maps
```

<!-- #region id="y_rpDr3kwnRL" -->
#### Visualization of Reconstructed Maps
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 256} id="mcU_b2E8wstT" outputId="646e551e-049e-495f-ead2-58c9ebd64898"
# Reshape and flip coefficient maps for improved visualization

## Concentrate all coefficients in the column dimension (1st/phase1)
! bart reshape $(bart bitmask 1 6) $((READ/2*NUM_COE)) 1 coeff_maps subspace_maps

## Flip the map in row dimension to have the forhead pointing to the top of the page
! bart flip $(bart bitmask 0) subspace_maps subspace_maps1

! echo "Subspace Coefficient Maps"
plot_map("subspace_maps1", "viridis", 0, 1, '')
```

<!-- #region id="_-JV1PCDpbvW" -->
### Model-Based T1 Mapping

A specialized tutorial for model-based reconstructions in BART can be found in the [Workshop Material of the ISMRM 2021](https://github.com/mrirecon/bart-workshop/tree/master/ismrm2021).
<!-- #endregion -->

<!-- #region id="sDOZbCGHDwXm" -->
#### Theory

**Single-Shot Inversion-Prepared T1 Mapping**

<img src="https://github.com/mrirecon/bart-workshop/blob/master/ismrm2021/model_based/IR_FLASH.png?raw=1" style="width: 550px;">


**General Idea of Model-based Reconstruction**:
> Formulate the estimation of MR physical parameters directly from k-space as a nonlinear inverse problem.




<!-- #endregion -->

<!-- #region id="BC-8Qb7uqG8-" -->
**Operator chain of parallel imaging and signal model (nonlinear)**

<img src="https://github.com/mrirecon/bart-workshop/blob/master/ismrm2021/model_based/operator_chain.png?raw=1" style="width: 400px;">

$$F: x \mapsto y = {\mathcal{P} \mathcal{F} C} \cdot {M(x_{p})}$$
- $\mathcal{P}$ - sampling pattern
- $\mathcal{F}$ - Fourier transform
- $C$ - coil sensitivity maps
- $M(\cdot)$ - MR physics model
- $x_{p}$ - MR parameters
- $y$ - acquired kspace data
<!-- #endregion -->

<!-- #region id="SaIAOQw4p5LI" -->
#### Optimization

We use the iteratively regularized Gauss-Newton method (IRGNM) in BART to solve the nonlinear inverse problem

$$\hat{x} = \text{argmin}_{x}\|F(x) -y\|_{2}^{2} + \lambda \cdot R(x), $$

i.e., the nonlinear problem can be linearly solved in each Gauss-Newton step:

$$\hat{x}_{n+1}= \text{argmin}_{x}\|DF(x_{n})(x−x_{n}) +F(x_{n})−y\|_{2}^{2} + \lambda \cdot R(x)$$

$DF(x_{n})$ is the Jacobian matrix of $F$ at the point $x_{n}$ of the $n$th Newton step.
<!-- #endregion -->

<!-- #region id="eHckyTZPp3Tk" -->
---

Therefore, we can directly estimate the MR parameter maps from undersampled k-space datasets. No pixel-wise fitting or intermediate reconstruction of contrast-weighted images is required!

For further information have a look into:

[Wang X](mailto:xiaoqing.wang@med.uni-goettingen.de), Roeloffs V, Klosowski J, Tan Z, Voit D, Uecker M, Frahm J.,  
[Model-based T1 Mapping with Sparsity Constraints Using Single-Shot Inversion-Recovery Radial FLASH](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26726).  
Magn Reson Med 2018;79:730-740.
<!-- #endregion -->

```python id="OvICvnDSDwXn" colab={"base_uri": "https://localhost:8080/"} outputId="cc2ba67b-9ec1-436d-882b-ba491772caa6"
! bart moba -h
```

<!-- #region id="oZcwQxOfDwXn" -->
The data preparation is dicussed in detail in the [3rd event of the BART webinar series](https://github.com/mrirecon/bart-webinars/tree/master/webinar3). Here you can find presentation and exercise materials for the individual steps

* Download raw data  
* Coil compression  
* Prepare radial trajectory (golden-angle) including gradient-delay correction  
* Prepare time vector

which are not mentioned in detail here.


<!-- #endregion -->

<!-- #region id="jdVE624Z_zMD" -->
#### Dimensions

The dataset is the used inthe subspace tutorial above. Therefore, the global variables (READ, SPOKES,...) defining its dimensions do not have to be changed.
<!-- #endregion -->

<!-- #region id="ejsKnyBDDwXo" -->
#### Coil Compression

We will compress our dataset to
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="0_J8LSgf_h1G" outputId="29787df3-3950-4241-cb52-84d210a8c3c7"
%env NUM_VCOILS=3
```

<!-- #region id="vqnAReGt_lNs" -->
coils. This is for the purpose of fast computation, for a better performance, 8 or 10 are recommended.
<!-- #endregion -->

```bash id="PeTLm8ZADwXo" colab={"base_uri": "https://localhost:8080/"} outputId="a180198f-1ffa-413b-dc67-e7a54991bfb0"

## Coil compression
bart transpose 0 1 IR-FLASH ksp

# coil compression
bart cc -A -p $NUM_VCOILS ksp ksp_cc
```

<!-- #region id="2rqNkt2hDwXp" -->
#### Trajectory Preparation with Gradient Delay Correction
<!-- #endregion -->

```bash id="nQ8GLGjuDwXp" colab={"base_uri": "https://localhost:8080/"} outputId="648bb7fb-ea8d-4039-ce09-02ce908c5bbf"

## Prepare radial trajectory (golden-angle)
bart traj -r -c -D -G -x$READ -y$SPOKES -s7 -t$REP traj

# Gradient Delay Correction
# Extract the steady-state data (data with less contrast change)
bart extract 10 920 1020 traj traj_extract
bart extract 10 920 1020 ksp_cc ksp_extract

# Transpose the 2nd and 10th dimension for the `estdelay` tool
bart transpose 10 2 traj_extract traj_extract1
bart transpose 10 2 ksp_extract ksp_extract1

# Gradient delay estimation usign RING
GDELAY=$(bart estdelay -R traj_extract1 ksp_extract1)

echo "Gradient Delays: "$GDELAY

# Calculate the "correct" trajectory with known gradient delays
bart traj -r -c -D -G -x$READ -y$SPOKES -s7 -t$REP -q $GDELAY trajn
```

<!-- #region id="S_N-Vq7sDwXq" -->
#### Preparation of Time Vector
<!-- #endregion -->

```bash id="vuS4vYCJDwXq" colab={"base_uri": "https://localhost:8080/"} outputId="fcae9c32-f58c-438c-9014-b202b0ec555b"

## Prepare time vector
TR=4100 #TR in [us]
BIN_SPOKES=20 # Bin data to save computational time 

NTIME=$((REP/BIN_SPOKES)) # Integer division!

# Create vector from 0 to NTIME
bart index 5 $NTIME tmp1
# use local index from newer bart with older bart
bart scale $(($BIN_SPOKES * $TR)) tmp1 tmp2
bart ones 6 1 1 1 1 1 $NTIME tmp1 
bart saxpy $((($BIN_SPOKES / 2) * $TR)) tmp1 tmp2 tmp3
bart scale 0.000001 tmp3 TI

# Reshape trajectory and data for model-based reconstruction
bart reshape $(bart bitmask 2 5 10) $BIN_SPOKES $NTIME 1 trajn traj_moba
bart reshape $(bart bitmask 2 5 10) $BIN_SPOKES $NTIME 1 ksp_cc ksp_cc_moba

# Resize data and trajectory for faster computation
bart resize -c 1 384 traj_moba traj_moba1
bart resize -c 1 384 ksp_cc_moba ksp_cc_moba1

echo "Trajectory:"
head -n2 traj_moba1.hdr

echo "Data:"
head -n2 ksp_cc_moba1.hdr

echo "TI:"
head -n2 TI.hdr''
```

<!-- #region id="dP5KouD8DwXs" -->
#### Nonlinear Model-based Reconstruction

The full nonlinear reconstruction can be applied to data by using only the `moba` command in the BART CLI. No coil sensitivity information is necessary, because they are jointly estimated.

A detailed description of the reconstruction can be also found in the [3rd event of the BART webinar series](https://github.com/mrirecon/bart-webinars/tree/master/webinar3).

Here we apply a non-linear inversion-recovery Look-Locker model `-L` to our single-shot data. We also exploit compressed sensing by adding a wavelet $l_1$ regularization with the `-l1` flag.
<!-- #endregion -->

```bash id="L2qNDbGaDwXs" colab={"base_uri": "https://localhost:8080/"} outputId="9536fbf6-39ee-457e-9782-a4d25ad02acd"

[ $CUDA ] && GPU=-g

bart moba -L $GPU -d4 -l1 -i8 -C100 -j0.09 -B0.0 -n -t traj_moba1 ksp_cc_moba1 TI reco_moba 

#-L  --- to select look-locker model
#-g  --- to use GPU
#-d  --- debug level
#-l1 --- to use l1-Wavelet regularization
#-i  --- number of Newton-steps
#-C  --- maximum number of inner iterations (FISTA)
#-j  --- minimum regularization parameter
#-B  --- lower bound for relaxivity (R1s > 0)

# NOTE: There is no need of input of coil sensitivity maps, because we jointly estimate coils using model-based reconstruction
```

<!-- #region id="i-KXXZYZDwXs" -->
#### Visualize Results

To visualize the output of the reconstruction we resize it and thus remove the applied oversampling. Additionally, we slice the individual maps out of its original file and place them next to each other for the final visualization.
<!-- #endregion -->

```python id="-XqJxYsEDwXt" colab={"base_uri": "https://localhost:8080/", "height": 322} outputId="9802b3c3-17b9-4af1-dd87-eae1d7ba8aa6"
# Remove oversampling on maps Mss, M0, R1s
! bart resize -c 0 $((READ/2)) 1 $((READ/2)) reco_moba reco_maps

# Concentrate all coefficients in the column dimension for visualization
! bart reshape $(bart bitmask 1 6) $((3*READ/2)) 1 reco_maps reco_maps_lin

# Flip the map in row dimension to have the forhead pointing to the top of the page
! bart flip $(bart bitmask 0) reco_maps_lin reco_maps_flip

! echo "Reconstructed Coefficient Maps: Mss, M0, R1s"
plot_map("reco_maps_flip", "viridis", 0, 3, '')
```

<!-- #region id="ygM6OQGHDwXt" -->
The output of the nonlinear Look-Locker model-based reconstruction are the parameter maps Mss, M0 and R1*.  
To estimate the desired T1 map we pass the reconstruction results to the `looklocker` command and visualize the T1 map here.
<!-- #endregion -->

```bash id="va_-giUVDwXu"

INVERSION_DELAV=0.0153
THRESHOLD=0.2


# use the "looklocker" tool in BART for estimating T1 from the parameters 
# Mss, M0, R1s
bart looklocker -t $THRESHOLD -D $INVERSION_DELAV reco_maps tmp

# Flip the map in row dimension to have the forhead pointing to the top of the page
bart flip $(bart bitmask 0) tmp moba_T1map
```

```python colab={"base_uri": "https://localhost:8080/", "height": 835} id="gDDc8HSOD8FQ" outputId="8fe7bd87-544d-4974-92fb-778ffbeb560f"
# python3 save_maps.py moba_T1map viridis 0 2.0 moba_T1map.png
plot_map("moba_T1map", "viridis", 0, 2, '$T_1$ / s')
```

<!-- #region id="xA8ldOwuDwXv" -->
### BART for Machine Learning - Reconstruction Networks

A specialized tutorial for neural networks in BART can be found in the [Workshop Material of the ISMRM 2021](https://github.com/mrirecon/bart-workshop/tree/master/ismrm2021).
<!-- #endregion -->

<!-- #region id="9GUPKERGrFnr" -->
#### Theory
We have implemented

> Variational Network<sup>1</sup>:
$$
x^{(i)} = x^{(i-1)}  - \lambda \nabla||Ax -b||^2 + Net(x^{(i-1)}, \Theta^{(i)} )
$$
> MoDL<sup>2</sup>:
$$
\begin{align}
z^{(i)} &= Net\left(x^{(i-1)}, \Theta \right)\\
x^{(i)} &= \mathrm{argmin}_x ||Ax -b||^2 + \lambda ||x - z^{(i)}||^2
\end{align}
$$

>Where
+ $A$ - MRI forward operator $\mathcal{PFC}$
    + $\mathcal{P}$ - Sampling pattern
    + $\mathcal{F}$ - Fourier transform
    + $\mathcal{C}$ - Coil sensitivity maps
+ $b$ - measured k-space data
+ $x^{(i)}$ - reconstruction after $i$ iterations
+ $x^{(0)}=A^Hb$ - initialization
+ $\Theta$ - Weights

>1: Hammernik, K. et al. (2018), [Learning a variational network for reconstruction of accelerated MRI data](https://doi.org/10.1002/mrm.26977). Magn. Reson. Med., 79: 3055-3071.

>2: Aggarwal, H. K. et al.(2019), [MoDL: Model-Based Deep Learning Architecture for Inverse Problems](https://doi.org/10.1109/TMI.2018.2865356). IEEE Trans. Med. Imag., 38(2): 394-405

To **train**, **evaluate** or **apply** unrolled networks, we provide the `bart reconet` command. It follows the same logic as the `bart nnet` command but gets the coil sensitivity maps as an additional input. Let us look at the help:
<!-- #endregion -->

```python id="gtzNi5vpDwXv" colab={"base_uri": "https://localhost:8080/"} outputId="15948ab5-613c-42a3-ffa9-fc36bf8da3d1"
! bart reconet -h
```

```python id="h2T5krOSDwXw" colab={"base_uri": "https://localhost:8080/"} outputId="3334c1fe-ea5e-4b2b-b4c1-ad505a4f8829"
! bart reconet --network h
```

<!-- #region id="dXvxFq6lDwXw" -->
#### Preparation of Knee-Data

Here, we use the data provided with the publication of the Variational Network, i.e. the coronal_pd_fs folder of the NYU-Dataset. The data has been converted to the .cfl-file format.  
In the data folder, we find the fully-sampled kspace data of a knee and a sampling pattern. As the kspace is fully sampled, we can define a ground truth reference.

Before we apply the networks, we will create/estimate:
+ the downsampled kspace
+ coil sensitivity maps
+ a ground truth reference
<!-- #endregion -->

```python id="7HvKWTioDwXw" colab={"base_uri": "https://localhost:8080/", "height": 408} outputId="cfb8799e-80d1-4dcc-99ab-76edef2cfcfe"
! echo $'\n# K-Space (fully sampled):'
! head -n2 data/kspace_fs.hdr

! echo $'\n# Pattern:'
! head -n2 data/pattern_po_4.hdr

pattern = np.abs(cfl.readcfl("data/pattern_po_4"))
plt.imshow(pattern, cmap="gray")
plt.show()
```

<!-- #region id="__DLJ4XLDwXx" -->
#### Create Downsampled Kspace

We downsample the fully-sampled kspace by multiplying it with the sampling pattern:
<!-- #endregion -->

```python id="5-6QuQMsDwXx"
! bart fmac data/kspace_fs data/pattern_po_4 kspace
```

<!-- #region id="rcv3PD0lDwXy" -->
#### Estimate Coil Sensitivity Maps

We estimate the coil sensitivity maps using ESPIRiT. 
<!-- #endregion -->

```python id="dWOcX0FUDwXy" colab={"base_uri": "https://localhost:8080/"} outputId="91f2107d-1d8e-468c-ea97-a7385b7c7870"
! bart ecalib -r24 -m1 kspace coils_l
! bart resize -c 0 320 coils_l coils
```

<!-- #region id="dbWg5HJWDwXy" -->
#### Reconstruction of the Reference

We construct the **ground truth reference** as the coil-combinded reconstruction of the fully-sampled kspace data. For comparison, we also compute a **l1-wavelet** regularized and the **zero-filled** reconstruction.
<!-- #endregion -->

```bash id="g5l3sB0UDwXz" colab={"base_uri": "https://localhost:8080/"} outputId="54ac33f1-d57c-4dc0-be39-56db6075d250"

mkdir -p tmp

FFT_FLAG=$(bart bitmask 0 1)
COIL_FLAG=$(bart bitmask 3)

# Reference
bart fft -i -u $FFT_FLAG data/kspace_fs tmp/coil_image
bart fmac -C -s$COIL_FLAG tmp/coil_image coils_l tmp/image

# PICS l1
bart pics -S -l1 -r0.001 -pdata/pattern_po_4 kspace coils_l tmp/pics_reco_l
#resize (frequency oversampling)

# Zero-filled
bart fft -i -u $FFT_FLAG kspace tmp/coil_image_zf
bart fmac -C -s$COIL_FLAG tmp/coil_image_zf coils_l tmp/image_zf_l

#resize (frequency oversampling)
bart resize -c 0 320 tmp/image ref
bart resize -c 0 320 tmp/pics_reco_l pics_reco
bart resize -c 0 320 tmp/image_zf_l zero_filled

rm -r tmp
```

<!-- #region id="0ciAHjCrDwX0" -->
We show the results:
<!-- #endregion -->

```python id="Pt06m9cUDwX0" colab={"base_uri": "https://localhost:8080/", "height": 464} outputId="881a16eb-c4e1-4fc3-db7b-edec0cff71b4"
ref = cfl.readcfl("ref")
pics_reco = cfl.readcfl("pics_reco")
zero_filled = cfl.readcfl("zero_filled")

vmax=0.9*np.max(np.abs(ref))

fig, axes = plt.subplots(figsize=(20,6), nrows=1, ncols=3, sharex=True, sharey=True)

axes[0].imshow(np.abs(ref[::-1,::-1]), cmap="gray", vmax=vmax)
axes[0].set_title("Coil Combined Reference", fontsize=20)

axes[1].imshow(np.abs(pics_reco[::-1,::-1]), cmap="gray", vmax=vmax)
axes[1].set_title("l1-Wavelet Regularized", fontsize=20)

axes[2].imshow(np.abs(zero_filled[::-1,::-1]), cmap="gray", vmax=vmax)
axes[2].set_title("Zero-filled Reconstruction", fontsize=20)

plt.tight_layout()
plt.show()
```

<!-- #region id="7Dv1UsVCDwX0" -->
#### Apply Variational Network

Having prepared the dataset, we can apply the Variational Network using the downloaded weights. The dataset is normalized by the maximum magnitude of the zero-filled reconstruction by using the `--normalize` option.  
We use the pretrained weights provided in the weights directory. They have been trained on the first 15 knees from the coronal_pd_fs directory of the NYU-Dataset
<!-- #endregion -->

```bash id="xVg4RtKoDwX1" colab={"base_uri": "https://localhost:8080/"} outputId="9cb99c19-a6f1-40a6-9d39-732a86c907a2"

[ $CUDA ] && GPU=--gpu; # if BART is compiled with gpu support, we add the --gpu option

bart reconet \
    $GPU \
    --network=varnet \
    --normalize \
    --apply \
    --pattern=data/pattern_po_4 \
    kspace \
    coils \
    weights/varnet \
    varnet
```

<!-- #region id="SsP12TvLDwX1" -->
We plot the results:
<!-- #endregion -->

```python id="qXfmNgJADwX1" colab={"base_uri": "https://localhost:8080/", "height": 464} outputId="7f27ea83-d7da-45dd-ccae-b9eadf1a6b01"
ref = cfl.readcfl("ref")
pics_reco = cfl.readcfl("pics_reco")
varnet = cfl.readcfl("varnet")

vmax=0.9*np.max(np.abs(ref))

fig, axes = plt.subplots(figsize=(20,6), nrows=1, ncols=3, sharex=True, sharey=True)

axes[0].imshow(np.abs(ref[::-1,::-1]), cmap="gray", vmax=vmax)
axes[0].set_title("Coil Combined Reference", fontsize=20)

axes[1].imshow(np.abs(pics_reco[::-1,::-1]), cmap="gray", vmax=vmax)
axes[1].set_title("l1-Wavelet Regularized", fontsize=20)

axes[2].imshow(np.abs(varnet[::-1,::-1]), cmap="gray", vmax=vmax)
axes[2].set_title("Variational Network", fontsize=20)

plt.tight_layout()
plt.show()
```

<!-- #region id="zXkSd322DwX2" -->
#### Apply MoDL

Similarly, MoDL can be applied using the provided weights. Here, we unroll 5 iterations.
<!-- #endregion -->

```bash id="ogSm6QC8DwX2" colab={"base_uri": "https://localhost:8080/"} outputId="c4014778-c2bf-4648-bb3e-775c4fdcc718"

[ $CUDA ] && GPU=--gpu; # if BART is compiled with gpu support, we add the --gpu option

bart reconet \
    $GPU \
    --network=modl \
    --iterations=5 \
    --normalize \
    --apply \
    --pattern=data/pattern_po_4 \
    kspace \
    coils \
    weights/modl \
    modl
```

<!-- #region id="RipDR2vQDwX2" -->
We plot the results:
<!-- #endregion -->

```python id="8Yyfi3DEDwX2" colab={"base_uri": "https://localhost:8080/", "height": 464} outputId="4d72e0ed-3543-48ef-eeb5-523dfe2501b1"
ref = cfl.readcfl("ref")
pics_reco = cfl.readcfl("pics_reco")
modl = cfl.readcfl("modl")

vmax=0.9*np.max(np.abs(ref))

fig, axes = plt.subplots(figsize=(20,6), nrows=1, ncols=3, sharex=True, sharey=True)

axes[0].imshow(np.abs(ref[::-1,::-1]), cmap="gray", vmax=vmax)
axes[0].set_title("Coil Combined Reference", fontsize=20)

axes[1].imshow(np.abs(pics_reco[::-1,::-1]), cmap="gray", vmax=vmax)
axes[1].set_title("l1-Wavelet Regularized", fontsize=20)

axes[2].imshow(np.abs(modl[::-1,::-1]), cmap="gray", vmax=vmax)
axes[2].set_title("MoDL", fontsize=20)

plt.tight_layout()
plt.show()
```

<!-- #region id="ncBxyjH9DwX2" -->
#### Evaluation of the Variational Network and MoDL
<!-- #endregion -->

```bash id="81wG5LfgDwX3" colab={"base_uri": "https://localhost:8080/"} outputId="d500337f-959c-4d55-fbfb-9e9502c51445"

# if BART is compiled with gpu support, we add the --gpu option
[ $CUDA ] && GPU=--gpu;

bart reconet \
    $GPU \
    --network=varnet \
    --normalize \
    --eval \
    --pattern=data/pattern_po_4 \
    kspace \
    coils \
    weights/varnet \
    ref 
```

```bash id="4NayUikFDwX3" colab={"base_uri": "https://localhost:8080/"} outputId="c83a0ac9-a4ff-4262-aeed-a8b227cbcbb5"

# if BART is compiled with gpu support, we add the --gpu option
[ $CUDA ] && GPU=--gpu;

bart reconet \
    $GPU \
    --network=modl \
    --iterations=5 \
    --normalize \
    --eval \
    --pattern=data/pattern_po_4 \
    kspace \
    coils \
    weights/modl \
    ref 
```

```python id="2KiJMXBZKvIw"

```
