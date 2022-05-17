# INFN-SOS
Exercises for INFN School of Statistics

### Dependencies and Installation
The jupyter notebooks in this package depend on a few well-known Python packages:

| __modules__   | __description__     |
| :---          | :---        |
| numpy         | array manipulation and numerical analysis      |
| scipy         | scientific computing    |
| iminuit       | Minuit, the celebrated CERN function minimizer |
| matplotlib    | a widely used plotting module for producing high quality plots |
| tqdm          | progress monitor |
| imageio       | to display images |
| emcee         | Markov chain Monte Carlo sampling |
| joblib        | to save objects to a file and read them back into memory |
| pandas        | data table manipulation, often with data loaded from csv files |
| pytorch       | a powerful, flexible, machine learning toolkit |
| scikit-learn  | easy to use machine learning toolkit |

Also recommended are

| __modules__   | __description__     |
| :---          | :--- |
| sympy         | an excellent symbolic algebra module |

The simplest way to install these packages is first to install miniconda (a slim version of Anaconda) on your laptop by following the instructions at:

https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

I recommend a __Regular installation__ of miniconda3, which comes pre-packaged with Python 3.

In principle, software release systems such as Anaconda (__conda__ for short) make it possible to have several separate self-consistent named *environments* on a single machine, say your laptop. For example, you may need to use Python 3.8.x sometimes and Python 3.11.x at other times. If you install software without using *environments* there is the very real danger that the software on your laptop will become inconsistent. Anaconda (and its lightweight companion miniconda) provide a way to create a consistent software *environments* But, like all software, Anaconda is far from perfect and problems do sometimes arise!


After installing miniconda3, It is a good idea to update conda before doing anything else using the command
```bash
conda update conda
```
Assuming conda is properly installed and initialized on your laptop, you can create an environment, here we call it __sos__, containing the __root__ package from CERN, plus a large subset of the packages in the conda system, using the command>
```bash
conda create -c conda-forge --name sos root
```
Before pressing __y__ to continue with the installation, scan through the list of packages and identify which of the above are in the list. That way, you will know which ones are missing and need to be installed using the __conda install__ command.  In order to install missing packages, first be sure to choose the conda environment into which the package is to be installed. Activate the desired environment, by doing, for example,
```bash
conda activate sos
```
Later, in order to update root together with a consistent set of packages do
```bash
conda update root
```
taking care to do so in the desired conda environment, here __sos__. 

(Unfortunately, after upgrading to macOS Monterey Version 12.3.1 and Xcode Version 3.2 on my MacBook Air with the M1 chip, __root__ no longer works on my Mac! I hope you have better luck.)  

### Other Packages

You may also wish to install the rather impressive 3D animation package __vpython__,
```bash
conda install vpython -c vpython
```

If all goes well, you will have installed a rather complete set of amazing high quality *absolutely free* software packages on your system that are consistent with Python 3.

For some quick help on conda see 

https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/


If you still prefer to do everything by hand, follow the instructions at

https://www.scipy.org/install.html

and 

https://jupyter.org/install


### 1. Download
It is a good idea to organize your computer-based projects in a systematic way. For example, in your home directory (usually the area identified by the environment variable $HOME), you may wish to create a directory (i.e., folder) called __Projects__
```bash
cd
mkdir Projects
```
In a terminal window dedicated to running the jupyter notebook app, do
```bash
cd
cd Projects
jupyter notebook
```
This will run the notebook in your browser and block the terminal window, which you can then iconize.

In another terminal window, go to folder Projects
```bash
cd
cd Projects
```
and execute the command
```bash
git clone https://github.com/hbprosper/INFN-SOS
```
This should download the package *INFN-SOS* to your current directory.

### 2. Notebooks

The notebooks provide detailed background information and explanations and are well-commented.

| __folders__                   | __description__     |
| :---          | :--- |
01_prob     | probability exercises |
02_stats    | statistics exercises |
03_ml       | machine learning exercises |
