# Code

1. Create conda environment
conda create --name rl-tax python=3.7 --yes
conda activate rl-tax

2. Install dependencies
pip install ai-economist>=1.5 # install using pip (there are no conda channels has this)
conda install -c conda-forge gym==0.21 # install using conda-forge channel
conda install tensorflow==1.14 
pip install "ray[rllib]==0.8.4" # install using pip (there are no conda channels has this) 
conda install numpy==1.21.0

3. Run 