# ExtraPose2

## Requirements ?!
ref: https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3
-Generated with:
conda list -e > conda_requirements.txt
-Install:
conda create --name <env> --file conda_requirements.txt

-Generated with:
conda activate <env>
conda install pip
pip list --format=freeze > pip_requirements.txt
-Install:
python3 -m venv env
source env/bin/activate
pip install -r pip_requirements.txt