# MLOPS-25 Assignment-1 : HousingRegression

Automated ML workflow to predict house prices (Boston dataset) using classical ML, with CI.

- Branches
  - `main`: README only (per instructions).
  - `reg`: 3 baseline regressors (Linear_Regression,Ridge_Regression,RandomForestRegressor).
  - `hyper`: Hyperparameter tuning of RandomForest on `n_estimators ∈ {50,100,200}`.

## Local run
```bash
Conda environment create :
conda create -y -n mlops-a1 python=3.9
conda activate mlops-a1
pip install -r requirements.txt
python regression.py --mode reg ( for reg_branch )  
python regression.py --mode hyper ( for hyper_branch)

## For Regresion 

# 2) Create reg branch: add code + CI
ggit checkout -b reg_branch 
mkdir -p .github/workflows
# create files: utils.py, regression.py, requirements.txt, .github/workflows/ci.yml
git add .
git commit -m "feat: regression baseline + CI"
git push -u origin reg_branch



