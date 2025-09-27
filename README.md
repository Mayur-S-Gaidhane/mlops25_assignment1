# MLOPS-25 Assignment-1 : HousingRegression

Automated ML workflow to predict house prices (Boston dataset) using classical ML, with CI.

- Branches
  - `main`: README only (per instructions).
  - `reg`: 3 baseline regressors (Linear_Regression,Ridge_Regression,RandomForestRegressor).
  - `hyper`: Hyperparameter tuning of RandomForest on `n_estimators ∈ {50,100,150}`.

## Local run
```bash
Conda environment create :
conda create -y -n mlops-a1 python=3.9
conda activate mlops-a1
pip install -r requirements.txt
python regression.py --mode reg ( for reg_branch )  
python regression.py --mode hyper ( for hyper_branch)

## For  hyperparameter tuning

# 2) # 3) Create hyper branch from main and add same files
git checkout -b hyper_branch main
mkdir -p .github/workflows
# add the SAME files (utils.py, regression.py, requirements.txt, ci.yml)
# update regression.py and ci.yml for hyperparameter tuning .
git add .
git commit -m "feat: RF hyperparameter tuning (n_estimators) + CI"
git push -u origin hyper_branch



