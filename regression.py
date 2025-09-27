# regression.py (for hyper_branch)

import utils
from sklearn.ensemble import RandomForestRegressor

def main():
    # Load and split the data
    df = utils.load_data()
    X_train, X_test, y_train, y_test = utils.split_data(df)

    # Define hyperparameter values to test
    n_estimator_values = [50, 100, 150]

    print("--- Hyperparameter Tuning for RandomForestRegressor ---")

    # Loop through the hyperparameter values
    for n in n_estimator_values:
        # Create a model with the specific hyperparameter
        model = RandomForestRegressor(n_estimators=n, random_state=42)

        # Train the model
        trained_model = utils.train_model(model, X_train, y_train)

        # Evaluate the model
        mse, r2 = utils.evaluate_model(trained_model, X_test, y_test)

        # Print the performance metrics
        print(f"\nHyperparameter: n_estimators = {n}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  R-squared (R2) Score: {r2:.4f}")

if __name__ == "__main__":
    main()