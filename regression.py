# regression.py

import utils
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

def main():
    # Load and split the data
    df = utils.load_data()
    X_train, X_test, y_train, y_test = utils.split_data(df)

    # Define the models to compare
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    print("--- Model Performance Comparison ---")

    # Train and evaluate each model
    for name, model in models.items():
        # Train the model
        trained_model = utils.train_model(model, X_train, y_train)

        # Evaluate the model
        mse, r2 = utils.evaluate_model(trained_model, X_test, y_test)

        # Print the performance metrics
        print(f"\nModel: {name}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  R-squared (R2) Score: {r2:.4f}")

if __name__ == "__main__":
    main()