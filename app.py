import os
import io
import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# === Create results folder ===
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df_store = {}


@app.route("/results/<path:filename>")
def results_files(filename):
    """Serve files from results folder."""
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/", methods=["GET", "POST"])
def index():
    global df_store
    uploaded_file = None
    df = None
    columns = []
    plot_paths = {}
    lin_metrics, ridge_metrics, lasso_metrics = {}, {}, {}
    y_col = None
    x_cols = []
    pred_value = None
    fit_status = None
    best_model_name = None

    if request.method == "POST":
        # ---- CSV Upload ----
        if "file" in request.files:
            uploaded_file = request.files["file"]
            if uploaded_file.filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                df_store["df"] = df
                columns = list(df.columns)
                return render_template("index.html", columns=columns, uploaded=True)

        # ---- Run Analysis ----
        elif "analyze" in request.form:
            df = df_store.get("df")
            if df is None:
                return redirect(url_for("index"))

            y_col = request.form.get("y_col")
            x_cols = request.form.getlist("x_cols")
            X = df[x_cols].values
            y = df[y_col].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Save scaler for manual prediction
            np.save("scaler_mean.npy", scaler.mean_)
            np.save("scaler_scale.npy", scaler.scale_)

            # === Evaluation Function ===
            def evaluate(model, name):
                model.fit(X_train_s, y_train)
                y_pred_train = model.predict(X_train_s)
                y_pred_test = model.predict(X_test_s)

                mse = mean_squared_error(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred_test)
                adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)

                # Save plots
                def save_plot(fig, filename):
                    path = os.path.join(RESULTS_DIR, filename)
                    fig.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    return filename

                # Actual vs Predicted
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred_test, color="blue")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Actual vs Predicted - {name}")
                plot_paths[f"{name}_scatter"] = save_plot(fig, f"{name.lower()}_scatter.png")

                # Residual Plot
                fig, ax = plt.subplots()
                ax.scatter(y_pred_train, y_train - y_pred_train, color="purple")
                ax.axhline(y=0, color="red", linestyle="--")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residuals")
                ax.set_title(f"Residual Plot - {name}")
                plot_paths[f"{name}_residual"] = save_plot(fig, f"{name.lower()}_residual.png")

                return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2, "Adj_R2": adj_r2}

            # === Train Models ===
            lin = LinearRegression()
            ridge = Ridge(alpha=1.0)
            lasso = Lasso(alpha=0.01)

            lin_metrics = evaluate(lin, "Linear")
            ridge_metrics = evaluate(ridge, "Ridge")
            lasso_metrics = evaluate(lasso, "Lasso")

            # Save Linear model params for manual prediction
            np.save("lin_coef.npy", lin.coef_)
            np.save("lin_intercept.npy", lin.intercept_)

            # === Determine best model ===
            best_model = max(
                [("Linear Regression", lin_metrics), ("Ridge", ridge_metrics), ("Lasso", lasso_metrics)],
                key=lambda x: x[1]["R2"]
            )
            best_model_name = best_model[0]

            return render_template(
                "index.html",
                columns=list(df.columns),
                lin_metrics=lin_metrics,
                ridge_metrics=ridge_metrics,
                lasso_metrics=lasso_metrics,
                y_col=y_col,
                x_cols=x_cols,
                plot_paths=plot_paths,
                best_model=best_model_name,
            )

        # ---- Manual Prediction ----
        elif "manual_predict" in request.form:
            df = df_store.get("df")
            if df is None:
                return redirect(url_for("index"))

            y_col = request.form.get("y_col")
            x_cols = request.form.getlist("x_cols")

            scaler = StandardScaler()
            scaler.mean_ = np.load("scaler_mean.npy")
            scaler.scale_ = np.load("scaler_scale.npy")
            lin_coef = np.load("lin_coef.npy")
            lin_intercept = np.load("lin_intercept.npy")

            input_vals = np.array([[float(request.form.get(c, 0)) for c in x_cols]])
            X_input_s = (input_vals - scaler.mean_) / scaler.scale_
            y_pred_val = X_input_s.dot(lin_coef) + lin_intercept
            pred_value = float(y_pred_val)

            # Determine fit type
            if lin_metrics:
                r2 = lin_metrics.get("R2", 0)
                if r2 > 0.9:
                    fit_status = "Overfitting (High R²)"
                elif r2 < 0.5:
                    fit_status = "Underfitting (Low R²)"
                else:
                    fit_status = "Good Fit"
            else:
                fit_status = "Unknown"

            return render_template(
                "index.html",
                columns=list(df.columns),
                pred_value=pred_value,
                fit_status=fit_status,
                y_col=y_col,
                x_cols=x_cols,
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
