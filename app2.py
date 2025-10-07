import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, url_for
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Store results inside static/results so Flask can serve them via url_for('static', ...)
app.config['RESULT_FOLDER'] = os.path.join("static", "results")
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Global in-memory store (small)
df_store = {"df": None}


# ---------------- Utilities ----------------
def safe_filename(name: str) -> str:
    """Unique filename using timestamp"""
    return f"{int(time.time() * 1000)}_{name}"


def compute_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    # heuristic accuracy (R² -> % could be misleading; here we show a simple mapping)
    accuracy = max(0.0, min(100.0, r2 * 100.0))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2, "Accuracy_%": accuracy}


def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float('nan')


def run_gradient_descent(X, y, lr=0.01, epochs=100):
    n, m = X.shape
    w = np.zeros(m)
    losses = []
    for _ in range(epochs):
        preds = X.dot(w)
        err = preds - y
        loss = float(np.mean(err ** 2))
        losses.append(loss)
        grad = (2.0 / n) * X.T.dot(err)
        w = w - lr * grad
    return w, losses


# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    # template variables default
    rows = cols = None
    head_html = None
    missing = {}
    columns = []
    results = None
    scatter_file = grad_file = None
    clean_file = None
    manual_pred_value = None
    last_features = None
    last_target = None
    fit_hint = None
    gd_info = None

    if request.method == "POST":
        # -------- 1) Upload dataset ----------
        if 'file' in request.files and request.files['file'].filename != "":
            file = request.files['file']
            fname = file.filename
            try:
                if fname.lower().endswith('.csv'):
                    df = pd.read_csv(file)
                elif fname.lower().endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(file)
                else:
                    return render_template("index2.html", error="Unsupported file type. Use CSV or Excel.")
            except Exception as e:
                return render_template("index2.html", error=f"Error reading file: {e}")

            # store raw df in memory
            df_store['df'] = df.copy()

            # missing counts
            for c in df.columns:
                col = df[c]
                na_count = int(col.isna().sum())
                blank_count = int((col.astype(str).str.strip().isin(['', 'nan', 'none', 'None', 'NA', 'na'])).sum())
                combined = int(((col.isna()) | (col.astype(str).str.strip().isin(['', 'nan', 'none', 'None', 'NA', 'na']))).sum())
                missing[c] = {"combined": combined, "na": na_count, "blank": blank_count}

            # Save a cleaned copy (not filled yet; user will run analysis which fills numeric)
            clean_file = safe_filename("uploaded_dataset.csv")
            path_clean = os.path.join(app.config['RESULT_FOLDER'], clean_file)
            df.to_csv(path_clean, index=False)

            rows, cols = df.shape
            head_html = df.head(5).to_html(classes="table table-striped", index=False)
            columns = list(df.columns)

            return render_template("index2.html",
                                   rows=rows, cols=cols, head=head_html,
                                   missing=missing, columns=columns,
                                   clean_file=clean_file,
                                   msg="File uploaded. Preview & missing summary shown.")

        # -------- 2) Run analysis (train) ----------
        # We expect form fields: y_col, x_cols (multiple), lam, learning_rate, epochs, split_ratio
        if request.form.get('action') == 'train' or request.form.get('y_col'):
            df = df_store.get('df')
            if df is None:
                return render_template("index2.html", error="Please upload a dataset first.")

            # Read parameters
            y_col = request.form.get('y_col')
            x_cols = request.form.getlist('x_cols')
            lam = float(request.form.get('lam', 1.0))
            lr = float(request.form.get('learning_rate', 0.01))
            epochs = int(request.form.get('epochs', 100))
            split_ratio = float(request.form.get('split_ratio', 0.8))

            if not y_col or len(x_cols) == 0:
                return render_template("index2.html", error="Select at least one feature and a target.", columns=list(df.columns))

            # Make a copy and coerce numeric features
            sub = df[x_cols + [y_col]].copy()
            sub.replace(['', ' ', 'NA', 'na', 'None', 'none', 'nan'], np.nan, inplace=True)
            sub.dropna(subset=[y_col], inplace=True)

            # Fill numeric features by mean (avoid chained assignment warning)
            for c in x_cols:
                sub[c] = pd.to_numeric(sub[c], errors='coerce')
                sub[c] = sub[c].fillna(sub[c].mean())

            # Ensure y numeric
            sub[y_col] = pd.to_numeric(sub[y_col], errors='coerce')
            sub = sub.dropna(subset=[y_col])  # drop rows where y couldn't be numeric

            # Save cleaned dataset (now numeric features filled)
            clean_file = safe_filename("cleaned_dataset.csv")
            path_clean = os.path.join(app.config['RESULT_FOLDER'], clean_file)
            sub.to_csv(path_clean, index=False)

            X_all = sub[x_cols].values.astype(float)
            y_all = sub[y_col].values.astype(float)

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=split_ratio, random_state=42)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            n_train, p = X_train_s.shape

            # Linear
            lin = LinearRegression().fit(X_train_s, y_train)
            lin_pred = lin.predict(X_test_s)
            lin_metrics = compute_metrics(y_test, lin_pred)
            lin_adj = adjusted_r2(lin_metrics['R2'], len(y_test), p)

            # Ridge
            ridge = Ridge(alpha=lam).fit(X_train_s, y_train)
            ridge_pred = ridge.predict(X_test_s)
            ridge_metrics = compute_metrics(y_test, ridge_pred)
            ridge_adj = adjusted_r2(ridge_metrics['R2'], len(y_test), p)

            # Lasso
            lasso = Lasso(alpha=lam, max_iter=20000).fit(X_train_s, y_train)
            lasso_pred = lasso.predict(X_test_s)
            lasso_metrics = compute_metrics(y_test, lasso_pred)
            lasso_adj = adjusted_r2(lasso_metrics['R2'], len(y_test), p)

            # Aggregate results and add Adj R2 (rounded in template)
            results = {
                "Linear Regression": {**lin_metrics, "Adj_R2": lin_adj},
                f"Ridge (alpha={lam})": {**ridge_metrics, "Adj_R2": ridge_adj},
                f"Lasso (alpha={lam})": {**lasso_metrics, "Adj_R2": lasso_adj}
            }

            # Best model by Adj R2
            best_model = max(results.keys(), key=lambda k: results[k]["Adj_R2"] if not np.isnan(results[k]["Adj_R2"]) else -999)

            # Scatter plot if single feature
            if X_all.shape[1] == 1:
                plt.figure(figsize=(7, 5))
                plt.scatter(X_train[:, 0], y_train, label='Train', alpha=0.6)
                plt.scatter(X_test[:, 0], y_test, label='Test', alpha=0.6)
                xx = np.linspace(X_all.min(), X_all.max(), 200).reshape(-1, 1)
                xx_s = scaler.transform(xx)
                plt.plot(xx.flatten(), lin.predict(xx_s), color='red', label='Linear Fit')
                plt.xlabel(x_cols[0])
                plt.ylabel(y_col)
                plt.legend()
                plt.title("Scatter & Best Fit Line")
                scatter_file = safe_filename("scatter.png")
                plt.savefig(os.path.join(app.config['RESULT_FOLDER'], scatter_file), bbox_inches='tight')
                plt.close()
                scatter_file = scatter_file
            else:
                scatter_file = None

            # Gradient descent training (for demonstration)
            Xg = np.hstack([np.ones((X_train_s.shape[0], 1)), X_train_s])
            w_gd, losses = run_gradient_descent(Xg, y_train, lr=lr, epochs=epochs)
            grad_file = safe_filename("grad.png")
            plt.figure(figsize=(7, 4))
            plt.plot(range(1, epochs + 1), losses, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.title("Gradient Descent Loss")
            plt.savefig(os.path.join(app.config['RESULT_FOLDER'], grad_file), bbox_inches='tight')
            plt.close()
            gd_info = {"min_loss": float(np.min(losses)), "min_epoch": int(np.argmin(losses) + 1)}

            # Save model artifacts for manual prediction
            np.save(os.path.join(app.config['RESULT_FOLDER'], "lin_coef.npy"), lin.coef_)
            np.save(os.path.join(app.config['RESULT_FOLDER'], "lin_intercept.npy"), lin.intercept_)
            np.save(os.path.join(app.config['RESULT_FOLDER'], "scaler_mean.npy"), scaler.mean_)
            np.save(os.path.join(app.config['RESULT_FOLDER'], "scaler_scale.npy"), scaler.scale_)
            np.save(os.path.join(app.config['RESULT_FOLDER'], "x_cols.npy"), np.array(x_cols))
            np.save(os.path.join(app.config['RESULT_FOLDER'], "y_col.npy"), np.array([y_col]))

            # Provide hint about over/underfitting (simple heuristic)
            train_rmse = float(np.sqrt(mean_squared_error(y_train, lin.predict(X_train_s))))
            test_rmse = float(lin_metrics["RMSE"])
            gap = test_rmse - train_rmse
            if gap > 0.5 * train_rmse:
                fit_hint = "Likely overfitting (train error much lower than test error). Consider regularization or more data."
            elif lin_metrics["R2"] < 0.2:
                fit_hint = "Model may be underfitting (low R²). Try adding features or non-linear models."
            else:
                fit_hint = "No strong sign of overfitting/underfitting from simple heuristic."

            # Save cleaned filename to show download
            clean_file = clean_file

            # Render template with results
            return render_template(
                "index2.html",
                rows=sub.shape[0],
                cols=sub.shape[1],
                head=sub.head(5).to_html(classes="table table-striped", index=False),
                missing=missing,
                columns=list(df.columns),
                results=results,
                best_model=best_model,
                scatter_file=scatter_file,
                grad_file=grad_file,
                clean_file=clean_file,
                last_features=x_cols,
                last_target=y_col,
                fit_hint=fit_hint,
                gd_info=gd_info
            )

        # -------- 3) Manual prediction ----------
        elif 'manual_predict' in request.form:
            # Load last features
            resdir = app.config['RESULT_FOLDER']
            try:
                x_cols = np.load(os.path.join(resdir, "x_cols.npy"), allow_pickle=True).tolist()
                y_col = np.load(os.path.join(resdir, "y_col.npy"), allow_pickle=True).item()
            except Exception:
                return render_template("index2.html", error="No trained model found. Run training first.")

            # load scaler & linear coefficients
            scaler = StandardScaler()
            scaler.mean_ = np.load(os.path.join(resdir, "scaler_mean.npy"))
            scaler.scale_ = np.load(os.path.join(resdir, "scaler_scale.npy"))
            lin_coef = np.load(os.path.join(resdir, "lin_coef.npy"))
            lin_intercept = np.load(os.path.join(resdir, "lin_intercept.npy"))

            # collect inputs
            try:
                input_vals = np.array([[float(request.form.get(c, 0.0)) for c in x_cols]])
            except Exception as e:
                return render_template("index2.html", error=f"Invalid input values: {e}")

            # standardize manually and predict
            X_input_s = (input_vals - scaler.mean_) / scaler.scale_
            y_pred_val = X_input_s.dot(lin_coef) + lin_intercept
            manual_pred_value = float(y_pred_val[0])

            return render_template("index2.html",
                                   manual_pred_value=manual_pred_value,
                                   manual_pred_target=y_col,
                                   last_features=x_cols)

    # GET (or fallthrough POST without early return) - show default page
    return render_template("index2.html", columns=(list(df_store['df'].columns) if df_store['df'] is not None else None))


if __name__ == "__main__":
    # enable debug mode locally for easier troubleshooting
    app.run(debug=True)
