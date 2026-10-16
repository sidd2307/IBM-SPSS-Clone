import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import plotly.express as px
import pyreadstat
import tempfile
import os
import io

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & SPSS CSS STYLING
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Marketing Research Agent - Discriminant Analysis & Logistic Regression", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f5;
    }
    .spss-header {
        background-color: #1f4e78;
        color: white;
        padding: 8px 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
        font-weight: bold;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .spss-table {
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
        font-family: 'Courier New', Courier, monospace;
        background-color: #ffffff;
    }
    .spss-table th {
        background-color: #d9e1f2;
        color: #000000;
        border: 1px solid #b4c6e7;
        padding: 6px;
        text-align: right;
        font-size: 13px;
    }
    .spss-table td {
        border: 1px solid #d9d9d9;
        padding: 5px;
        text-align: right;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if 'df' not in st.session_state:
    # Default Sample Data from PDF (Shiva Dhaba / Brand Loyalty Case)[cite: 8, 20]
    default_data = {
        'Customer_ID': list(range(1, 31)),
        'Loyalty_Code': [3,3,3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2,2,2, 1,1,1,1,1,1,1,1,1,1],
        'Loyalty_Binary': [1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1, 0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
        'Brand_Attitude': [4,6,5,7,6,3,5,5,7,7, 6,5,7,5,7, 3,4,2,5,4, 3,3,3,4,6,3,4,3,5,1],
        'Product_Attitude': [3,4,2,5,3,4,5,4,5,6, 7,6,3,1,5, 1,6,5,2,1, 3,4,6,4,3,6,3,5,5,3],
        'Shopping_Attitude': [5,4,4,5,4,5,5,2,4,4, 2,4,3,4,5, 3,2,2,4,3, 4,5,3,2,6,3,2,2,3,2],
        'Avg_Spend_INR': [3082,2510,3705,3499,2699,2975,3806,2189,2957,2686,
                          1757,1362,1631,1954,1446,820,1640,966,1187,1400,
                          515,213,441,545,764,539,291,566,654,627],
        'Complaint_Rate_Pct': [4.2,3.7,2.7,2.9,4.8,3.0,1.4,1.5,0.8,0.1,
                               9.8,9.6,8.6,6.1,7.8,12.4,13.1,11.5,14.3,11.9,
                               38.0,36.4,26.8,18.3,24.9,32.1,32.0,30.2,22.6,29.5]
    }
    st.session_state.df = pd.DataFrame(default_data)

if 'var_metadata' not in st.session_state:
    st.session_state.var_metadata = pd.DataFrame({
        'Variable': st.session_state.df.columns,
        'Label': st.session_state.df.columns,
        'Type': ['Numeric']*len(st.session_state.df.columns),
        'Measure': ['Nominal' if c in ['Customer_ID','Loyalty_Code','Loyalty_Binary'] else 'Scale' for c in st.session_state.df.columns]
    })

def convert_results_to_csv(tables_dict):
    output = io.StringIO()
    for title, df in tables_dict.items():
        output.write(f"=== {title} ===\n")
        df.to_csv(output, index=True if df.index.name else False)
        output.write("\n\n")
    return output.getvalue()

# -----------------------------------------------------------------------------
# TOP NAVBAR / HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="spss-header">IBM SPSS Statistics - [DataSet1.sav]</div>', unsafe_allow_html=True)

sidebar_menu = st.sidebar.selectbox("Analyze Menu", [
    "1. Data Editor (Data / Variable View)",
    "2. Regression -> Binary Logistic...",
    "3. Classify -> Discriminant..."
])

# File Uploader
uploaded_file = st.sidebar.file_uploader("Open Data File (.sav, .csv)", type=["sav", "csv"])

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_ext == ".sav":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sav") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            df_sav, meta = pyreadstat.read_sav(tmp_path)
            st.session_state.df = df_sav
            
            measure_mapping = {'nominal': 'Nominal', 'ordinal': 'Ordinal', 'scale': 'Scale', 'unknown': 'Scale'}
            var_types = ['Numeric' if pd.api.types.is_numeric_dtype(df_sav[col]) else 'String' for col in df_sav.columns]
            var_labels = [meta.column_names_to_labels.get(col, col) for col in df_sav.columns]
            var_measures = [measure_mapping.get(meta.variable_measure.get(col, 'scale'), 'Scale') for col in df_sav.columns]
            
            st.session_state.var_metadata = pd.DataFrame({
                'Variable': df_sav.columns,
                'Label': var_labels,
                'Type': var_types,
                'Measure': var_measures
            })
            st.sidebar.success(f"Successfully loaded '{uploaded_file.name}'")
        except Exception as e:
            st.sidebar.error(f"Error reading .sav file: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    elif file_ext == ".csv":
        df_csv = pd.read_csv(uploaded_file)
        st.session_state.df = df_csv
        st.session_state.var_metadata = pd.DataFrame({
            'Variable': df_csv.columns,
            'Label': df_csv.columns,
            'Type': ['Numeric' if pd.api.types.is_numeric_dtype(df_csv[col]) else 'String' for col in df_csv.columns],
            'Measure': ['Scale']*len(df_csv.columns)
        })
        st.sidebar.success(f"Successfully loaded '{uploaded_file.name}'")

# -----------------------------------------------------------------------------
# MODULE 1: DATA EDITOR
# -----------------------------------------------------------------------------
if sidebar_menu == "1. Data Editor (Data / Variable View)":
    tab1, tab2 = st.tabs(["Data View", "Variable View"])
    
    with tab1:
        st.caption("Data View - Dynamic Cell Editor")
        edited_df = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True, key="data_editor_grid")
        st.session_state.df = edited_df

    with tab2:
        st.caption("Variable View - Metadata Properties")
        edited_meta = st.data_editor(
            st.session_state.var_metadata,
            num_rows="dynamic",
            column_config={
                "Type": st.column_config.SelectboxColumn("Type", options=["Numeric", "String"]),
                "Measure": st.column_config.SelectboxColumn("Measure", options=["Scale", "Nominal", "Ordinal"])
            },
            hide_index=True,
            use_container_width=True
        )
        st.session_state.var_metadata = edited_meta

# -----------------------------------------------------------------------------
# MODULE 2: BINARY LOGISTIC REGRESSION
# -----------------------------------------------------------------------------
elif sidebar_menu == "2. Regression -> Binary Logistic...":
    st.subheader("Logistic Regression")
    
    col_left, col_right = st.columns([1, 2])
    all_vars = list(st.session_state.df.columns)
    
    with col_left:
        dep_var = st.selectbox("Dependent Variable (Binary)", options=all_vars, index=min(2, len(all_vars)-1))
        indep_vars = st.multiselect("Covariates (Predictors)", options=[v for v in all_vars if v != dep_var], default=[v for v in all_vars if v != dep_var][:3])
        
        cutoff = st.number_input("Classification Cutoff", value=0.5, step=0.05)
        run_logit = st.button("OK (Run Analysis)", type="primary")

    with col_right:
        if run_logit:
            if not dep_var or not indep_vars:
                st.error("Please select both Dependent Variable and Covariates.")
            else:
                data = st.session_state.df[[dep_var] + indep_vars].dropna()
                Y = data[dep_var]
                X = data[indep_vars]
                X_const = sm.add_constant(X)
                
                model = sm.Logit(Y, X_const).fit(disp=False)
                pred_probs = model.predict(X_const)
                pred_labels = (pred_probs >= cutoff).astype(int)
                
                # Model Summary
                ll_null = model.llnull
                ll_fit = model.llf
                n = len(Y)
                cox_snell = 1 - np.exp((2/n) * (ll_null - ll_fit))
                nagelkerke = cox_snell / (1 - np.exp((2/n) * ll_null))
                
                st.markdown("### Model Summary")
                summary_df = pd.DataFrame({
                    "Step": [1],
                    "-2 Log likelihood": [f"{-2 * ll_fit:.3f}"],
                    "Cox & Snell R Square": [f"{cox_snell:.3f}"],
                    "Nagelkerke R Square": [f"{nagelkerke:.3f}"]
                })
                st.table(summary_df)

                # Hosmer and Lemeshow Test
                st.markdown("### Hosmer and Lemeshow Test")
                data_hl = pd.DataFrame({'obs': Y, 'prob': pred_probs})
                data_hl['decile'] = pd.qcut(data_hl['prob'].rank(method='first'), q=10, duplicates='drop')
                
                hl_grouped = data_hl.groupby('decile', observed=False).agg(
                    obs_1=('obs', 'sum'),
                    obs_0=('obs', 'count'),
                    exp_1=('prob', 'sum'),
                    mean_prob=('prob', 'mean'),
                    total=('obs', 'count')
                )
                hl_grouped['obs_0'] = hl_grouped['obs_0'] - hl_grouped['obs_1']
                hl_grouped['exp_0'] = hl_grouped['total'] - hl_grouped['exp_1']
                
                hl_stat = np.sum(
                    ((hl_grouped['obs_1'] - hl_grouped['exp_1'])**2 / (hl_grouped['exp_1'] + 1e-9)) +
                    ((hl_grouped['obs_0'] - hl_grouped['exp_0'])**2 / (hl_grouped['exp_0'] + 1e-9))
                )
                hl_df = max(1, len(hl_grouped) - 2)
                hl_p = 1 - stats.chi2.cdf(hl_stat, hl_df)
                
                hl_res_df = pd.DataFrame({
                    "Step": [1],
                    "Chi-square": [f"{hl_stat:.3f}"],
                    "df": [hl_df],
                    "Sig.": [f"{hl_p:.3f}"]
                })
                st.table(hl_res_df)

                # Classification Table
                st.markdown("### Classification Table")
                cm = confusion_matrix(Y, pred_labels)
                tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0,0], 0, 0, cm[1,1])
                
                correct_0 = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
                correct_1 = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
                overall_acc = ((tn + tp) / len(Y)) * 100
                
                class_df = pd.DataFrame({
                    "Observed": ["0 (Unsuccessful)", "1 (Successful)", "Overall Percentage"],
                    "Predicted 0": [tn, fn, ""],
                    "Predicted 1": [fp, tp, ""],
                    "Percentage Correct": [f"{correct_0:.1f}%", f"{correct_1:.1f}%", f"{overall_acc:.1f}%"]
                })
                st.table(class_df)

                # Variables in the Equation
                st.markdown("### Variables in the Equation")
                params = model.params
                bse = model.bse
                wald = (params / bse) ** 2
                p_values = model.pvalues
                odds_ratios = np.exp(params)
                
                var_eq_df = pd.DataFrame({
                    "Variable": params.index,
                    "B": params.values.round(3),
                    "S.E.": bse.values.round(3),
                    "Wald": wald.values.round(3),
                    "df": [1]*len(params),
                    "Sig.": p_values.values.round(3),
                    "Exp(B)": odds_ratios.values.round(3)
                })
                st.table(var_eq_df)

                results_export = {
                    "Model Summary": summary_df,
                    "Hosmer and Lemeshow Test": hl_res_df,
                    "Classification Table": class_df,
                    "Variables in the Equation": var_eq_df
                }
                st.download_button(
                    label="📥 Download Analysis Output (CSV)",
                    data=convert_results_to_csv(results_export),
                    file_name="Logistic_Regression_Results.csv",
                    mime="text/csv"
                )

# -----------------------------------------------------------------------------
# MODULE 3: DISCRIMINANT ANALYSIS (EXACT SPSS MATRIX MATH)
# -----------------------------------------------------------------------------
elif sidebar_menu == "3. Classify -> Discriminant...":
    st.subheader("Discriminant Analysis")
    
    col_left, col_right = st.columns([1, 2])
    all_vars = list(st.session_state.df.columns)
    
    with col_left:
        group_var = st.selectbox("Grouping Variable", options=all_vars, index=min(1, len(all_vars)-1))
        indep_vars = st.multiselect("Independent Variables", options=[v for v in all_vars if v != group_var], 
                                     default=[v for v in all_vars if v != group_var][:3])
        
        run_da = st.button("OK (Run Analysis)", type="primary")

    with col_right:
        if run_da:
            if not group_var or not indep_vars:
                st.error("Please define Grouping Variable and Independent Variables.")
            else:
                data = st.session_state.df[[group_var] + indep_vars].dropna()
                X_df = data[indep_vars]
                y_s = data[group_var]
                
                X = X_df.values
                y = y_s.values
                
                groups = np.unique(y)
                g = len(groups)
                p = X.shape[1]
                N = X.shape[0]
                n_functions = min(g - 1, p)
                
                # --- Exact MANOVA Matrix Computations ---
                grand_mean = np.mean(X, axis=0)
                
                # Calculate Between-Groups (B) and Within-Groups (W) Matrices
                W = np.zeros((p, p))
                B = np.zeros((p, p))
                
                group_means = {}
                group_counts = {}
                
                for k in groups:
                    X_k = X[y == k]
                    n_k = X_k.shape[0]
                    mean_k = np.mean(X_k, axis=0)
                    
                    group_means[k] = mean_k
                    group_counts[k] = n_k
                    
                    W += np.dot((X_k - mean_k).T, (X_k - mean_k))
                    diff_k = (mean_k - grand_mean).reshape(-1, 1)
                    B += n_k * np.dot(diff_k, diff_k.T)
                
                # Generalized Eigenvalue Problem: W^-1 B v = lambda v
                W_inv = np.linalg.pinv(W)
                eigenvals_raw, eigenvecs_raw = np.linalg.eig(np.dot(W_inv, B))
                
                # Sort Eigenvalues descending
                idx = np.argsort(np.real(eigenvals_raw))[::-1][:n_functions]
                eigenvalues = np.real(eigenvals_raw[idx])
                raw_coefficients = np.real(eigenvecs_raw[:, idx])
                
                # Canonical Correlations
                canonical_correlations = np.sqrt(eigenvalues / (1 + eigenvalues))
                
                pct_variance = (eigenvalues / np.sum(eigenvalues)) * 100
                cum_pct_variance = np.cumsum(pct_variance)
                
                # --- 1. Eigenvalues Table ---
                st.markdown("### Eigenvalues")
                eig_df = pd.DataFrame({
                    "Function": [i+1 for i in range(n_functions)],
                    "Eigenvalue": eigenvalues.round(3),
                    "% of Variance": pct_variance.round(1),
                    "Cumulative %": cum_pct_variance.round(1),
                    "Canonical Correlation": canonical_correlations.round(3)
                })
                st.table(eig_df)

                # --- 2. Wilks' Lambda Table ---
                st.markdown("### Wilks' Lambda")
                wilks_lambdas = []
                chi_sqs = []
                dfs = []
                p_vals = []
                
                for k in range(n_functions):
                    w_lambda = np.prod(1 / (1 + eigenvalues[k:]))
                    df_val = (p - k) * (g - 1 - k)
                    chi_sq = -(N - 1 - (p + g) / 2) * np.log(w_lambda)
                    p_val = 1 - stats.chi2.cdf(chi_sq, df_val) if df_val > 0 else 1.0
                    
                    wilks_lambdas.append(round(w_lambda, 3))
                    chi_sqs.append(round(chi_sq, 3))
                    dfs.append(int(df_val))
                    p_vals.append("<.001" if p_val < 0.001 else f"{p_val:.3f}")

                wilks_df = pd.DataFrame({
                    "Test of Function(s)": [f"{i+1} through {n_functions}" for i in range(n_functions)],
                    "Wilks' Lambda": wilks_lambdas,
                    "Chi-square": chi_sqs,
                    "df": dfs,
                    "Sig.": p_vals
                })
                st.table(wilks_df)

                # --- 3. Standardized Canonical Coefficients ---
                st.markdown("### Standardized Canonical Discriminant Function Coefficients")
                # Pooled Within-Group Standard Deviations
                pooled_cov = W / (N - g)
                pooled_std = np.sqrt(np.diag(pooled_cov))
                
                # Normalize raw coefficients so v^T (W / (N-g)) v = 1
                for j in range(n_functions):
                    v = raw_coefficients[:, j]
                    scale = np.sqrt(np.dot(v.T, np.dot(pooled_cov, v)))
                    raw_coefficients[:, j] = v / scale
                
                std_coefficients = raw_coefficients * pooled_std[:, np.newaxis]
                
                std_coef_df = pd.DataFrame(
                    std_coefficients.round(3),
                    index=indep_vars,
                    columns=[f"Function {i+1}" for i in range(n_functions)]
                )
                st.table(std_coef_df)

                # --- 4. Structure Matrix ---
                st.markdown("### Structure Matrix")
                # Structure Correlations: R = Pooled_Cov * Raw_Coeffs
                struct_matrix = np.zeros((p, n_functions))
                for j in range(n_functions):
                    struct_matrix[:, j] = np.dot(pooled_cov, raw_coefficients[:, j]) / pooled_std
                
                struct_df = pd.DataFrame(
                    struct_matrix.round(3),
                    index=indep_vars,
                    columns=[f"Function {i+1}" for i in range(n_functions)]
                )
                st.table(struct_df)

                # --- 5. Classification Results ---
                st.markdown("### Classification Results")
                lda_sk = LinearDiscriminantAnalysis(store_covariance=True)
                lda_sk.fit(X, y)
                y_pred = lda_sk.predict(X)
                cm_da = confusion_matrix(y, y_pred)
                
                cm_df = pd.DataFrame(cm_da, index=[f"Actual Group {c}" for c in groups], columns=[f"Pred Group {c}" for c in groups])
                st.table(cm_df)
                
                # Territorial Map Scatter Plot
                if n_functions >= 2:
                    st.markdown("### Canonical Discriminant Plot")
                    scores = np.dot(X - grand_mean, raw_coefficients[:, :2])
                    plot_df = pd.DataFrame(scores, columns=['Function 1', 'Function 2'])
                    plot_df['Group'] = y_s.astype(str).values
                    fig = px.scatter(plot_df, x='Function 1', y='Function 2', color='Group', title="Territorial Map / Group Separation")
                    st.plotly_chart(fig, use_container_width=True)

                results_export = {
                    "Eigenvalues": eig_df,
                    "Wilks Lambda": wilks_df,
                    "Standardized Canonical Coefficients": std_coef_df,
                    "Structure Matrix": struct_df,
                    "Classification Results": cm_df
                }
                st.download_button(
                    label="📥 Download Analysis Output (CSV)",
                    data=convert_results_to_csv(results_export),
                    file_name="Discriminant_Analysis_Results.csv",
                    mime="text/csv"
                )