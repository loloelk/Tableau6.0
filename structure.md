# Codebase Structure Analysis: TableaudeBord6.0

Root Directory: `/Users/laurentelkrief/Desktop/Neuromod/Research/TableaudeBord/TableaudeBord6.0`

## Detected Frameworks & Libraries

- **streamlit** (confidence: 28)
- **pandas** (confidence: 27)
- **plotly** (confidence: 20)
- **numpy** (confidence: 14)
- **scikit-learn** (confidence: 3)
- **pytorch** (confidence: 2)

## File Structure

  📄 .DS_Store
  📄 .Rhistory
  📄 .gitignore
  📄 README.md
  📄 app.py
    *Analysis (AST):*
      - Functions: `check_login, nurse_inputs_page, run_db_initialization`
      - Imports: `components.dashboard, components.ml_performance, components.nurse_imputs, components.nurse_inputs, components.overview, components.patient_journey, components.protocol_analysis, components.side_effects, components.sidebar, components.sign_out, ...`
      - Variables: `EXTENDED_DATA_CSV, PATIENT_DATA_CSV, SIMULATED_EMA_CSV, VALID_CREDENTIALS, config, css_path, current_patient, entered_password, entered_username, extended_data, final_data, keys_to_clear, login_button, ml_init_status, needs_patient, page_selected, role_display, simulated_ema_data, user_info`
  📄 dockerfile
    *Special File Analysis:*
      - Commands: `FROM, COPY, RUN, COPY, RUN, EXPOSE, ENV, ENTRYPOINT`
  📄 enhanced_simulate_patient_data.py
    *Analysis (AST):*
      - Functions: `create_synthetic_patients, distribute_phq9_score, generate_bfi_scores, generate_datasets, generate_ema_data, generate_medication_data, generate_nurse_notes_data, generate_patient_data, generate_side_effects_data, initialize_database, load_config, save_dataset, save_nurse_inputs, save_side_effect_report`
      - Imports: `RandomForestClassifier, datetime, initialize_database, joblib, logging, math, numpy, os, pandas, random, ...`
      - Variables: `ANXIETY_ITEMS, BFI_BASELINE_MEAN, BFI_BASELINE_STD, BFI_ITEMS_MAP, DB_INTERACTION_ENABLED, EMA_ENTRIES_PER_DAY_WEIGHTS, EMA_MISSING_DAY_PROB, EMA_MISSING_ENTRY_PROB, FEATURE_CORRELATIONS, MADRS_ITEMS, MEDICATION_CATEGORIES, MEDICATION_DOSAGES, NEUROTICISM_CHANGE_NON_RESPONDER, NEUROTICISM_CHANGE_RESPONDER, NEUROTICISM_RESPONSE_FACTOR, NUM_PATIENTS, N_PATIENTS, OTHER_FACTOR_CHANGE_STD, PROTOCOLS, PROTOCOL_REMISSION_RATES, PROTOCOL_RESPONSE_RATES, RESPONSE_RATE, SIDE_EFFECT_DECAY_DAY, SIDE_EFFECT_PROB_INITIAL, SIDE_EFFECT_PROB_LATER, SIMULATION_DURATION_DAYS, START_DATE, SYMPTOMS, adherence, adj_bl, adj_fu, adjusted_response_prob, age, all_ema_rows, all_nurse_rows, all_protocols, anxiety_base, anxiety_score, base_response_prob, base_start_date, baseline_item, baseline_mean, baseline_severity, basic_cols, bl_calc_score, bl_key, bl_score, bl_val, change, col_name, cols_to_drop, config, config_content, config_path, conn, correlation_sum, curr, current, current_severity, daily_variation, day_effect, day_imp, day_offset, day_progress, days, db_path, delta, df, df_basic, df_ema, df_ml, df_ml_training, df_nurse, df_patients, df_protocol, df_side, ect, el, ema_csv_path, ema_data_df, ema_date, ema_days, ema_entries, ema_entry, entry_severity, factor_scores_bl, factor_scores_fu, feat_range, feat_vals, filepath, final_comment, final_day, final_note, final_status, followup_item, followup_mean, fu_calc_score, fu_key, fu_score, fu_val, hour, idx, improved, improvement, improvement_curve, improvement_factor, improvement_rate, initial_day, initial_note, inv_n, inv_w, is_resp, item_scores_bl_calc, item_scores_fu_calc, items, latent_response, liste_comorbidites, log_file, logger, lower_doses, madrs_bl, madrs_bl_sum, madrs_fu, madrs_fu_sum, madrs_improvement, measure_pct, med, med1, med2, medications, meds_formatted, mid_comment, mid_day, mid_note, mid_status, minute, mismatch, ml_cols, mood_base, mood_score, mood_today, n_entries_planned, n_items, n_patients, n_rev, networks, neuroticism_adjustment, noise_level, non_resp_idx, non_responder_protocols, norm_feat, norm_w, notes, num_comorbidities, num_emas, num_meds, num_records, num_reports, num_saved, nurse_id, nurse_ids, patient, patient_comorbidities_list, patient_csv_path, patient_data_df, patient_data_df_clean, patient_data_simple_csv_path, patient_id, patient_start_date, patients, phq9_bl, phq9_fu, phq9_items, possible_doses, primary_category, primary_dose, primary_med, prob_cutoff, progress, protocol, protocol_cols, protocol_effect, protocols, prov_n_items_bl, prov_n_items_bl_calc, provisional_neuroticism_score_bl, psychotherapie, raw, record_date, record_days, reduction_pct, remaining_categories, report_data, report_date, resp_idx, responder_protocols, rtms, secondary_category, secondary_dose, secondary_med, severity, sex, side_effect_probs, side_effect_types, sleep_base, sleep_score, snp_names, stability, start_dates, sum_inv, table_name, target_severity, tdcs, tertiary_categories, tertiary_category, tertiary_dose, tertiary_med, threshold, timestamp, today, treatment_days, weight, weights, will_remit, will_respond`
  📄 requirements.txt
    *Special File Analysis:*
      - Packages: `streamlit, pandas, numpy, plotly, networkx, pyyaml, statsmodels, seaborn, python-dotenv, matplotlib, ...`
  📄 structure.md
📁 **config/**
  📄 config.yaml
    *Special File Analysis:*
      - Configuration: `paths, mappings`
📁 **utils/**
  📄 config_manager.py
    *Analysis (AST):*
      - Functions: `__new__, _load_config, get_config, get_value, load_config`
      - Classs: `ConfigManager`
      - Imports: `os, yaml`
      - Variables: `_config, _instance, config_path, env`
  📄 error_handler.py
    *Analysis (AST):*
      - Functions: `handle_error`
      - Imports: `logging, streamlit`
  📄 logging_config.py
    *Analysis (AST):*
      - Functions: `configure_logging, setup_logger`
      - Imports: `datetime, logging, os`
      - Variables: `console_handler, file_handler, formatter, log_dir, log_file, logger, today`
  📄 visualization.py
    *Analysis (AST):*
      - Functions: `create_bar_chart, create_heatmap, create_line_chart, create_radar_chart`
      - Imports: `pandas, plotly.express, plotly.graph_objects`
      - Variables: `cat_closed, fig, values_closed`
📁 **.devcontainer/**
  📄 devcontainer.json
📁 **components/**
  📄 dashboard.py
    *Analysis (AST):*
      - Functions: `get_patient_ema_data, patient_dashboard, treatment_progress`
      - Imports: `base64, generate_person_specific_network, get_latest_nurse_inputs, get_nurse_inputs_history, get_side_effects_history, logging, numpy, pandas, plotly.express, plotly.graph_objects, ...`
      - Variables: `MEDICATION_CATEGORIES, assumed_duration_days, author, available_categories, bfi_data_available, bfi_factors_map, bfi_table_data, bfi_table_df, bl_col, bl_score, bl_val, categories, category, clinical_tabs, color, cols, comorbidities, comorbidity_list, confidence, corr_matrix, count, created_by, csv, current_index, current_milestone_index, daily_score, daily_symptoms, date_col, days_elapsed, delta_score, display_columns, display_df_hist, dosage, exp_title, fig, fig_bfi_radar, fig_ema_trends, fig_ema_variability, fig_heatmap, fig_items, fig_madrs_total, fig_max, fig_network, fig_phq9, first_entry, fu_col, fu_score, fu_val, history_data, history_df, improvement_pct, is_remitter, is_responder, is_responder_pred, item_columns, item_label, items_data, last_entry, latest_note, latest_other, latest_plan, latest_report, madrs_bl, madrs_data, madrs_fu, madrs_items_df, madrs_items_long, madrs_total_df, max_sev, meds_data, meds_df, meds_list, metrics_cols, milestone_cols, milestones, name, name_map, notes_history_df, numeric_cols, numeric_ema_cols, parts, patient_data, patient_ema, patient_id, patient_main_df, patient_row, phq9_cols_exist, phq9_days, phq9_df, phq9_scores_over_time, plan_date, plot_data, prediction_result, progress_percentage, remission_color, remission_delta, remission_status, rename_map, report_date, response_prob, response_prob_pct, rolling_window, score, selected_category_avg, selected_category_corr, selected_category_var, selected_symptoms_avg, selected_symptoms_corr, selected_symptoms_var, severity, severity_cols, side_effects_history, status_color, status_text, summary, summary_list, symptom_categories, symptoms_available, symptoms_present, threshold, treatments, valid_items_found, values_bl, values_fu, variability_df`
  📄 ml_performance.py
    *Analysis (AST):*
      - Functions: `ml_performance_page, predict_for_range`
      - Imports: `DEFAULT_FEATURE_COLUMNS, auc, classification_report, confusion_matrix, create_response_labels, generate_evaluation_cohort, joblib, logging, numpy, os, ...`
      - Variables: `EVALUATION_COHORT_PATH, FEATURES_PATH, MODEL_PATH, TEST_METRICS_PATH, TRAINING_DATA_PATH, X, X_valid, __all__, accuracy, annotations, available_features, baseline, bl_scores, cm, cohort, cols, confusion_mat, corr_matrix, cv_df, cv_scores, df_valid, f1, feature_importance, feature_max, feature_min, feature_range, feature_to_analyze, features_df, features_markdown, fig, follow_up_col, follow_up_columns, has_follow_up, impact_df, importance, improvement, indices, metrics_df, model, n_thresholds, percentages, pr_df, precision, prediction_df, proba, probas, recall, response_counts, roc_auc, sample_copy, sample_data, selected_features, sensitivity, sensitivity_df, shapes, specificity, test_metrics, total, training_data, valid_bl, valid_fu, valid_idx, y, y_pred, y_pred_proba, y_true, y_valid`
  📄 nurse_inputs.py
    *Analysis (AST):*
      - Functions: `nurse_inputs_page`
      - Imports: `get_latest_nurse_inputs, get_nurse_inputs_history, pandas, save_nurse_inputs, services.nurse_service, streamlit`
      - Variables: `GOAL_STATUS_OPTIONS, comments_input, current_status, display_columns, display_df, expander_title, goal_status_input, history_df, latest_inputs, objectives_input, patient_id, planned_interventions_input, rename_map, status_index, submit_button, success, target_symptoms_input, tasks_input`
  📄 overview.py
    *Analysis (AST):*
      - Functions: `main_dashboard_page`
      - Imports: `datetime, pandas, plotly.express, plotly.graph_objects, streamlit`
      - Variables: `all_patient_ids, avg_improvement, display_df, fig, fig_age, fig_before_after, improvement, madrs_df, madrs_scores, madrs_scores_sorted, percent_improvement, protocol_counts, recent_patients, remission, responders, response_data, response_df, response_rate, selected_patient, total_analyzed, total_patients`
  📄 patient_journey.py
    *Analysis (AST):*
      - Functions: `patient_journey_page, summarize_effects`
      - Imports: `datetime, get_nurse_inputs_history, get_side_effects_history, logging, numpy, pandas, plotly.express, services.nurse_service, streamlit, timedelta`
      - Variables: `all_event_dfs, assessment_events_list, df_assessments, df_nurse, df_side_effects, event_type_map, fig, fu_date_approx, journey_df, madrs_bl, madrs_fu, nurse_history, other, patient_id, patient_main_data, s, side_effect_history, start_date, start_date_str, valid_dfs`
  📄 protocol_analysis.py
    *Analysis (AST):*
      - Functions: `protocol_analysis_page`
      - Imports: `numpy, pandas, plotly.express, plotly.graph_objects, streamlit`
      - Variables: `all_protocols, comparison_df, diff, diff_matrix, fig_box, fig_dist, fig_imp, fig_pie, fig_rates, fig_strip, madrs_df, mean1, mean2, means_pivot, num_cols, proto1, proto2, protocol_counts, protocol_metrics, rates_long, required_cols, selected_protocols, stats_df, valid_data_for_analysis`
  📄 side_effects.py
    *Analysis (AST):*
      - Functions: `side_effect_page`
      - Imports: `datetime, get_side_effects_history, os, pandas, plotly.express, save_side_effect_report, services.nurse_service, streamlit`
      - Variables: `columns_to_rename, date, date_col, display_columns, display_df, dizziness, effect_cols, fig, fig_max, headache, id_vars, nausea, notes, other, patient_side_effects, rename_map, report_data, scalp_discomfort, side_effect_long, submitted, success, summary, value_vars`
  📄 sidebar.py
    *Analysis (AST):*
      - Functions: `extract_number, render_sidebar`
      - Imports: `datetime, logging, re, streamlit`
      - Variables: `ROLE_PERMISSIONS, all_main_options, allowed_pages, available_options, current_page, current_selection, existing_patient_ids, match, patient_list, selected_index, selected_option, selected_patient, session_duration, user_role, view_count`
  📄 sign_out.py
    *Analysis (AST):*
      - Functions: `sign_out_page`
      - Imports: `logging, streamlit, time`
  📁 **common/**
    📄 charts.py
      *Analysis (AST):*
        - (No key elements found)
    📄 header.py
      *Analysis (AST):*
        - Functions: `app_header, create_card, patient_profile_header`
        - Imports: `base64, os, streamlit`
        - Variables: `columns, current_phase, phases, selected_tab, tab_options`
    📄 metrics.py
      *Analysis (AST):*
        - (No key elements found)
    📄 tables.py
      *Analysis (AST):*
        - (No key elements found)
📁 **data/**
  📄 .DS_Store
  📄 6.synthetic_data_synthpop.csv
  📄 dashboard_data.db
  📄 evaluation_cohort.pkl
  📄 extended_patient_data.csv
  📄 extended_patient_data_ml.csv
  📄 ml_training_data.csv
  📄 nurse_inputs.csv
  📄 patient_data_simulated.csv
  📄 patient_data_with_protocol_simulated.csv
  📄 response_prediction_model.pkl
  📄 selected_features.pkl
  📄 side_effects.csv
  📄 simulated_ema_data.csv
📁 **assets/**
  📄 styles.css
    *Analysis (Regex):*
      - Selectors: `.action-button, .action-button:hover, .app-header h1, .app-logo h3, .custom-card, .dashboard-card, .custom-card:hover, .dashboard-card-body, .dashboard-card-header, .dashboard-card-header h3, .dashboard-card:hover, .dataframe, .dataframe tbody tr:hover, .dataframe td, .dataframe th, .grid-item-1, .grid-item-12, .grid-item-2, .grid-item-3, .grid-item-4, .grid-item-6, .grid-item-8, .grid-item-9, .indicator-card.blue, .indicator-card.green, .indicator-card.orange, .indicator-card.red, .mb-16, .mb-24, .mb-32, .mt-24, .mt-32, .p-16, .p-24, .p-32, .patient-actions, .patient-demographics, .patient-demographics span, .patient-header-content, .patient-info h2, .patient-info-box h2, .phase-indicator.active .phase-label, .phase-indicator.active .phase-marker, .phase-indicator:last-child .phase-marker:after, .phase-label, .phase-marker, .phase-marker:after, .stError, .stInfo, .stRadio > div:hover, .stSuccess, .stTabs [aria-selected="true"], .stTabs [aria-selected="true"]:after, .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"]:hover, .stTabs [data-baseweb="tab-panel"], .stWarning, .stat-circle, .stat-circle .label, .stat-circle .value, .status-badge-danger, .status-badge-info, .status-badge-success, .status-badge-warning, .streamlit-expanderHeader:hover, .timeline-item, .timeline-item:before, .timeline:before, .user-profile, .user-profile i, /* Button Improvements - More depth and interactive */
button[data-testid="baseButton-primary"], /* CSS Grid System - 12 column */
.grid-container, /* Card styles - New modern card designs */
.dashboard-card, /* Color indicator cards */
.indicator-card, /* Custom Card Styling - Enhanced with more modern shadow */
.custom-card, /* Custom status badges - Enhanced */
.status-badge, /* Dark mode radio styling */
@media (prefers-color-scheme: dark), /* Dashboard Card Styling */
.dashboard-card, /* DataFrame (tables) styling - More modern */
[data-testid="stTable"], /* Ensures mobile-friendly layouts */
@media (max-width: 768px), /* Expander styling */
.streamlit-expanderHeader, /* Form Inputs - More modern styling */
[data-testid="stTextInput"], [data-testid="stSelectbox"], /* Global Layout */
.main .block-container, /* Header styling - More modern with shadow */
.app-header, /* Hide the original radio buttons */
div.row-widget.stRadio > div[role="radiogroup"] [data-testid="stRadio"], /* Info, Success, Warning Boxes - Enhanced with gradient backgrounds */
.stInfo, .stSuccess, .stWarning, .stError, /* Light mode radio styling */
@media (prefers-color-scheme: light), /* Metrics styling - Enhanced with gradient and better shadows */
[data-testid="stMetric"], /* Patient Header Styling */
.patient-header, /* Patient info box */
.patient-info-box, /* Progress bar styling - Taller and more visible */
[data-testid="stProgress"] > div > div, /* Radio Button Improvements - More tactile */
.stRadio > div, /* Radio Button Improvements for Navigation Tabs */
div.row-widget.stRadio > div, /* Selectbox Styling */
[data-testid="stSelectbox"] > div > div, /* Sidebar Styling - Modern gradient */
[data-testid="stSidebar"], /* Spacing utilities */
.mt-16, /* Stat circle */
.stat-circle, /* Tabs styling - More modern and tactile */
.stTabs [data-baseweb="tab-list"], /* Timeline styling */
.timeline, /* Treatment Phase Timeline */
.phase-indicator, /* Unified App Header */
.app-logo, /* assets/styles.css */

/* Base Typography */
body, [data-testid="stHorizontalBlock"] > div, [data-testid="stMetric"]:hover, [data-testid="stProgress"] > div, [data-testid="stSidebar"] h2, [data-testid="stTextInput"] > div > input, [data-testid="stTextInput"] > div > input:focus, button[data-testid="baseButton-primary"]:hover, div.row-widget.stRadio > div[role="radiogroup"] > label, div.row-widget.stRadio > div[role="radiogroup"] > label[aria-checked="true"], div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child, h1, h1, h2, h3, h4, h2, h3`
📁 **services/**
  📄 data_loader.py
    *Analysis (AST):*
      - Functions: `load_extended_patient_data, load_patient_data, load_simulated_ema_data, merge_simulated_data, merge_with_ml_data, validate_patient_data`
      - Imports: `handle_error, logging, pandas, utils.error_handler`
      - Variables: `base_col, data, error_msg, merged_df`
  📄 ml_service.py
    *Analysis (AST):*
      - Functions: `__init__, generate_evaluation_cohort, get_evaluation_cohort, get_feature_importance, get_model_performance, load_model, predict_response_probability, train_response_prediction_model`
      - Classs: `MLService`
      - Imports: `CalibratedClassifierCV, ColumnTransformer, ConfigManager, GradientBoostingClassifier, GridSearchCV, OneHotEncoder, Pipeline, RandomForestClassifier, SelectKBest, StandardScaler, ...`
      - Variables: `DEFAULT_FEATURE_COLUMNS, FN, FP, TN, TP, X, X_cohort, __all__, accuracy, base_model, best_model, categorical_features, classifier, cohort, cohort_accuracy, cohort_f1, cohort_precision, cohort_pred, cohort_proba, cohort_recall, cohort_roc_auc, confidence, config, confusion_matrix, data_path, df, empty_cols, f1, f1_t, feature_importances, feature_names_out, grid_search, importances, is_responder, logger, metrics, metrics_default, metrics_optimized, missing_cols, missing_features, n_non_responders, n_per_class, n_responders, non_responders, non_responders_sample, numeric_features, optimal_metrics, optimal_threshold, param_grid, patient_data, pipeline, precision, precision_t, preprocessed_X_train, preprocessor, recall, recall_t, required_cols, responders, responders_sample, response_prob, response_rate, result, roc_auc, saved_data, selected_features, selected_indices, selector, service, sorted_importances, threshold_metrics, thresholds, y, y_cohort, y_pred, y_pred_default, y_pred_optimized, y_pred_proba, y_pred_t, y_pred_threshold`
  📄 network_analysis.py
    *Analysis (AST):*
      - Functions: `construct_network, fit_multilevel_model, generate_person_specific_network, plot_network`
      - Imports: `mixedlm, networkx, numpy, pandas, plotly.graph_objects, statsmodels.formula.api, streamlit`
      - Variables: `G, coef, coef_matrix, coef_value, connections, df_model, df_patient, edge_text, edge_trace, edge_x, edge_y, fig, formula, lag_col, legend_text, model, node_adjacencies, node_text, node_trace, node_x, node_y, pos, predictors, result, weight`
  📄 nurse_service.py
    *Analysis (AST):*
      - Functions: `_add_column_if_not_exists, get_db, get_latest_nurse_inputs, get_nurse_inputs_history, get_side_effects_history, initialize_database, save_nurse_inputs, save_side_effect_report`
      - Imports: `Dict, List, Optional, logging, os, pandas, sqlite3, streamlit, typing`
      - Variables: `DATABASE_PATH, all_side_effects, combined_df, conn, csv_data, csv_path, cursor, db_data, db_success, df, existing_df, query, report_df, required_keys, result, row`
  📄 patient_service.py
    *Analysis (AST):*
      - (No key elements found)

## Project Summary

### Entrypoints

- `app.py`

### Config Files

- `config/config.yaml`

### Data Files

- `.devcontainer/devcontainer.json`
- `data/6.synthetic_data_synthpop.csv`
- `data/dashboard_data.db`
- `data/extended_patient_data.csv`
- `data/extended_patient_data_ml.csv`
- `data/ml_training_data.csv`
- `data/nurse_inputs.csv`
- `data/patient_data_simulated.csv`
- `data/patient_data_with_protocol_simulated.csv`
- `data/side_effects.csv`
- `data/simulated_ema_data.csv`

### Documentation

- `README.md`
- `requirements.txt`
- `structure.md`

### Utility Files

- `utils/logging_config.py`

### Project Complexity

- Total Files: 48
- Python Files: 24
- Data Files: 11


---
Analysis Complete.
