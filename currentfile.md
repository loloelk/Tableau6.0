# Codebase Structure Analysis: tms-dashboard-demo-main

Root Directory: `/Users/laurentelkrief/Desktop/Neuromod/Research/TableaudeBord/tms-dashboard-demo-main`

## Detected Frameworks & Libraries

- **pandas** (confidence: 23)
- **streamlit** (confidence: 23)
- **plotly** (confidence: 17)
- **numpy** (confidence: 10)

## File Structure

  📄 .DS_Store
  📄 .Rhistory
  📄 .gitignore
  📄 README.md
  📄 Scriptforstructure.py
    *Analysis (AST):*
      - Functions: `__init__, analyze_codebase, analyze_file_content_regex, analyze_python_ast, analyze_special_file, categorize_files, detect_frameworks, get_language_patterns_regex, visit_Assign, visit_AsyncFunctionDef, visit_Call, visit_ClassDef, visit_Decorator, visit_FunctionDef, visit_Import, visit_ImportFrom`
      - Classs: `PythonAstVisitor`
      - Imports: `argparse, ast, collections, defaultdict, json, os, re, sys, yaml`
      - Variables: `DEFAULT_ANALYZE_EXT, DEFAULT_ANALYZE_EXT_REGEX, DEFAULT_IGNORE_DIRS, EXT_TO_LANG, FRAMEWORK_PATTERNS, PATTERNS, PYTHON_EXT, SPECIAL_FILE_PATTERNS, all_files, analysis_indent, analysis_type, analyze_ext_py, analyze_ext_regex, args, categories, category_title, commands, content, data, dependencies, depth, detected_frameworks, devDependencies, display_items, empty_analysis, ext_lower, extracted, extracted_info, file_categories, file_indent, file_path, filename, final_analyze_ext_all, final_ignore_dirs, found_name, framework_matches, frameworks, indent, lang, matches, module_name, packages, parser, patterns, python_file_contents, relative_path, relative_root, root_dir, sorted_files, special_info, tree, visitor`
  📄 app.py
    *Analysis (AST):*
      - Functions: `check_login, nurse_inputs_page, run_db_initialization`
      - Imports: `components.dashboard, components.nurse_imputs, components.nurse_inputs, components.overview, components.patient_journey, components.protocol_analysis, components.side_effects, components.sidebar, configure_logging, datetime, ...`
      - Variables: `PATIENT_DATA_CSV, SIMULATED_EMA_CSV, VALID_CREDENTIALS, config, css_path, current_patient, entered_password, entered_username, final_data, keys_to_clear, login_button, needs_patient, page_selected, role_display, simulated_ema_data, user_info`
  📄 currentfile.md
  📄 dockerfile
    *Special File Analysis:*
      - Commands: `FROM, COPY, RUN, COPY, RUN, EXPOSE, ENV, ENTRYPOINT`
  📄 enhanced_simulate_patient_data.py
    *Analysis (AST):*
      - Functions: `distribute_phq9_score, generate_bfi_scores, generate_ema_data, generate_medication_data, generate_nurse_notes_data, generate_patient_data, generate_side_effects_data, initialize_database, save_nurse_inputs, save_side_effect_report`
      - Imports: `datetime, initialize_database, logging, math, numpy, os, pandas, random, save_nurse_inputs, save_side_effect_report, ...`
      - Variables: `ANXIETY_ITEMS, BFI_BASELINE_MEAN, BFI_BASELINE_STD, BFI_ITEMS_MAP, DB_INTERACTION_ENABLED, EMA_ENTRIES_PER_DAY_WEIGHTS, EMA_MISSING_DAY_PROB, EMA_MISSING_ENTRY_PROB, MADRS_ITEMS, MEDICATION_CATEGORIES, MEDICATION_DOSAGES, NEUROTICISM_CHANGE_NON_RESPONDER, NEUROTICISM_CHANGE_RESPONDER, NEUROTICISM_RESPONSE_FACTOR, NUM_PATIENTS, OTHER_FACTOR_CHANGE_STD, PROTOCOLS, PROTOCOL_REMISSION_RATES, PROTOCOL_RESPONSE_RATES, SIDE_EFFECT_DECAY_DAY, SIDE_EFFECT_PROB_INITIAL, SIDE_EFFECT_PROB_LATER, SIMULATION_DURATION_DAYS, START_DATE, SYMPTOMS, adj_bl, adj_fu, adjusted_response_prob, age, base_response_prob, base_start_date, baseline_item, baseline_mean, baseline_severity, bl_calc_score, bl_score, change, cols_to_drop, config_content, config_path, curr, current, current_severity, day_effect, day_imp, day_offset, days, delta, ect, el, ema_csv_path, ema_data_df, ema_entries, ema_entry, entry_severity, factor_scores_bl, factor_scores_fu, final_comment, final_day, final_note, final_status, followup_item, followup_mean, fu_calc_score, fu_score, hour, idx, improved, improvement, improvement_curve, improvement_factor, initial_day, initial_note, inv_n, inv_w, item_scores_bl_calc, item_scores_fu_calc, items, liste_comorbidites, log_file, lower_doses, madrs_bl, madrs_bl_sum, madrs_fu, madrs_fu_sum, madrs_improvement, med, med1, med2, medications, meds_formatted, mid_comment, mid_day, mid_note, mid_status, minute, n_entries_planned, n_items, n_rev, neuroticism_adjustment, noise_level, norm_w, num_comorbidities, num_meds, num_reports, num_saved, patient, patient_comorbidities_list, patient_csv_path, patient_data_df, patient_data_df_clean, patient_data_simple_csv_path, patient_id, patient_start_date, patients, phq9_bl, phq9_fu, phq9_items, possible_doses, primary_category, primary_dose, primary_med, prob_cutoff, progress, protocol, protocol_effect, prov_n_items_bl, prov_n_items_bl_calc, provisional_neuroticism_score_bl, psychotherapie, raw, remaining_categories, report_data, report_date, rtms, secondary_category, secondary_dose, secondary_med, severity, sex, stability, sum_inv, target_severity, tdcs, tertiary_categories, tertiary_category, tertiary_dose, tertiary_med, timestamp, weights, will_remit, will_respond`
  📄 requirements.txt
    *Special File Analysis:*
      - Packages: `streamlit, pandas, numpy, plotly, networkx, pyyaml, statsmodels, seaborn, python-dotenv, matplotlib`
📁 **config/**
  📄 config.yaml
    *Special File Analysis:*
      - Configuration: `paths, mappings`
📁 **utils/**
  📄 config_manager.py
    *Analysis (AST):*
      - Functions: `load_config`
      - Imports: `os, yaml`
      - Variables: `config_path, env`
  📄 error_handler.py
    *Analysis (AST):*
      - Functions: `handle_error`
      - Imports: `logging, streamlit`
  📄 logging_config.py
    *Analysis (AST):*
      - Functions: `configure_logging`
      - Imports: `datetime, logging, os`
      - Variables: `log_dir, log_file, today`
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
      - Variables: `MEDICATION_CATEGORIES, assumed_duration_days, author, available_categories, bfi_data_available, bfi_factors_map, bfi_table_data, bfi_table_df, bl_col, bl_score, bl_val, categories, category, cols, corr_matrix, count, created_by, csv, current_milestone_index, daily_score, daily_symptoms, days_elapsed, delta_score, display_columns, display_df_hist, dosage, exp_title, fig_bfi_radar, fig_ema_trends, fig_ema_variability, fig_heatmap, fig_items, fig_madrs_total, fig_network, fig_phq9, first_entry, fu_col, fu_score, fu_val, improvement_pct, is_remitter, is_responder, item_columns, item_label, items_data, last_entry, latest_note, latest_other, latest_plan, latest_report, madrs_bl, madrs_fu, madrs_items_df, madrs_items_long, madrs_total_df, max_sev, meds_data, meds_df, meds_list, milestones, name, notes_history_df, numeric_cols, numeric_ema_cols, parts, patient_data, patient_ema, patient_id, patient_main_df, patient_row, phq9_cols_exist, phq9_days, phq9_df, phq9_scores_over_time, plan_date, progress_percentage, rename_map, report_date, rolling_window, score, selected_category_avg, selected_category_corr, selected_category_var, selected_symptoms_avg, selected_symptoms_corr, selected_symptoms_var, severity, severity_cols, sex, sex_numeric, side_effects_history, summary_list, symptom_categories, symptoms_available, symptoms_present, threshold, valid_items_found, values_bl, values_fu, variability_df`
  📄 nurse_inputs.py
    *Analysis (AST):*
      - Functions: `nurse_inputs_page`
      - Imports: `get_latest_nurse_inputs, get_nurse_inputs_history, pandas, save_nurse_inputs, services.nurse_service, streamlit`
      - Variables: `GOAL_STATUS_OPTIONS, comments_input, current_status, display_columns, display_df, expander_title, goal_status_input, history_df, latest_inputs, objectives_input, patient_id, planned_interventions_input, rename_map, status_index, submit_button, success, target_symptoms_input, tasks_input`
  📄 overview.py
    *Analysis (AST):*
      - Functions: `main_dashboard_page`
      - Imports: `pandas, plotly.express, plotly.graph_objects, streamlit`
      - Variables: `all_patient_ids, avg_improvement, display_df, fig, fig_age, fig_before_after, fig_improvement, improvement, madrs_df, madrs_scores, madrs_scores_sorted, percent_improvement, protocol_counts, recent_patients, response_rate, selected_patient, total_patients`
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
  📁 **common/**
    📄 charts.py
      *Analysis (AST):*
        - (No key elements found)
    📄 metrics.py
      *Analysis (AST):*
        - (No key elements found)
    📄 tables.py
      *Analysis (AST):*
        - (No key elements found)
📁 **data/**
  📄 dashboard_data.db
  📄 ml_training_data.csv
  📄 nurse_inputs.csv
  📄 patient_data_simulated.csv
  📄 patient_data_with_protocol_simulated.csv
  📄 side_effects.csv
  📄 simulated_ema_data.csv
📁 **assets/**
  📄 styles.css
    *Analysis (Regex):*
      - Selectors: `.dataframe, .dataframe td, .dataframe th, .stError, .stInfo, .stRadio > div:hover, .stSuccess, .stTabs [data-baseweb="tab"], .stWarning, /* Button Improvements */
button[data-testid="baseButton-primary"], /* Dark mode radio styling */
@media (prefers-color-scheme: dark), /* DataFrame (tables) styling */
[data-testid="stTable"], /* Ensures mobile-friendly layouts */
@media (max-width: 768px), /* Expander styling */
.streamlit-expanderHeader, /* Form Inputs */
[data-testid="stTextInput"], [data-testid="stSelectbox"], /* Info, Success, Warning Boxes */
.stInfo, .stSuccess, .stWarning, .stError, /* Light mode radio styling */
@media (prefers-color-scheme: light), /* Metrics styling */
[data-testid="stMetric"], /* Radio Button Improvements */
.stRadio > div, /* Selectbox Styling */
[data-testid="stSelectbox"] > div > div, /* Sidebar Styling */
[data-testid="stSidebar"], /* Tabs styling */
.stTabs [data-baseweb="tab-list"], /* assets/styles.css */

/* Base Typography */
body, [data-testid="stHorizontalBlock"] > div, [data-testid="stMetric"]:hover, button[data-testid="baseButton-primary"]:hover, h1, h2, h3, h4`
📁 **services/**
  📄 data_loader.py
    *Analysis (AST):*
      - Functions: `load_patient_data, load_simulated_ema_data, merge_simulated_data, validate_patient_data`
      - Imports: `handle_error, logging, pandas, utils.error_handler`
      - Variables: `data, error_msg, merged_df`
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
- `data/dashboard_data.db`
- `data/ml_training_data.csv`
- `data/nurse_inputs.csv`
- `data/patient_data_simulated.csv`
- `data/patient_data_with_protocol_simulated.csv`
- `data/side_effects.csv`
- `data/simulated_ema_data.csv`

### Documentation

- `README.md`
- `currentfile.md`
- `requirements.txt`

### Utility Files

- `utils/logging_config.py`

### Project Complexity

- Total Files: 38
- Python Files: 21
- Data Files: 8


---
Analysis Complete.
