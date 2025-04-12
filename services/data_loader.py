# services/data_loader.py
import pandas as pd
import logging
from utils.error_handler import handle_error

def load_patient_data(csv_file: str) -> pd.DataFrame:
    """
    Load patient data from a CSV file with appropriate encoding.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing patient data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing patient data
    """
    try:
        data = pd.read_csv(csv_file, dtype={'ID': str}, encoding='utf-8')
        logging.debug(f"Patient data loaded successfully from {csv_file} with 'utf-8' encoding.")
        return data
    except UnicodeDecodeError:
        logging.warning(f"UnicodeDecodeError with 'utf-8' encoding for {csv_file}. Trying 'latin1'.")
        try:
            data = pd.read_csv(csv_file, dtype={'ID': str}, encoding='latin1')
            logging.debug(f"Patient data loaded successfully from {csv_file} with 'latin1' encoding.")
            return data
        except Exception as e:
            logging.error(f"Failed to load patient data from {csv_file}: {e}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load patient data from {csv_file}: {e}")
        return pd.DataFrame()

def validate_patient_data(data: pd.DataFrame):
    """
    Validate the structure and content of patient data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing patient data to validate
        
    Raises:
    -------
    ValueError
        If data validation fails
    """
    if 'ID' not in data.columns:
        error_msg = "La colonne 'ID' est manquante dans le fichier CSV des patients."
        logging.error("The 'ID' column is missing in the patient data.")
        raise ValueError(error_msg)

    if data['ID'].isnull().any():
        error_msg = "Certaines entrées de la colonne 'ID' sont vides. Veuillez les remplir."
        logging.error("There are empty entries in the 'ID' column.")
        raise ValueError(error_msg)

    if data['ID'].duplicated().any():
        error_msg = "Il y a des IDs dupliqués dans la colonne 'ID'. Veuillez assurer l'unicité."
        logging.error("There are duplicate IDs in the 'ID' column.")
        raise ValueError(error_msg)

    logging.debug("Patient data validation passed.")

def load_simulated_ema_data(csv_file: str) -> pd.DataFrame:
    """
    Load simulated EMA data from a CSV file with appropriate encoding.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing EMA data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing EMA data
    """
    try:
        data = pd.read_csv(csv_file, dtype={'PatientID': str}, encoding='utf-8')
        logging.debug(f"Simulated EMA data loaded successfully from {csv_file} with 'utf-8' encoding.")
        return data
    except UnicodeDecodeError:
        logging.warning(f"UnicodeDecodeError with 'utf-8' encoding for {csv_file}. Trying 'latin1'.")
        try:
            data = pd.read_csv(csv_file, dtype={'PatientID': str}, encoding='latin1')
            logging.debug(f"Simulated EMA data loaded successfully from {csv_file} with 'latin1' encoding.")
            return data
        except Exception as e:
            logging.error(f"Failed to load simulated EMA data from {csv_file}: {e}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load simulated EMA data from {csv_file}: {e}")
        return pd.DataFrame()

def merge_simulated_data(final_df: pd.DataFrame, ema_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge simulated EMA data with the final patient data.
    
    Parameters:
    -----------
    final_df : pd.DataFrame
        DataFrame containing patient data
    ema_df : pd.DataFrame
        DataFrame containing EMA data
        
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame
    """
    if ema_df.empty:
        logging.warning("Simulated EMA data is empty. Skipping merge.")
        return final_df
    
    # Assuming 'ID' in final_data corresponds to 'PatientID' in simulated_ema_data
    merged_df = final_df.merge(ema_df, how='left', left_on='ID', right_on='PatientID')
    return merged_df

def load_extended_patient_data(csv_file: str) -> pd.DataFrame:
    """
    Load extended patient data with features for ML predictions.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing extended patient data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing extended patient data
    """
    try:
        data = pd.read_csv(csv_file, dtype={'ID': str}, encoding='utf-8')
        logging.debug(f"Extended patient data loaded successfully from {csv_file} with 'utf-8' encoding.")
        return data
    except UnicodeDecodeError:
        logging.warning(f"UnicodeDecodeError with 'utf-8' encoding for {csv_file}. Trying 'latin1'.")
        try:
            data = pd.read_csv(csv_file, dtype={'ID': str}, encoding='latin1')
            logging.debug(f"Extended patient data loaded successfully from {csv_file} with 'latin1' encoding.")
            return data
        except Exception as e:
            logging.error(f"Failed to load extended patient data from {csv_file}: {e}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load extended patient data from {csv_file}: {e}")
        return pd.DataFrame()

def merge_with_ml_data(main_df: pd.DataFrame, extended_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the main patient data with extended ML features.
    
    Parameters:
    -----------
    main_df : pd.DataFrame
        DataFrame containing main patient data
    extended_df : pd.DataFrame
        DataFrame containing extended ML features
        
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with all features
    """
    if extended_df.empty:
        logging.warning("Extended ML data is empty. Skipping merge.")
        return main_df
    
    try:
        # Merge on patient ID
        merged_df = main_df.merge(extended_df, how='left', on='ID', suffixes=('', '_ext'))
        
        # For duplicate columns, keep the original if not null, otherwise take from extended
        for col in merged_df.columns:
            if col.endswith('_ext'):
                base_col = col[:-4]  # Remove '_ext' suffix
                if base_col in merged_df.columns:
                    # Fill NaN values in original column with values from extended column
                    merged_df[base_col] = merged_df[base_col].fillna(merged_df[col])
                    # Drop the extended column
                    merged_df = merged_df.drop(col, axis=1)
        
        logging.debug(f"Merged main data with extended ML data successfully. Final shape: {merged_df.shape}")
        return merged_df
    
    except Exception as e:
        logging.error(f"Failed to merge with extended ML data: {e}")
        return main_df