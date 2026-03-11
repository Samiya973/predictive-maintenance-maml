"""
Preprocessing pipeline for NASA C-MAPSS dataset
Removes constant sensors, normalizes data, and prepares for modeling
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

class CMAPSSPreprocessor:
    """
    Preprocessor for NASA C-MAPSS turbofan engine dataset
    """
    
    def __init__(self, dataset='FD001', data_dir='data/raw'):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        dataset : str
            Dataset name (e.g., 'FD001')
        data_dir : str
            Directory containing raw data
        """
        self.dataset = dataset
        self.data_dir = data_dir
        self.scalers = {}
        
        # Define column names
        self.index_names = ['engine_id', 'cycle']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names = [f'sensor_{i}' for i in range(1, 22)]
        self.col_names = self.index_names + self.setting_names + self.sensor_names
        
        # Constant sensors to remove (from EDA)
        self.constant_sensors = [
            'sensor_1', 'sensor_5', 'sensor_6', 'sensor_8', 'sensor_10',
            'sensor_13', 'sensor_15', 'sensor_16', 'sensor_18', 'sensor_19'
        ]
        
        # Useful sensors to keep
        self.useful_sensors = [
            'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_9',
            'sensor_11', 'sensor_12', 'sensor_14', 'sensor_17', 
            'sensor_20', 'sensor_21'
        ]
        
        print(f"✓ Preprocessor initialized for {dataset}")
        print(f"  Constant sensors to remove: {len(self.constant_sensors)}")
        print(f"  Useful sensors to keep: {len(self.useful_sensors)}")
    
    def load_data(self):
        """Load raw data from files"""
        # File paths
        train_file = os.path.join(self.data_dir, f'train_{self.dataset}.txt')
        test_file = os.path.join(self.data_dir, f'test_{self.dataset}.txt')
        rul_file = os.path.join(self.data_dir, f'RUL_{self.dataset}.txt')
        
        # Load training data
        self.train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=self.col_names)
        
        # Load test data
        self.test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=self.col_names)
        
        # Load RUL labels
        self.rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
        
        print(f"\n✓ Data loaded:")
        print(f"  Training: {self.train_df.shape}")
        print(f"  Test: {self.test_df.shape}")
        print(f"  RUL labels: {self.rul_df.shape}")
        
        return self.train_df, self.test_df, self.rul_df
    
    def add_rul(self, df):
        """
        Add RUL (Remaining Useful Life) column
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with RUL column added
        """
        # Get max cycle for each engine
        max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycle']
        
        # Merge and calculate RUL
        df = df.merge(max_cycles, on='engine_id', how='left')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df = df.drop('max_cycle', axis=1)
        
        return df
    
    def remove_constant_sensors(self, df):
        """
        Remove constant sensors identified in EDA
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with constant sensors removed
        """
        df = df.drop(columns=self.constant_sensors)
        print(f"\n✓ Removed {len(self.constant_sensors)} constant sensors")
        print(f"  Remaining sensors: {len(self.useful_sensors)}")
        return df
    
    def normalize_sensors(self, train_df, test_df):
        """
        Normalize sensor values using Min-Max scaling
        Fit on training data, apply to both train and test
        
        Parameters:
        -----------
        train_df : pd.DataFrame
            Training dataframe
        test_df : pd.DataFrame
            Test dataframe
            
        Returns:
        --------
        train_df, test_df : pd.DataFrame
            Normalized dataframes
        """
        # Columns to normalize: settings + useful sensors
        cols_to_normalize = self.setting_names + self.useful_sensors
        
        # Fit scaler on training data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize])
        
        # Apply same scaling to test data
        test_df[cols_to_normalize] = scaler.transform(test_df[cols_to_normalize])
        
        # Save scaler for later use
        self.scalers['sensor_scaler'] = scaler
        
        print(f"\n✓ Normalized {len(cols_to_normalize)} columns")
        print(f"  Range: [0, 1]")
        
        return train_df, test_df
    
    def add_rolling_features(self, df, windows=[5, 10, 20]):
        """
        Add rolling statistics features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        windows : list
            Window sizes for rolling statistics
            
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with rolling features added
        """
        print(f"\n✓ Adding rolling features...")
        
        # Sort by engine_id and cycle for proper rolling calculation
        df = df.sort_values(['engine_id', 'cycle'])
        
        feature_count = 0
        for sensor in self.useful_sensors:
            for window in windows:
                # Rolling mean
                df[f'{sensor}_rolling_mean_{window}'] = df.groupby('engine_id')[sensor].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std
                df[f'{sensor}_rolling_std_{window}'] = df.groupby('engine_id')[sensor].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ).fillna(0)
                
                feature_count += 2
        
        print(f"  Added {feature_count} rolling features")
        return df
    
    def add_rate_of_change(self, df):
        """
        Add rate of change (velocity) and acceleration features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with velocity and acceleration features
        """
        print(f"\n✓ Adding rate of change features...")
        
        feature_count = 0
        for sensor in self.useful_sensors:
            # Velocity (first derivative)
            df[f'{sensor}_velocity'] = df.groupby('engine_id')[sensor].diff().fillna(0)
            
            # Acceleration (second derivative)
            df[f'{sensor}_acceleration'] = df.groupby('engine_id')[f'{sensor}_velocity'].diff().fillna(0)
            
            feature_count += 2
        
        print(f"  Added {feature_count} rate of change features")
        return df
    
    def clip_rul(self, df, max_rul=130):
        """
        Clip RUL values (common practice in literature)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        max_rul : int
            Maximum RUL value
            
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with clipped RUL
        """
        df['RUL_clipped'] = df['RUL'].clip(upper=max_rul)
        print(f"\n✓ Clipped RUL at {max_rul} cycles")
        return df
    
    def create_sequences(self, df, sequence_length=30, stride=1):
        """
        Create sequences for LSTM/CNN input
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        sequence_length : int
            Length of each sequence (timesteps)
        stride : int
            Step size between sequences
            
        Returns:
        --------
        X : np.array
            Feature sequences [samples, timesteps, features]
        y : np.array
            Target values [samples]
        engine_ids : np.array
            Engine ID for each sequence
        """
        print(f"\n✓ Creating sequences...")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Stride: {stride}")
        
        # Get feature columns (exclude id, cycle, RUL)
        feature_cols = [col for col in df.columns if col not in ['engine_id', 'cycle', 'RUL', 'RUL_clipped']]
        
        X_list = []
        y_list = []
        engine_id_list = []
        
        # Create sequences for each engine
        for engine_id in df['engine_id'].unique():
            engine_data = df[df['engine_id'] == engine_id]
            
            # Extract features and target
            features = engine_data[feature_cols].values
            rul = engine_data['RUL_clipped'].values
            
            # Create overlapping sequences
            for i in range(0, len(features) - sequence_length + 1, stride):
                X_list.append(features[i:i+sequence_length])
                y_list.append(rul[i+sequence_length-1])  # Target is RUL at last timestep
                engine_id_list.append(engine_id)
        
        X = np.array(X_list)
        y = np.array(y_list)
        engine_ids = np.array(engine_id_list)
        
        print(f"  Created {len(X)} sequences")
        print(f"  Shape: X={X.shape}, y={y.shape}")
        
        return X, y, engine_ids, feature_cols
    
    def split_data(self, X, y, engine_ids, train_ratio=0.7, val_ratio=0.15):
        """
        Split data by engines (not by sequences)
        
        Parameters:
        -----------
        X : np.array
            Feature sequences
        y : np.array
            Target values
        engine_ids : np.array
            Engine IDs
        train_ratio : float
            Proportion of engines for training
        val_ratio : float
            Proportion of engines for validation
            
        Returns:
        --------
        Dictionary containing train/val/test splits
        """
        print(f"\n✓ Splitting data...")
        
        # Get unique engines
        unique_engines = np.unique(engine_ids)
        n_engines = len(unique_engines)
        
        # Shuffle engines
        np.random.seed(42)
        shuffled_engines = np.random.permutation(unique_engines)
        
        # Calculate splits
        n_train = int(n_engines * train_ratio)
        n_val = int(n_engines * val_ratio)
        
        train_engines = shuffled_engines[:n_train]
        val_engines = shuffled_engines[n_train:n_train+n_val]
        test_engines = shuffled_engines[n_train+n_val:]
        
        # Create masks
        train_mask = np.isin(engine_ids, train_engines)
        val_mask = np.isin(engine_ids, val_engines)
        test_mask = np.isin(engine_ids, test_engines)
        
        # Split data
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"  Train engines: {len(train_engines)} ({n_train} engines, {len(X_train)} sequences)")
        print(f"  Val engines: {len(val_engines)} ({n_val} engines, {len(X_val)} sequences)")
        print(f"  Test engines: {len(test_engines)} ({len(test_engines)} engines, {len(X_test)} sequences)")
        
        return {
            'X_train': X_train, 'y_train': y_train, 'train_engines': train_engines,
            'X_val': X_val, 'y_val': y_val, 'val_engines': val_engines,
            'X_test': X_test, 'y_test': y_test, 'test_engines': test_engines
        }
    
    def save_preprocessed_data(self, data_dict, feature_cols, output_dir='data/processed'):
        """
        Save preprocessed data
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing all splits
        feature_cols : list
            List of feature column names
        output_dir : str
            Output directory
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        output_file = os.path.join(output_dir, f'{self.dataset}_preprocessed.npz')
        np.savez_compressed(
            output_file,
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            train_engines=data_dict['train_engines'],
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            val_engines=data_dict['val_engines'],
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test'],
            test_engines=data_dict['test_engines'],
            feature_names=feature_cols
        )
        
        # Save scaler
        scaler_file = os.path.join(output_dir, f'{self.dataset}_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        print(f"\n✓ Saved preprocessed data:")
        print(f"  Data: {output_file}")
        print(f"  Scaler: {scaler_file}")
    
    def run_pipeline(self, sequence_length=30, add_features=True):
        """
        Run complete preprocessing pipeline
        
        Parameters:
        -----------
        sequence_length : int
            Sequence length for LSTM
        add_features : bool
            Whether to add engineered features
            
        Returns:
        --------
        data_dict : dict
            Dictionary containing all preprocessed data
        """
        print("="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        train_df, test_df, rul_df = self.load_data()
        
        # Step 2: Add RUL
        train_df = self.add_rul(train_df)
        print(f"\n✓ Added RUL column to training data")
        
        # Step 3: Remove constant sensors
        train_df = self.remove_constant_sensors(train_df)
        test_df = self.remove_constant_sensors(test_df)
        
        # Step 4: Normalize
        train_df, test_df = self.normalize_sensors(train_df, test_df)
        
        # Step 5: Add engineered features (optional)
        if add_features:
            train_df = self.add_rolling_features(train_df)
            train_df = self.add_rate_of_change(train_df)
        
        # Step 6: Clip RUL
        train_df = self.clip_rul(train_df)
        
        # Step 7: Create sequences
        X, y, engine_ids, feature_cols = self.create_sequences(train_df, sequence_length=sequence_length)
        
        # Step 8: Split data
        data_dict = self.split_data(X, y, engine_ids)
        
        # Step 9: Save
        self.save_preprocessed_data(data_dict, feature_cols)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        
        return data_dict, feature_cols


if __name__ == '__main__':
    # Run preprocessing
    preprocessor = CMAPSSPreprocessor(dataset='FD001', data_dir='data/raw')
    data_dict, feature_cols = preprocessor.run_pipeline(
        sequence_length=30,
        add_features=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL DATA SUMMARY")
    print("="*60)
    print(f"Training samples: {len(data_dict['X_train'])}")
    print(f"Validation samples: {len(data_dict['X_val'])}")
    print(f"Test samples: {len(data_dict['X_test'])}")
    print(f"Sequence shape: {data_dict['X_train'].shape}")
    print(f"Number of features: {data_dict['X_train'].shape[2]}")
    print(f"Feature names: {len(feature_cols)} features")
    print("="*60)