"""
Preprocessing pipeline for NASA C-MAPSS dataset
Removes constant sensors, normalizes data, and prepares for modeling

FIXES APPLIED:
  1. add_rolling_features and add_rate_of_change now applied to BOTH
     train_df and test_df in run_pipeline. Previously only train got
     engineered features → train had 102 features, test had 11 → mismatch.
  2. max_rul is now hardcoded to 130 (the clip ceiling) instead of
     np.max(y_train). np.max(y_train) also returns 130 after clipping,
     but being explicit avoids any edge-case where a split has a different
     max and the wrong scale gets saved.
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
        self.dataset = dataset
        self.data_dir = data_dir
        self.scalers = {}

        self.index_names   = ['engine_id', 'cycle']
        self.setting_names = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_names  = [f'sensor_{i}' for i in range(1, 22)]
        self.col_names     = self.index_names + self.setting_names + self.sensor_names

        self.constant_sensors = [
            'sensor_1', 'sensor_5', 'sensor_6', 'sensor_8', 'sensor_10',
            'sensor_13', 'sensor_15', 'sensor_16', 'sensor_18', 'sensor_19'
        ]

        self.useful_sensors = [
            'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_9',
            'sensor_11', 'sensor_12', 'sensor_14', 'sensor_17',
            'sensor_20', 'sensor_21'
        ]

        print(f"✓ Preprocessor initialized for {dataset}")
        print(f"  Constant sensors to remove : {len(self.constant_sensors)}")
        print(f"  Useful sensors to keep     : {len(self.useful_sensors)}")

    # ── Data loading ───────────────────────────────────────────────────────

    def load_data(self):
        train_file = os.path.join(self.data_dir, f'train_{self.dataset}.txt')
        test_file  = os.path.join(self.data_dir, f'test_{self.dataset}.txt')
        rul_file   = os.path.join(self.data_dir, f'RUL_{self.dataset}.txt')

        self.train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=self.col_names)
        self.test_df  = pd.read_csv(test_file,  sep=r'\s+', header=None, names=self.col_names)
        self.rul_df   = pd.read_csv(rul_file,   sep=r'\s+', header=None, names=['RUL'])

        print(f"\n✓ Data loaded:")
        print(f"  Training : {self.train_df.shape}")
        print(f"  Test     : {self.test_df.shape}")
        print(f"  RUL labels: {self.rul_df.shape}")

        return self.train_df, self.test_df, self.rul_df

    # ── RUL ────────────────────────────────────────────────────────────────

    def add_rul(self, df):
        max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycle']
        df = df.merge(max_cycles, on='engine_id', how='left')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df = df.drop('max_cycle', axis=1)
        return df

    def clip_rul(self, df, max_rul=130):
        df['RUL_clipped'] = df['RUL'].clip(upper=max_rul)
        print(f"\n✓ Clipped RUL at {max_rul} cycles")
        return df

    # ── Sensors ────────────────────────────────────────────────────────────

    def remove_constant_sensors(self, df):
        df = df.drop(columns=self.constant_sensors)
        print(f"\n✓ Removed {len(self.constant_sensors)} constant sensors")
        return df

    def normalize_sensors(self, train_df, test_df):
        cols_to_normalize = self.setting_names + self.useful_sensors
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df[cols_to_normalize] = scaler.fit_transform(train_df[cols_to_normalize])
        test_df[cols_to_normalize]  = scaler.transform(test_df[cols_to_normalize])
        self.scalers['sensor_scaler'] = scaler
        print(f"\n✓ Normalized {len(cols_to_normalize)} columns to [0, 1]")
        return train_df, test_df

    # ── Feature engineering ────────────────────────────────────────────────

    def add_rolling_features(self, df, windows=[5, 10, 20]):
        """Rolling mean and std per sensor per engine."""
        print(f"\n✓ Adding rolling features...")
        df = df.sort_values(['engine_id', 'cycle'])
        feature_count = 0
        for sensor in self.useful_sensors:
            for window in windows:
                df[f'{sensor}_rolling_mean_{window}'] = (
                    df.groupby('engine_id')[sensor]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
                df[f'{sensor}_rolling_std_{window}'] = (
                    df.groupby('engine_id')[sensor]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).std())
                    .fillna(0)
                )
                feature_count += 2
        print(f"  Added {feature_count} rolling features")
        return df

    def add_rate_of_change(self, df):
        """First and second derivative per sensor per engine."""
        print(f"\n✓ Adding rate-of-change features...")
        feature_count = 0
        for sensor in self.useful_sensors:
            df[f'{sensor}_velocity']     = df.groupby('engine_id')[sensor].diff().fillna(0)
            df[f'{sensor}_acceleration'] = (
                df.groupby('engine_id')[f'{sensor}_velocity'].diff().fillna(0)
            )
            feature_count += 2
        print(f"  Added {feature_count} rate-of-change features")
        return df

    # ── Sequences ──────────────────────────────────────────────────────────

    def create_sequences(self, df, sequence_length=30, stride=1):
        print(f"\n✓ Creating sequences...")
        print(f"  Sequence length : {sequence_length}")
        print(f"  Stride          : {stride}")

        feature_cols = [
            col for col in df.columns
            if col not in ['engine_id', 'cycle', 'RUL', 'RUL_clipped']
        ]

        X_list, y_list, engine_id_list = [], [], []

        for engine_id in df['engine_id'].unique():
            engine_data = df[df['engine_id'] == engine_id]
            features    = engine_data[feature_cols].values
            rul         = engine_data['RUL_clipped'].values

            for i in range(0, len(features) - sequence_length + 1, stride):
                X_list.append(features[i:i + sequence_length])
                y_list.append(rul[i + sequence_length - 1])
                engine_id_list.append(engine_id)

        X          = np.array(X_list)
        y          = np.array(y_list)
        engine_ids = np.array(engine_id_list)

        print(f"  Created {len(X)} sequences")
        print(f"  Shape  : X={X.shape}, y={y.shape}")

        return X, y, engine_ids, feature_cols

    # ── Train / val / test split ───────────────────────────────────────────

    def split_data(self, X, y, engine_ids, train_ratio=0.7, val_ratio=0.15):
        print(f"\n✓ Splitting data by engine...")

        unique_engines = np.unique(engine_ids)
        n_engines      = len(unique_engines)

        np.random.seed(42)
        shuffled_engines = np.random.permutation(unique_engines)

        n_train = int(n_engines * train_ratio)
        n_val   = int(n_engines * val_ratio)

        train_engines = shuffled_engines[:n_train]
        val_engines   = shuffled_engines[n_train:n_train + n_val]
        test_engines  = shuffled_engines[n_train + n_val:]

        train_mask = np.isin(engine_ids, train_engines)
        val_mask   = np.isin(engine_ids, val_engines)
        test_mask  = np.isin(engine_ids, test_engines)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        # FIX: use the hard clip ceiling (130) as max_rul, not np.max(y_train).
        # np.max(y_train) also equals 130 after clipping, but being explicit
        # guarantees the saved scalar always matches what the model is trained on.
        max_rul = 130.0

        y_train = y_train / max_rul   # → [0, 1]
        y_val   = y_val   / max_rul
        y_test  = y_test  / max_rul

        print(f"  Train  : {len(train_engines)} engines, {len(X_train)} sequences")
        print(f"  Val    : {len(val_engines)}   engines, {len(X_val)}   sequences")
        print(f"  Test   : {len(test_engines)}  engines, {len(X_test)}  sequences")
        print(f"  max_rul: {max_rul}")
        print(f"  y_train range after normalisation: "
              f"[{y_train.min():.4f}, {y_train.max():.4f}]")

        return {
            'X_train': X_train, 'y_train': y_train,
            'train_engines': train_engines, 'train_engine_ids': engine_ids[train_mask],

            'X_val':   X_val,   'y_val':   y_val,
            'val_engines':   val_engines,   'val_engine_ids':   engine_ids[val_mask],

            'X_test':  X_test,  'y_test':  y_test,
            'test_engines':  test_engines,  'test_engine_ids':  engine_ids[test_mask],

            'max_rul': max_rul,
        }

    # ── Save ───────────────────────────────────────────────────────────────

    def save_preprocessed_data(self, data_dict, feature_cols, output_dir='data/processed'):
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{self.dataset}_preprocessed.npz')

        np.savez_compressed(
            output_file,
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            train_engines=data_dict['train_engines'],
            train_engine_ids=data_dict['train_engine_ids'],

            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            val_engines=data_dict['val_engines'],
            val_engine_ids=data_dict['val_engine_ids'],

            X_test=data_dict['X_test'],
            y_test=data_dict['y_test'],
            test_engines=data_dict['test_engines'],
            test_engine_ids=data_dict['test_engine_ids'],

            max_rul=data_dict['max_rul'],
            feature_names=feature_cols,
        )

        scaler_file = os.path.join(output_dir, f'{self.dataset}_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scalers, f)

        print(f"\n✓ Saved preprocessed data → {output_file}")
        print(f"✓ Saved scaler            → {scaler_file}")

    # ── Full pipeline ──────────────────────────────────────────────────────

    def run_pipeline(self, sequence_length=30, add_features=True):
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)

        # 1. Load
        train_df, test_df, rul_df = self.load_data()

        # 2. RUL for train only (test RUL comes from rul_df, handled separately)
        train_df = self.add_rul(train_df)
        print(f"\n✓ Added RUL column to training data")

        # 3. Remove constant sensors (both)
        train_df = self.remove_constant_sensors(train_df)
        test_df  = self.remove_constant_sensors(test_df)

        # 4. Normalise (fit on train, transform both)
        train_df, test_df = self.normalize_sensors(train_df, test_df)

        # 5. Feature engineering — FIX: apply to BOTH train and test
        if add_features:
            print("\nApplying feature engineering to train data...")
            train_df = self.add_rolling_features(train_df)
            train_df = self.add_rate_of_change(train_df)

            print("\nApplying feature engineering to test data...")
            test_df  = self.add_rolling_features(test_df)   # ← was missing
            test_df  = self.add_rate_of_change(test_df)     # ← was missing

        # 6. Clip RUL
        train_df = self.clip_rul(train_df)

        # 7. Create sequences (train only — test uses NASA RUL labels)
        X, y, engine_ids, feature_cols = self.create_sequences(
            train_df, sequence_length=sequence_length
        )

        # 8. Split
        data_dict = self.split_data(X, y, engine_ids)
        data_dict['feature_names'] = feature_cols

        # 9. Save
        self.save_preprocessed_data(data_dict, feature_cols)

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)

        return data_dict, feature_cols


if __name__ == '__main__':
    preprocessor = CMAPSSPreprocessor(dataset='FD001', data_dir='data/raw')
    data_dict, feature_cols = preprocessor.run_pipeline(
        sequence_length=30,
        add_features=True
    )

    print("\n" + "=" * 60)
    print("FINAL DATA SUMMARY")
    print("=" * 60)
    print(f"Training samples  : {len(data_dict['X_train'])}")
    print(f"Validation samples: {len(data_dict['X_val'])}")
    print(f"Test samples      : {len(data_dict['X_test'])}")
    print(f"Sequence shape    : {data_dict['X_train'].shape}")
    print(f"Features          : {data_dict['X_train'].shape[2]}")
    print(f"y_train range     : [{data_dict['y_train'].min():.4f}, "
          f"{data_dict['y_train'].max():.4f}]")
    print(f"max_rul           : {data_dict['max_rul']}")
    print("=" * 60)