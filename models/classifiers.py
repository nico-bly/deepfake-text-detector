import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import copy

from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

import xgboost as xgb

from scipy.stats import entropy

import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class NeuralClassifier(nn.Module):
    def __init__(self, input_dim=4096, hidden_dims=[1024, 512], dropout_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers.append(nn.Sigmoid())
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepBinaryDetector:
    def __init__(self, input_dim=4096, hidden_dims=[1024, 512], device='cuda'):
        self.device = device
        self.model = NeuralClassifier(input_dim, hidden_dims).to(device)
        self.scaler = StandardScaler()
        self._is_fitted = False
        
    def fit(self, embeddings, labels, validation_split=0.2, 
        epochs=10, batch_size=32, lr=0.001):
        # Preprocess
        embeddings = self.scaler.fit_transform(embeddings)
        # Move data to CPU before splitting
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        X = torch.FloatTensor(embeddings)
        y = torch.FloatTensor(labels)
        
        # Train/val split on CPU
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X.cpu().numpy(), y.cpu().numpy(), 
                test_size=validation_split, 
                stratify=y.cpu().numpy()
            )
            # Convert back to tensors and move to device
            X_train = torch.FloatTensor(X_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
        else:
            X_train, y_train = X.to(self.device), y.to(self.device)
            X_val, y_val = None, None
            
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train.view(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = TensorDataset(X_val, y_val.view(-1, 1))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss = float('inf')
        best_model = None
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if X_val is not None:
                self.model.eval()
                val_loss = 0
                val_preds = []
                val_true = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        val_loss += criterion(outputs, batch_y).item()
                        val_preds.extend((outputs > 0.5).cpu().numpy())
                        val_true.extend(batch_y.cpu().numpy())
                
                val_acc = accuracy_score(val_true, val_preds)
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(self.model.state_dict())
            
        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        self._is_fitted = True
        return {'val_accuracy': val_acc if X_val is not None else None}
    
    def predict(self, embeddings, return_probabilities=True):
        """
        Predict using the trained neural network
        Args:
            embeddings: numpy array of embeddings
            return_probabilities: whether to return probabilities along with predictions
        Returns:
            predictions: binary predictions (0=real, 1=fake)
            probabilities: probability of being fake (if return_probabilities=True)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted yet!")
            
        # Preprocess embeddings
        embeddings = self.scaler.transform(embeddings)
        X = torch.FloatTensor(embeddings).to(self.device)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            probs = outputs.cpu().numpy().squeeze()
            preds = (probs > 0.5).astype(int)
        
        # Return predictions and optionally probabilities 
        if return_probabilities:
            return preds, probs
        return preds
    
class BinaryDetector:
    """Handles binary classification with optional PCA reduction"""
    
    def __init__(self, n_components=None, contamination=0.1, random_state=42, input_dim=None):
        """
        Args:
            n_components: Optional PCA components (None for no PCA)
            contamination: Expected fraction of outliers
            random_state: Random state for reproducibility
            input_dim: Input dimension for neural classifier
        """
        self.scaler = StandardScaler()
        self.pca = None  # Only initialize PCA if needed
        self.n_components = n_components
        self.random_state = random_state
        self.input_dim = input_dim  # Store raw input dimension
        
        # Components
        self.classifier = None
        self.real_centroid = None
        self._is_fitted = False
    
    def _init_classifier(self, classifier_type="svm"):
        """Initialize classifier (SVM or logistic regression)"""
        if classifier_type == "svm":
            return SVC(
                kernel='rbf',
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif classifier_type == "lr":
            
            return LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            )
        elif classifier_type== "xgb":
            return xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                max_depth=7,
                learning_rate=0.01,
                n_estimators=500,
                use_label_encoder=False,
                random_state=self.random_state
            )
        elif classifier_type == "neural":
            # Use PCA-reduced dim if PCA is fitted, else raw input dim
            dim = self.pca.n_components_ if self.pca is not None else self.input_dim
            return DeepBinaryDetector(
                input_dim=dim,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
        
    def _preprocess_features(self, features, fit=False):
        """Preprocess features with scaling and optional PCA"""
        # Always scale
        if fit:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
            
        # Apply PCA if specified
        if self.n_components is not None:
            if self.pca is None:
                self.pca = PCA(n_components=self.n_components)
                features = self.pca.fit_transform(features)
                print(f"PCA reduced shape: {features.shape}")
                print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
            else:
                features = self.pca.transform(features)
                
        return features
    
    def fit(self, embeddings, labels, validation_split=0.2, classifier_type="svm", pca=True):
        """
        Train the classifier on embeddings
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            labels: numpy array of shape (n_samples,) with binary labels (0=real, 1=fake)
            validation_split: Fraction of data for validation (0 for no validation)
            classifier_type: Type of classifier ("svm", "lr", "xgb", "neural")
            pca: Whether to apply PCA dimensionality reduction
            
        Returns:
            training_results: Dictionary with training metrics
        """

        # Store raw input dimension
        self.input_dim = embeddings.shape[1]
        
        # Preprocess features
        features = self._preprocess_features(embeddings, fit=True)

        # Calculate real centroid for distance calculations
        real_mask = (labels == 0)
        if np.sum(real_mask) > 0:
            self.real_centroid = np.mean(embeddings[real_mask], axis=0)
        else:
            print("Warning: No real samples found (label=0)")
            self.real_centroid = np.mean(embeddings, axis=0)
        
        # Train/validation split if requested
        if validation_split > 0 and len(embeddings) > 50:
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels,
                test_size=validation_split,
                random_state=self.random_state,
                stratify=labels
            )
        else:
            X_train, y_train = features, labels
            X_val, y_val = None, None
        
        # Train classifier
        print(f"Fitting {classifier_type.upper()} classifier...")
        self.classifier = self._init_classifier(classifier_type)
        
        # Handle neural network training separately
        if classifier_type == "neural":
            training_results = self.classifier.fit(
                X_train, y_train, 
                validation_split=validation_split if X_val is None else 0.0,
                epochs=10, 
                batch_size=32, 
                lr=0.001
            )
            val_acc = training_results.get('val_accuracy')
            
            # Get training predictions (neural network returns tuple)
            train_pred = self.classifier.predict(X_train, return_probabilities=False)
        else:
            # Standard sklearn-style classifiers
            self.classifier.fit(X_train, y_train)
            train_pred = self.classifier.predict(X_train)
            val_acc = None
        
        # Training metrics
        train_acc = accuracy_score(y_train, train_pred)
        
        if pca:
            results = {
                'train_accuracy': train_acc,
                'n_samples': len(embeddings),
                'n_features': embeddings.shape[1],
                'n_components': self.pca.n_components_
            }
        else:
            results = {
                'train_accuracy': train_acc,
                'n_samples': len(embeddings),
                'n_features': embeddings.shape[1],
                'n_components': embeddings.shape[1]
            }
        
        print(f"Training accuracy: {train_acc:.3f}")
        print("Training Classification Report:")
        print(classification_report(y_train, train_pred, target_names=['Real', 'Fake']))
        
        # Validation metrics if available
        if X_val is not None and classifier_type != "neural":
            val_pred = self.classifier.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            results['val_accuracy'] = val_acc
            
            print(f"Validation accuracy: {val_acc:.3f}")
            print("Validation Classification Report:")
            print(classification_report(y_val, val_pred, target_names=['Real', 'Fake']))
        elif val_acc is not None:
            results['val_accuracy'] = val_acc
            print(f"Validation accuracy: {val_acc:.3f}")
        
        self._is_fitted = True
        print(f"{classifier_type.upper()} training completed!")
        
        return results
    
    def predict(self, embeddings, return_probabilities=True, return_distances=True, pca=True):
        """
        Predict labels for embeddings
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            return_probabilities: Whether to return prediction probabilities
            return_distances: Whether to return distances to real centroid
            pca: Whether to apply PCA transformation
            
        Returns:
            predictions: numpy array of predictions (0=real, 1=fake)
            probabilities: numpy array of probabilities (if requested)
            distances: numpy array of distances to real centroid (if requested)
        """
        if not self._is_fitted:
            raise ValueError("Detector not fitted! Call fit() first.")
        
        features = self._preprocess_features(embeddings, fit=False)
        
        # Check if classifier is neural network
        is_neural = isinstance(self.classifier, DeepBinaryDetector)
        
        # Get predictions
        if is_neural:
            # Neural network was trained on preprocessed features; reuse them here
            if return_probabilities:
                predictions, probabilities = self.classifier.predict(features, return_probabilities=True)
            else:
                predictions = self.classifier.predict(features, return_probabilities=False)
        else:
            # Standard sklearn classifiers
            predictions = self.classifier.predict(features)
            if return_probabilities:
                probabilities = self.classifier.predict_proba(features)[:, 1]  # P(fake)
        
        # Build results list
        results = [predictions]
        
        if return_probabilities:
            results.append(probabilities)
        
        if return_distances:
            distances = np.linalg.norm(embeddings - self.real_centroid, axis=1)
            results.append(distances)
        
        return results if len(results) > 1 else results[0]

    def predict_pairs(self, embeddings_1, embeddings_2, pca=True):
        """
        Predict which text in each pair is real
        
        Args:
            embeddings_1: embeddings for first texts in pairs
            embeddings_2: embeddings for second texts in pairs
            pca: Whether to apply PCA transformation
            
        Returns:
            pair_predictions: 1 if first text is real, 2 if second text is real
            details: detailed predictions for each text
        """
        # Get predictions for both sets
        pred_1, prob_1, dist_1 = self.predict(embeddings_1, pca=pca)
        pred_2, prob_2, dist_2 = self.predict(embeddings_2, pca=pca)
        
        # For each pair, choose text with higher probability of being real
        prob_real_1 = 1 - prob_1  # Convert P(fake) to P(real)
        prob_real_2 = 1 - prob_2
        
        pair_predictions = np.where(prob_real_1 > prob_real_2, 1, 2)
        
        details = []
        for i in range(len(embeddings_1)):
            details.append({
                'text1_pred': int(pred_1[i]),
                'text2_pred': int(pred_2[i]),
                'text1_prob_fake': float(prob_1[i]),
                'text2_prob_fake': float(prob_2[i]),
                'text1_prob_real': float(prob_real_1[i]),
                'text2_prob_real': float(prob_real_2[i]),
                'text1_distance': float(dist_1[i]),
                'text2_distance': float(dist_2[i]),
                'pair_prediction': int(pair_predictions[i])
            })
        
        return pair_predictions, details


    

class OutlierDetections:
    def __init__(self, detector_type="elliptic", contamination=0.1, random_state=42,n_components=0.95, use_trajectory=False):
        """
        Generalized outlier detection class

        Args:
            detector_type: one of ["elliptic", "ocsvm", "iforest"]
            contamination: expected fraction of outliers
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.detector_type = detector_type
        self.contamination = contamination
        self.random_state = random_state

        self.use_trajectory = use_trajectory
        self.traj_detector = None  

        self.outlier_detector = None

        # Data storage
        self.real_embeddings = None
        self.real_embeddings_scaled = None
        self.real_embeddings_pca = None
        self.real_centroid = None

        # For SVM training
        self.fake_embeddings = None
        self.fake_embeddings_scaled = None
        self.fake_embeddings_pca = None


    def _init_detector(self):
        """Factory to initialize outlier detector based on type"""
        if self.detector_type == "elliptic":
            return EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state,
                support_fraction=0.9
            )
        elif self.detector_type == "ocsvm":
            return OneClassSVM(
                kernel="rbf",
                nu=self.contamination,  # nu ~ proportion of anomalies
                gamma="scale"
            )
        elif self.detector_type == "iforest":
            return IsolationForest(
                contamination=self.contamination,
                n_estimators=200,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown detector_type: {self.detector_type}. Supported: 'elliptic', 'ocsvm', 'iforest'")

    def fit(self, embeddings, labels, validation_split=0.2):
        """
        Train the outlier detector on embeddings (only uses real samples, ignores fake labels)
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            labels: numpy array of shape (n_samples,) with binary labels (0=real, 1=fake)
            validation_split: Fraction of data for validation (not used for outlier detection)
            
        Returns:
            training_results: Dictionary with training metrics
        """
        print(f"Training {self.detector_type} outlier detector on {embeddings.shape[0]} samples...")
        

        real_embeddings = embeddings
        
        # Check for NaN/Inf
        n_nan = np.isnan(real_embeddings).sum()
        n_inf = np.isinf(real_embeddings).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"Warning: Found {n_nan} NaN and {n_inf} Inf values in embeddings")
        
        # Store real embeddings
        self.real_embeddings = real_embeddings
        
        # Preprocessing
        self.real_embeddings_scaled = self.scaler.fit_transform(real_embeddings)
        self.real_embeddings_pca = self.pca.fit_transform(self.real_embeddings_scaled)
        
        print(f"PCA reduced shape: {self.real_embeddings_pca.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Calculate real centroid for distance calculations
        self.real_centroid = np.mean(self.real_embeddings_pca, axis=0)
        
        # Train outlier detector on real samples only
        print(f"Fitting {self.detector_type} outlier detector...")
        self.outlier_detector = self._init_detector()
        self.outlier_detector.fit(self.real_embeddings_pca)
        
        # Evaluate on all data (including fake samples for validation)
        all_embeddings_scaled = self.scaler.transform(embeddings)
        all_embeddings_pca = self.pca.transform(all_embeddings_scaled)
        
        # Predictions: 1 = inlier (real), -1 = outlier (fake)
        raw_preds = self.outlier_detector.predict(all_embeddings_pca)
        predictions = [0 if pred == 1 else 1 for pred in raw_preds]  # Convert to 0=real, 1=fake
        
        # Calculate accuracy
        train_acc = accuracy_score(labels, predictions)
        
        results = {
            'train_accuracy': train_acc,
            'n_samples': len(embeddings),
            'n_real_samples': len(real_embeddings),
            'n_features': self.real_embeddings_pca.shape[1],
            'n_components': self.pca.n_components_,
            'contamination': self.contamination
        }
        
        print(f"Training accuracy: {train_acc:.3f}")
        print("Training Classification Report:")
        print(classification_report(labels, predictions, target_names=['Real', 'Fake']))
        
        print(f"{self.detector_type.upper()} outlier detector training completed!")
        
        return results

    def predict(self, embeddings, return_probabilities=True, return_distances=True):
        """
        Predict labels for embeddings using trained outlier detector
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            return_probabilities: Whether to return prediction probabilities
            return_distances: Whether to return distances to real centroid
            
        Returns:
            predictions: numpy array of predictions (0=real, 1=fake)
            probabilities: numpy array of probabilities (if requested)
            distances: numpy array of distances to real centroid (if requested)
        """
        if self.outlier_detector is None:
            raise ValueError("Detector not fitted! Call fit() first.")
        
        # Preprocess embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)
        
        # Predictions: 1 = inlier (real), -1 = outlier (fake)
        raw_preds = self.outlier_detector.predict(embeddings_pca)
        predictions = np.array([0 if pred == 1 else 1 for pred in raw_preds])  # Convert to 0=real, 1=fake
        
        results = [predictions]
        
        if return_probabilities:
            # Get outlier scores and convert to probabilities
            if hasattr(self.outlier_detector, "decision_function"):
                scores = self.outlier_detector.decision_function(embeddings_pca)
            elif hasattr(self.outlier_detector, "score_samples"):
                scores = self.outlier_detector.score_samples(embeddings_pca)
            else:
                # Fallback: use negative distance to centroid
                distances = np.linalg.norm(embeddings_pca - self.real_centroid, axis=1)
                scores = -distances
            
            # Convert scores to probabilities (higher score = more normal = lower P(fake))
            probabilities = 1 / (1 + np.exp(scores))  # Sigmoid transformation
            results.append(probabilities)
        
        if return_distances:
            distances = np.linalg.norm(embeddings_pca - self.real_centroid, axis=1)
            results.append(distances)
        
        return results if len(results) > 1 else results[0]

    def predict_pairs(self, embeddings_1, embeddings_2):
        """
        Predict which embedding in each pair is real using outlier detection
        
        Args:
            embeddings_1: embeddings for first items in pairs
            embeddings_2: embeddings for second items in pairs
            
        Returns:
            pair_predictions: 1 if first embedding is real, 2 if second embedding is real
            details: detailed predictions for each embedding
        """
        # Get predictions for both sets
        pred_1, prob_1, dist_1 = self.predict(embeddings_1)
        pred_2, prob_2, dist_2 = self.predict(embeddings_2)
        
        # For each pair, choose embedding with lower probability of being fake (higher probability of being real)
        prob_real_1 = 1 - prob_1  # Convert P(fake) to P(real)
        prob_real_2 = 1 - prob_2
        
        pair_predictions = np.where(prob_real_1 > prob_real_2, 1, 2)
        
        details = []
        for i in range(len(embeddings_1)):
            details.append({
                'text1_pred': int(pred_1[i]),
                'text2_pred': int(pred_2[i]),
                'text1_prob_fake': float(prob_1[i]),
                'text2_prob_fake': float(prob_2[i]),
                'text1_prob_real': float(prob_real_1[i]),
                'text2_prob_real': float(prob_real_2[i]),
                'text1_distance': float(dist_1[i]),
                'text2_distance': float(dist_2[i]),
                'pair_prediction': int(pair_predictions[i])
            })
        
        return pair_predictions, details

    def learn_real_manifold(self, real_texts, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        Train outlier detector on real texts only - texts that deviate from this manifold are considered fake.
        
        Args:
            real_texts: list of real text strings to learn the manifold from
            extractor_model: embedding extractor model
            target_layer: which transformer layer to use for embeddings
            pooling: pooling method ('mean', 'cls', 'last_token', 'all')
            batch_size: batch size for processing
        """
        
        print(f"Learning real text manifold from {len(real_texts)} samples...")
        print(f"Using layer {target_layer} with {self.detector_type} detector")

        # Extract embeddings in one go (batched)
        if pooling == 'all':
            all_embeds = extractor_model._get_all_token_embeddings(real_texts, batch_size=batch_size)
        else:
            all_embeds = extractor_model.get_all_layer_embeddings(real_texts, pooling=pooling, batch_size=batch_size)
        
        self.real_embeddings = all_embeds[target_layer]  # shape: (num_texts, hidden_dim)

        # Check for NaN/Inf values
        n_nan = np.isnan(self.real_embeddings).sum()
        n_inf = np.isinf(self.real_embeddings).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"Warning: Found {n_nan} NaN and {n_inf} Inf values in embeddings")

        print("Embedding stats:",
            "Shape:", self.real_embeddings.shape,
            "NaNs:", n_nan,
            "Infs:", n_inf,
            "Max abs:", np.max(np.abs(self.real_embeddings)))

        # Standardize + PCA
        self.real_embeddings_scaled = self.scaler.fit_transform(self.real_embeddings)
        self.real_embeddings_pca = self.pca.fit_transform(self.real_embeddings_scaled)

        print(f"PCA reduced shape: {self.real_embeddings_pca.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")

        # Centroid for distance measure
        self.real_centroid = np.mean(self.real_embeddings_pca, axis=0)

        # Initialize and fit outlier detector on real text manifold
        self.outlier_detector = self._init_detector()
        self.outlier_detector.fit(self.real_embeddings_pca)

        print(f"✓ {self.detector_type.upper()} outlier detector trained on real text manifold!")
        
        # Optional trajectory-based features
        if self.use_trajectory:
            print("Extracting trajectory features from real texts...")
            traj_features = []
            for text in real_texts:
                try:
                    if pooling == 'all':
                        token_embeds = extractor_model._get_all_token_embeddings([text], batch_size=1)[target_layer]
                    else:
                        # Get token-level embeddings for trajectory
                        token_embeds = extractor_model.get_all_layer_embeddings(text, pooling="all", batch_size=1)[target_layer]
                    
                    traj_metrics = self.trajectory_metrics(token_embeds)
                    traj_vector = [
                        traj_metrics['mean_angle'],
                        traj_metrics['std_angle'], 
                        traj_metrics['angle_entropy'],
                        traj_metrics['trajectory_smoothness'],
                        traj_metrics['max_curvature']
                    ]
                    traj_features.append(traj_vector)
                except Exception as e:
                    print(f"Warning: Failed to extract trajectory for one text: {e}")
                    traj_features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            
            if traj_features:
                traj_features = np.array(traj_features)
                # Simple EllipticEnvelope as trajectory detector
                self.traj_detector = EllipticEnvelope(
                    contamination=self.contamination,
                    random_state=self.random_state
                )
                self.traj_detector.fit(traj_features)
                print("✓ Trajectory detector trained!")


    def predict_texts_batch(self, texts, extractor_model, target_layer=-2, pooling="mean", batch_size=32, return_probabilities=True):
        """
        Predict multiple texts in batch using trained outlier detector.

        Args:
            texts: list of text strings to predict
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            pooling: pooling method
            batch_size: batch size for processing
            return_probabilities: whether to return probability estimates
            
        Returns:
            predictions: list[int] (0=real/inlier, 1=fake/outlier)
            probabilities: list[float] probability of being fake (if requested)
            distances: list[float] distance to real centroid
            scores: list[float] outlier scores
        """
        if self.outlier_detector is None:
            raise ValueError("Detector not trained! Call learn_real_manifold() first.")
            
        # Extract embeddings
        if pooling == 'all':
            all_layer_embeds = extractor_model._get_all_token_embeddings(texts, batch_size=batch_size)
        else:
            all_layer_embeds = extractor_model.get_all_layer_embeddings(
                texts, pooling=pooling, batch_size=batch_size
            )
        embeddings = all_layer_embeds[target_layer]  # shape: (num_texts, hidden_dim)

        # Transform embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)

        # Distance to centroid
        distances = np.linalg.norm(embeddings_pca - self.real_centroid, axis=1)

        # Predictions: 1 = inlier (real), -1 = outlier (fake)
        raw_preds = self.outlier_detector.predict(embeddings_pca)
        predictions = [0 if pred == 1 else 1 for pred in raw_preds]  # Convert to 0=real, 1=fake

        # Outlier scores (higher = more normal/real)
        if hasattr(self.outlier_detector, "decision_function"):
            scores = self.outlier_detector.decision_function(embeddings_pca)
        elif hasattr(self.outlier_detector, "score_samples"):
            scores = self.outlier_detector.score_samples(embeddings_pca)
        else:
            scores = -distances  # fallback: negative distance (closer to centroid = higher score)

        results = [predictions]
        
        if return_probabilities:
            # Convert scores to probabilities using sigmoid-like transformation
            # Higher scores (more normal) -> lower probability of being fake
            prob_fake = 1 / (1 + np.exp(scores))  # Sigmoid transformation
            results.append(prob_fake.tolist())
        
        results.extend([distances.tolist(), scores.tolist()])
        
        return results if len(results) > 1 else results[0]


    def predict_text(self, text, extractor_model, target_layer=-2, pooling="mean", batch_size=32, return_probabilities=True):
        """
        Predict single text using trained outlier detector.
        
        Args:
            text: text string to predict
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            pooling: pooling method
            batch_size: batch size for processing
            return_probabilities: whether to return probability estimates
            
        Returns:
            prediction: int (0=real, 1=fake)
            probability: float probability of being fake (if requested)
            distance: float distance to real centroid
            score: float outlier score
        """
        # Use batch prediction for consistency
        results = self.predict_texts_batch(
            [text], extractor_model, target_layer, pooling, batch_size, return_probabilities
        )
        
        if return_probabilities:
            predictions, probabilities, distances, scores = results
            return predictions[0], probabilities[0], distances[0], scores[0]
        else:
            predictions, distances, scores = results
            return predictions[0], distances[0], scores[0]

    def predict_pairs(self, texts_1, texts_2, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        Predict which text in each pair is real using outlier detection.
        
        Args:
            texts_1: list of first texts in pairs
            texts_2: list of second texts in pairs
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            pooling: pooling method
            batch_size: batch size for processing
            
        Returns:
            pair_predictions: list[int] (1 if first text is real, 2 if second text is real)
            details: list[dict] detailed predictions for each text
        """
        # Get predictions for both sets
        pred_1, prob_1, dist_1, score_1 = self.predict_texts_batch(
            texts_1, extractor_model, target_layer, pooling, batch_size
        )
        pred_2, prob_2, dist_2, score_2 = self.predict_texts_batch(
            texts_2, extractor_model, target_layer, pooling, batch_size
        )
        
        # For each pair, choose text with lower probability of being fake (higher probability of being real)
        pair_predictions = []
        details = []
        
        for i in range(len(texts_1)):
            prob_real_1 = 1 - prob_1[i]  # Convert P(fake) to P(real)
            prob_real_2 = 1 - prob_2[i]
            
            prediction = 1 if prob_real_1 > prob_real_2 else 2
            pair_predictions.append(prediction)
            
            details.append({
                'text1_pred': pred_1[i],
                'text2_pred': pred_2[i],
                'text1_prob_fake': prob_1[i],
                'text2_prob_fake': prob_2[i],
                'text1_prob_real': prob_real_1,
                'text2_prob_real': prob_real_2,
                'text1_distance': dist_1[i],
                'text2_distance': dist_2[i],
                'text1_score': score_1[i],
                'text2_score': score_2[i],
                'pair_prediction': prediction
            })
        
        return pair_predictions, details

    
    def trajectory_metrics(self, token_embeddings):
        """
        Compute turning angles between successive embedding steps.

        Args:
            token_embeddings: np.ndarray of shape (num_tokens, hidden_dim)

        Returns:
            dict: trajectory features including angles, entropy, smoothness, etc.
        """
        if len(token_embeddings) < 3:
            return {
                'mean_angle': 0.0,
                'std_angle': 0.0,
                'angle_entropy': 0.0,
                'trajectory_smoothness': 0.0,
                'max_curvature': 0.0
            }
            
        angles = []
        for i in range(len(token_embeddings) - 2):
            p1 = token_embeddings[i]
            p2 = token_embeddings[i + 1]
            p3 = token_embeddings[i + 2]

            v1 = p2 - p1
            v2 = p3 - p2

            cos_traj = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
            angle = np.arccos(np.clip(cos_traj, -1.0, 1.0))  # radians in [0, π]

            angles.append(angle)
            
        if len(angles) == 0:
            return {
                'mean_angle': 0.0,
                'std_angle': 0.0,
                'angle_entropy': 0.0,
                'trajectory_smoothness': 0.0,
                'max_curvature': 0.0
            }
            
        hist, _ = np.histogram(angles, bins=10, density=True)
        hist = hist + 1e-10

        return {
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'angle_entropy': entropy(hist),
            'trajectory_smoothness': np.mean(np.diff(angles)) if len(angles) > 1 else 0.0,
            'max_curvature': np.max(angles)
        }

    def visualize_manifold(self, df_train, df_train_gt, extractor_model, target_layer=-2, n_samples=20, n_components=0.95):
        """
        Visualize PCA manifold of real vs fake texts.

        Args:
            df_train: dataframe with 'id', 'file_1', 'file_2'
            df_train_gt: dataframe with 'id', 'real_text_id'
            extractor_model: embedding extractor
            target_layer: which transformer layer to extract
            n_samples: number of fake texts to plot
            n_components: number of PCA components
        """
        # Merge training data with ground truth labels
        df_merged = df_train.merge(df_train_gt, on='id')

        # Select real and fake texts
        real_texts = [
            row['file_1'] if row['real_text_id'] == 1 else row['file_2']
            for idx, row in df_merged.iterrows()
        ]
        fake_texts = [
            row['file_2'] if row['real_text_id'] == 1 else row['file_1']
            for idx, row in df_merged.iterrows()
        ]

        # --- Compute embeddings for real texts ---
        #layer_embeds_real = extractor_model.get_all_layer_embeddings(real_texts, pooling='mean')
        layer_embeds_real = extractor_model._get_all_token_embeddings(real_texts, batch_size=32)

        print(layer_embeds_real)
        size = len(real_texts)
        size =4000
        real_embeddings = np.array([layer_embeds_real[target_layer][i] for i in range(size)])

        # Standardize + PCA
        print(real_embeddings.shape)
        scaler = StandardScaler()
        real_embeddings_scaled = scaler.fit_transform(real_embeddings)

        pca = PCA(n_components=n_components)
        real_embeddings_pca = pca.fit_transform(real_embeddings_scaled)
        real_centroid = np.mean(real_embeddings_pca, axis=0)

        # --- Compute embeddings for fake texts (subset for plotting) ---
        #layer_embeds_fake = extractor_model.get_all_layer_embeddings(fake_texts[:n_samples], pooling='mean')
        layer_embeds_fake = extractor_model._get_all_token_embeddings(fake_texts, batch_size=32)


        #size = len(fake_texts)
  
        fake_embeddings = np.array([layer_embeds_fake[target_layer][i] for i in range(size)])
        #fake_embeddings = np.array([layer_embeds_fake[target_layer][i] for i in range(len(fake_texts[:n_samples]))])
        fake_embeddings_scaled = scaler.transform(fake_embeddings)  # use same scaler
        fake_embeddings_pca = pca.transform(fake_embeddings_scaled)  # use same PCA

        # --- Plot ---
        plt.figure(figsize=(12, 8))
        plt.scatter(real_embeddings_pca[:, 0], real_embeddings_pca[:, 1],
                    alpha=0.6, c='green', label='Real Texts', s=50)
        plt.scatter(real_centroid[0], real_centroid[1],
                    c='red', s=200, marker='*', label='Real Centroid')
        plt.scatter(fake_embeddings_pca[:, 0], fake_embeddings_pca[:, 1],
                    alpha=0.6, c='orange', label='Fake Texts', s=50)

        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title(f'Real vs Fake Text Manifold (Layer {target_layer})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
       


    def plot_angle_distributions(self, df_train, df_train_gt, extractor_model, target_layer=-2):
        """
        Plot trajectory angle distributions for real vs fake texts 
        in a dataframe of text pairs.

        Args:
            df_train: pd.DataFrame with ['file_1', 'file_2']
            df_train_gt: pd.DataFrame with ['real_text_id'] (1 or 2)
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
        """
        all_data = []

        df_merged = df_train.merge(df_train_gt, on='id')


        real_text = [
                    row['file_1'] if row['real_text_id'] == 1 else row['file_2']
                    for idx, row in df_merged.iterrows()
                ]
        fake_text = [
                    row['file_2'] if row['real_text_id'] == 1 else row['file_1']
                    for idx, row in df_merged.iterrows()
                ]

        # --- Real text ---
        layer_embeds_real = extractor_model.get_all_layer_embeddings(
            real_text, pooling="all"
        )

        token_embeds_real = layer_embeds_real[target_layer]
        real_angles = self.trajectory_metrics(token_embeds_real)

        for angle in real_angles:
            all_data.append({"label": "Real", "angle": angle})

        # --- Fake text ---
        layer_embeds_fake = extractor_model.get_all_layer_embeddings(
            fake_text, pooling="all"
        )
        print(layer_embeds_fake)
        token_embeds_fake = layer_embeds_fake[target_layer]
        fake_angles = self.trajectory_metrics(token_embeds_fake)

        for angle in fake_angles:
            all_data.append({"label": "Fake", "angle": angle})

        df_plot = pd.DataFrame(all_data)
        if df_plot.empty:
            print("No angle data collected. Check your inputs.")
            return

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_plot, x="label", y="angle", cut=0, inner="quartile")
        plt.title(f"Distribution of trajectory angles at layer {target_layer}")
        plt.ylabel("Angle (radians)")
        plt.xlabel("Text Type")
        plt.grid(True, alpha=0.3)
        plt.show()


class TrajectoryClassifier:
    """
    A standalone classifier that uses trajectory metrics from token embeddings
    """
    def __init__(self, classifier_type="svm", random_state=42):
        """
        Args:
            classifier_type: "svm", "rf" (random forest), or "lr" (logistic regression)
            random_state: random seed
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = None
        
    def _init_classifier(self):
        """Initialize the underlying classifier"""
        if self.classifier_type == "svm":
            from sklearn.svm import SVC
            return SVC(kernel="rbf", C=1.0, probability=True, random_state=self.random_state)
        elif self.classifier_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.classifier_type == "lr":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")
    
    def extract_trajectory_features(self, texts, extractor_model, target_layer=-2, batch_size=32):
        """
        Extract trajectory features from a list of texts
        
        Args:
            texts: list of text strings
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            batch_size: batch size for processing
            
        Returns:
            features: np.array of shape (n_texts, n_features)
        """
        print(f"Extracting trajectory features from {len(texts)} texts...")
        
        features = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                try:
                    # Get token-level embeddings - use mean pooling as fallback if 'all' fails
                    try:
                        layer_embeds = extractor_model.get_all_layer_embeddings(
                            text, pooling="all", batch_size=1
                        )
                        token_embeds = layer_embeds[target_layer]
                    except Exception as e:
                        # Fallback: use mean pooling and create pseudo-trajectory
                        print(f"Warning: Failed to get token embeddings, using mean pooling fallback: {e}")
                        layer_embeds = extractor_model.get_all_layer_embeddings(
                            text, pooling="mean", batch_size=1
                        )
                        # Create a pseudo-trajectory by replicating the mean embedding
                        mean_embed = layer_embeds[target_layer]
                        token_embeds = np.tile(mean_embed, (3, 1))  # Minimum 3 embeddings for trajectory
                    
                    # Compute trajectory metrics
                    traj_metrics = self.trajectory_metrics(token_embeds)
                    
                    # Convert to feature vector
                    feature_vector = [
                        traj_metrics['mean_angle'],
                        traj_metrics['std_angle'],
                        traj_metrics['angle_entropy'],
                        traj_metrics['trajectory_smoothness'],
                        traj_metrics['max_curvature']
                    ]
                    features.append(feature_vector)
                    
                except Exception as e:
                    print(f"Error processing text: {e}")
                    # Add zero features for failed texts
                    features.append([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def trajectory_metrics(self, token_embeddings):
        """
        Compute turning angles between successive embedding steps.
        Same as OutlierDetections.trajectory_metrics but standalone
        """
        if len(token_embeddings) < 3:
            return {
                'mean_angle': 0.0,
                'std_angle': 0.0,
                'angle_entropy': 0.0,
                'trajectory_smoothness': 0.0,
                'max_curvature': 0.0
            }
            
        angles = []
        for i in range(len(token_embeddings) - 2):
            p1 = token_embeddings[i]
            p2 = token_embeddings[i + 1]
            p3 = token_embeddings[i + 2]

            v1 = p2 - p1
            v2 = p3 - p2

            cos_traj = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
            angle = np.arccos(np.clip(cos_traj, -1.0, 1.0))  # radians in [0, π]

            angles.append(angle)
            
        if len(angles) == 0:
            return {
                'mean_angle': 0.0,
                'std_angle': 0.0,
                'angle_entropy': 0.0,
                'trajectory_smoothness': 0.0,
                'max_curvature': 0.0
            }
            
        hist, _ = np.histogram(angles, bins=10, density=True)
        hist = hist + 1e-10

        return {
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'angle_entropy': entropy(hist),
            'trajectory_smoothness': np.mean(np.diff(angles)) if len(angles) > 1 else 0.0,
            'max_curvature': np.max(angles)
        }
    
    def train(self, real_texts, fake_texts, extractor_model, target_layer=-2, batch_size=32):
        """
        Train the trajectory classifier
        
        Args:
            real_texts: list of real text strings
            fake_texts: list of fake text strings
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            batch_size: batch size for processing
        """
        print(f"Training trajectory classifier with {len(real_texts)} real and {len(fake_texts)} fake texts...")
        
        # Extract features
        real_features = self.extract_trajectory_features(real_texts, extractor_model, target_layer, batch_size)
        fake_features = self.extract_trajectory_features(fake_texts, extractor_model, target_layer, batch_size)
        
        # Combine features and labels
        X = np.vstack([real_features, fake_features])
        y = np.hstack([np.ones(len(real_features)), np.zeros(len(fake_features))])  # 1=real, 0=fake
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature statistics: mean={X.mean():.3f}, std={X.std():.3f}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.classifier = self._init_classifier()
        self.classifier.fit(X_scaled, y)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.classifier, X_scaled, y, cv=5)
        print(f"Trajectory classifier CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Training predictions for analysis
        train_predictions = self.classifier.predict(X_scaled)
        print("\nTraining Classification Report:")
        print(classification_report(y, train_predictions, target_names=['Fake', 'Real']))
        
        return cv_scores.mean()
    
    def predict_batch(self, texts, extractor_model, target_layer=-2, batch_size=32):
        """
        Predict multiple texts
        
        Returns:
            predictions: list[int] (1=real, 0=fake)
            probabilities: list[float] probability of being real
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained! Call train() first.")
        
        # Extract features
        features = self.extract_trajectory_features(texts, extractor_model, target_layer, batch_size)
        
        # Standardize and predict
        features_scaled = self.scaler.transform(features)
        predictions = self.classifier.predict(features_scaled)
        
        if hasattr(self.classifier, 'predict_proba'):
            probabilities = self.classifier.predict_proba(features_scaled)[:, 1]  # Prob of class 1 (real)
        else:
            # For classifiers without probability estimates
            probabilities = predictions.astype(float)
        
        return predictions.tolist(), probabilities.tolist()
    
    def predict_single(self, text, extractor_model, target_layer=-2):
        """Predict single text"""
        predictions, probabilities = self.predict_batch([text], extractor_model, target_layer, batch_size=1)
        return predictions[0], probabilities[0]
    
    def evaluate_pairs(self, df, labels_df, extractor_model, target_layer=-2, batch_size=32):
        """
        Evaluate text pairs using trajectory classifier
        """
        print("Evaluating text pairs with trajectory classifier...")
        
        # Prepare texts
        texts1 = df['file_1'].tolist()
        texts2 = df['file_2'].tolist()
        
        # Predict in batch
        pred1_list, prob1_list = self.predict_batch(texts1, extractor_model, target_layer, batch_size)
        pred2_list, prob2_list = self.predict_batch(texts2, extractor_model, target_layer, batch_size)
        
        predictions = []
        details = []
        
        for idx, (prob1, prob2, pred1, pred2) in enumerate(
            zip(prob1_list, prob2_list, pred1_list, pred2_list)
        ):
            # Choose text with higher probability of being real
            prediction = 1 if prob1 > prob2 else 2
            predictions.append(prediction)
            
            true_label = labels_df.loc[df.index[idx]]['real_text_id'] if df.index[idx] in labels_df.index else None
            
            details.append({
                'text1_prediction': pred1,
                'text2_prediction': pred2,
                'text1_probability': prob1,
                'text2_probability': prob2,
                'prediction': prediction,
                'true_label': true_label
            })
        
        return np.array(predictions), details