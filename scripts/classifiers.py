import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from scipy.stats import entropy

import matplotlib.pyplot as plt

import seaborn as sns
from tqdm import tqdm

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
        """Factory to initialize detector based on type"""
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
        elif self.detector_type == "svm_binary":
            return SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,  # Enable probability estimates
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown detector_type: {self.detector_type}")

    def learn_real_manifold(self, real_texts, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        

        """
        
        print(f"Learning manifold from {len(real_texts)} real texts...")
        print(f"Using layer {target_layer} with detector {self.detector_type}")

        # Extract embeddings in one go (batched)
        if pooling == 'all':
            all_embeds = extractor_model._get_all_token_embeddings(real_texts, batch_size=batch_size)
        else:
            all_embeds = extractor_model.get_all_layer_embeddings(real_texts, pooling=pooling, batch_size=batch_size)
        
        self.real_embeddings = all_embeds[target_layer]  # shape: (num_texts, hidden_dim)

        print("Embedding stats:",
            np.isnan(self.real_embeddings).sum(), "NaNs;",
            np.isinf(self.real_embeddings).sum(), "Infs;",
            "max abs:", np.max(np.abs(self.real_embeddings)))

        # Standardize + PCA
        self.real_embeddings_scaled = self.scaler.fit_transform(self.real_embeddings)
        self.real_embeddings_pca = self.pca.fit_transform(self.real_embeddings_scaled)

        print(f"PCA reduced shape: {self.real_embeddings_pca.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")

        # Centroid for distance measure
        self.real_centroid = np.mean(self.real_embeddings_pca, axis=0)

        # Initialize and fit outlier detector
        self.outlier_detector = self._init_detector()
        self.outlier_detector.fit(self.real_embeddings_pca)

        print("Real text manifold learned!")
        
        if self.use_trajectory:
            print(" Fit trajecotries features")
            traj_features = []
            for text in real_texts:
                token_embeds = extractor_model.get_all_layer_embeddings(text, pooling="all")[target_layer]
                traj_features.append(self.trajectory_metrics(token_embeds))
            traj_features = np.array(traj_features)

            # Simple EllipticEnvelope as trajectory detector
            self.traj_detector = EllipticEnvelope(
                contamination=self.contamination,
                random_state=self.random_state
            )
            self.traj_detector.fit(traj_features)

    def learn_binary_classification(self, real_texts, fake_texts, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        Learn binary classification from both real and fake texts (for SVM)
        Args:
            real_texts: list of real text strings
            fake_texts: list of fake text strings
            extractor_model: embedding extractor
            target_layer: which transformer layer to use
            pooling: pooling strategy
            batch_size: batch size for processing
        """
        print(f"Learning binary classification from {len(real_texts)} real and {len(fake_texts)} fake texts...")
        print(f"Using layer {target_layer} with SVM binary classifier")
        
        # Extract real embeddings
        if pooling == 'all':
            real_embeds = extractor_model._get_all_token_embeddings(real_texts, batch_size=batch_size)
            fake_embeds = extractor_model._get_all_token_embeddings(fake_texts, batch_size=batch_size)
        else:
            real_embeds = extractor_model.get_all_layer_embeddings(real_texts, pooling=pooling, batch_size=batch_size)
            fake_embeds = extractor_model.get_all_layer_embeddings(fake_texts, pooling=pooling, batch_size=batch_size)
        
        self.real_embeddings = real_embeds[target_layer]
        self.fake_embeddings = fake_embeds[target_layer]
        
        print(f"Real embeddings shape: {self.real_embeddings.shape}")
        print(f"Fake embeddings shape: {self.fake_embeddings.shape}")
        
        # Combine embeddings and create labels
        all_embeddings = np.vstack([self.real_embeddings, self.fake_embeddings])
        labels = np.hstack([np.ones(len(self.real_embeddings)),    # 1 for real
                           np.zeros(len(self.fake_embeddings))])   # 0 for fake
        
        print("Combined embedding stats:",
              np.isnan(all_embeddings).sum(), "NaNs;",
              np.isinf(all_embeddings).sum(), "Infs;",
              "max abs:", np.max(np.abs(all_embeddings)))
        
        # Standardize + PCA on combined data
        all_embeddings_scaled = self.scaler.fit_transform(all_embeddings)
        all_embeddings_pca = self.pca.fit_transform(all_embeddings_scaled)
        
        print(f"PCA reduced shape: {all_embeddings_pca.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Split back for storage
        n_real = len(self.real_embeddings)
        self.real_embeddings_pca = all_embeddings_pca[:n_real]
        self.fake_embeddings_pca = all_embeddings_pca[n_real:]
        self.real_centroid = np.mean(self.real_embeddings_pca, axis=0)
        
        # Train SVM classifier
        self.svm_classifier = self._init_detector()
        self.svm_classifier.fit(all_embeddings_pca, labels)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.svm_classifier, all_embeddings_pca, labels, cv=5)
        print(f"SVM Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Predictions on training data for analysis
        train_predictions = self.svm_classifier.predict(all_embeddings_pca)
        print("\nTraining Classification Report:")
        print(classification_report(labels, train_predictions, target_names=['Fake', 'Real']))
        
        print("Binary SVM classifier trained successfully!")

    def predict_texts_batch_svm(self, texts, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        Predict multiple texts using SVM binary classifier
        Returns:
            predictions: list[int] (1=real, 0=fake)
            probabilities: list[float] probability of being real
            distances: list[float] distance to real centroid
        """
        if self.svm_classifier is None:
            raise ValueError("SVM classifier not trained! Call learn_binary_classification first.")
        
        # Get embeddings
        all_layer_embeds = extractor_model.get_all_layer_embeddings(
            texts, pooling=pooling, batch_size=batch_size
        )
        embeddings = all_layer_embeds[target_layer]
        
        # Transform embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)
        
        # Distance to real centroid
        distances = np.linalg.norm(embeddings_pca - self.real_centroid, axis=1)
        
        # SVM predictions and probabilities
        predictions = self.svm_classifier.predict(embeddings_pca)
        probabilities = self.svm_classifier.predict_proba(embeddings_pca)[:, 1]  # Probability of class 1 (real)
        
        return predictions.tolist(), probabilities.tolist(), distances.tolist()
    
    def predict_text_svm(self, text, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """Predict single text using SVM"""
        if self.svm_classifier is None:
            raise ValueError("SVM classifier not trained! Call learn_binary_classification first.")
        
        # Get embedding
        layer_embeddings = extractor_model.get_all_layer_embeddings(text, pooling=pooling, batch_size=batch_size)
        embedding = layer_embeddings[target_layer].reshape(1, -1)
        
        # Transform
        embedding_scaled = self.scaler.transform(embedding)
        embedding_pca = self.pca.transform(embedding_scaled)
        
        # Distance to centroid
        distance = np.linalg.norm(embedding_pca - self.real_centroid)
        
        # SVM prediction and probability
        prediction = self.svm_classifier.predict(embedding_pca)[0]
        probability = self.svm_classifier.predict_proba(embedding_pca)[0, 1]  # Prob of being real
        
        return int(prediction), float(probability), float(distance)

    def predict_texts_batch(self, texts, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        Predict multiple texts in batch.

        Returns:
            is_real_list: list[int] (1=inlier/real, 0=outlier/fake)
            distances: list[float] distance to centroid
            scores: list[float] outlier score
        """
        all_layer_embeds = extractor_model.get_all_layer_embeddings(
            texts, pooling=pooling, batch_size=batch_size
        )
        embeddings = all_layer_embeds[target_layer]  # shape: (num_texts, hidden_dim)

        # Transform embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)

        # Distance to centroid
        distances = np.linalg.norm(embeddings_pca - self.real_centroid, axis=1)

        # Predictions
        raw_preds = self.outlier_detector.predict(embeddings_pca)
        is_real_list = [(pred == 1) for pred in raw_preds]

        # Scores
        if hasattr(self.outlier_detector, "decision_function"):
            scores = self.outlier_detector.decision_function(embeddings_pca)
        elif hasattr(self.outlier_detector, "score_samples"):
            scores = self.outlier_detector.score_samples(embeddings_pca)
        else:
            scores = -distances  # fallback

        return is_real_list, distances.tolist(), scores.tolist()


    def predict_text(self, text, extractor_model, target_layer=-2,pooling="mean", batch_size=32):
        
        if self.detector_type == "svm_binary":
            return self.predict_text_svm(text, extractor_model, target_layer, pooling, batch_size)
        # Embedding
        layer_embeddings = extractor_model.get_all_layer_embeddings(text, pooling=pooling, batch_size=batch_size)
        embedding = layer_embeddings[target_layer].reshape(1, -1)

        # Transform
        embedding_scaled = self.scaler.transform(embedding)
        embedding_pca = self.pca.transform(embedding_scaled)

        # Distance to centroid
        distance = np.linalg.norm(embedding_pca - self.real_centroid)

        # Prediction
        raw_pred = self.outlier_detector.predict(embedding_pca)[0]
        if self.detector_type == "iforest":
            # For IsolationForest, predict returns 1 (inlier), -1 (outlier)
            is_real = (raw_pred == 1)
        elif self.detector_type in ["elliptic", "ocsvm"]:
            # Same convention: 1 = inlier, -1 = outlier
            is_real = (raw_pred == 1)

        # Outlier score convention: higher = more normal
        if hasattr(self.outlier_detector, "decision_function"):
            score = self.outlier_detector.decision_function(embedding_pca)[0]
        elif hasattr(self.outlier_detector, "score_samples"):
            score = self.outlier_detector.score_samples(embedding_pca)[0]
        else:
            score = -distance  # fallback

        if self.use_trajectory:
            # layer_embeddings[target_layer] shape: (num_tokens, hidden_dim)
            layer_embeddings = extractor_model.get_all_layer_embeddings(text, pooling="all", batch_size=batch_size)
            token_embeddings = layer_embeddings[target_layer]
            traj = self.trajectory_metrics(token_embeddings)
            # You could combine trajectory score with outlier score, e.g., mean_cosine
            traj_score = traj['mean_cosine']
            # Example: weighted combination
            combined_score = traj_score
        else:
            combined_score = score

        return int(is_real), distance, combined_score
    

    def evaluate_pairs_svm(self, df, labels_df, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        Evaluate text pairs using SVM binary classifier
        """
        print("Evaluating text pairs with SVM classifier...")
        
        # Prepare texts
        texts1 = df['file_1'].tolist()
        texts2 = df['file_2'].tolist()
        
        # Predict in batch
        pred1_list, prob1_list, dist1_list = self.predict_texts_batch_svm(
            texts1, extractor_model, target_layer, pooling, batch_size
        )
        pred2_list, prob2_list, dist2_list = self.predict_texts_batch_svm(
            texts2, extractor_model, target_layer, pooling, batch_size
        )
        
        predictions = []
        details = []
        
        for idx, (prob1, prob2, pred1, pred2, dist1, dist2) in enumerate(
            zip(prob1_list, prob2_list, pred1_list, pred2_list, dist1_list, dist2_list)
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
                'text1_distance': dist1,
                'text2_distance': dist2,
                'prediction': prediction,
                'true_label': true_label
            })
        
        return np.array(predictions), details

    def evaluate_pairs(self, df, labels_df, extractor_model, target_layer=-2, pooling="mean", batch_size=32):
        """
        Evaluate text pairs using batch predictions.
        """
        print("Evaluating text pairs in batch...")

        if self.detector_type == "svm_binary":
            return self.evaluate_pairs_svm(df, labels_df, extractor_model, target_layer, pooling, batch_size)
        

        # Prepare texts
        texts1 = df['file_1'].tolist()
        texts2 = df['file_2'].tolist()

        # Predict in batch
        pred1_list, dist1_list, score1_list = self.predict_texts_batch(
            texts1, extractor_model, target_layer, pooling, batch_size
        )
        pred2_list, dist2_list, score2_list = self.predict_texts_batch(
            texts2, extractor_model, target_layer, pooling, batch_size
        )

        predictions = []
        details = []

        for idx, (score1, score2, dist1, dist2) in enumerate(zip(score1_list, score2_list, dist1_list, dist2_list)):
            # Decide which text is more "real-like"
            prediction = 1 if score1 > score2 else 2
            predictions.append(prediction)

            true_label = labels_df.loc[df.index[idx]]['real_text_id'] if df.index[idx] in labels_df.index else None
            details.append({
                'text1_score': score1,
                'text2_score': score2,
                'text1_distance': dist1,
                'text2_distance': dist2,
                'prediction': prediction,
                'true_label': true_label
            })

        return np.array(predictions), details
    
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
       


    def plot_angle_distributions(self, df, labels_df, extractor_model, target_layer=-2):
        """
        Plot trajectory angle distributions for real vs fake texts 
        in a dataframe of text pairs.

        Args:
            df: pd.DataFrame with ['file_1', 'file_2']
            labels_df: pd.DataFrame with ['real_text_id'] (1 or 2)
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

