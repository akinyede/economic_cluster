"""Machine Learning Enhancement for Cluster Decision Making - Version 2.0
Enhanced with ensemble predictions combining 13 and 18 feature models
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
import pickle
from datetime import datetime
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import ensemble model
try:
    from ml.ensemble_model import EnsemblePredictor
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False
    logging.warning("Ensemble model not available. Using fallback predictions.")

# ML and analysis imports
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    import shap
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    StandardScaler = None  # type: ignore
    logging.warning("ML libraries not installed. ML enhancement will be disabled.")

# Network analysis imports
try:
    import networkx as nx
    from sklearn.cluster import SpectralClustering
    HAS_NETWORK_LIBS = True
except ImportError:
    HAS_NETWORK_LIBS = False
    logging.warning("Network analysis libraries not installed. Network features will be disabled.")

logger = logging.getLogger(__name__)

class MLClusterEnhancerV2:
    """Enhanced ML cluster enhancer with ensemble predictions and KC features"""
    
    def __init__(self, config=None):
        # Accept either a dict-like config or a Config object with attributes
        if isinstance(config, dict):
            self.config = config
            _use_kc = self.config.get('use_kc_features', True)
        else:
            self.config = {}
            # Gracefully read from attribute-style config objects
            _use_kc = True
            try:
                _use_kc = getattr(config, 'use_kc_features', getattr(config, 'USE_KC_FEATURES', True))
            except Exception:
                pass
        self.ensemble_predictor = None
        self.gdp_model = None
        self.job_model = None
        self.roi_model = None
        self.network_analyzer = None
        self.explainer = None
        self.scaler = StandardScaler() if StandardScaler else None
        self.model_path = os.path.join(os.path.dirname(__file__), '../models')
        
        # Configuration for KC features
        self.use_kc_features = bool(_use_kc)
        self.kc_data_available = False
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize components
        self._initialize_models()
        
        if HAS_NETWORK_LIBS:
            try:
                from analysis.business_network_analyzer import BusinessNetworkAnalyzer
                self.network_analyzer = BusinessNetworkAnalyzer()
                logger.debug("BusinessNetworkAnalyzer initialized")
            except ImportError:
                # Silently skip - this is an optional enhancement
                self.network_analyzer = None
                logger.debug("BusinessNetworkAnalyzer not available (optional feature)")
    
    def _initialize_models(self):
        """Initialize ML models including ensemble"""
        
        # Initialize ensemble predictor if available
        if HAS_ENSEMBLE:
            try:
                # Prefer global singleton to avoid duplicate loads/logs
                try:
                    from ml.ensemble_model import get_global_predictor
                    self.ensemble_predictor = get_global_predictor(self.model_path)
                    if getattr(self.ensemble_predictor, 'is_loaded', False):
                        logger.info("Ensemble predictor loaded successfully")
                except Exception:
                    # Fallback to local instance
                    ep = EnsemblePredictor()
                    if ep.load_models(self.model_path):
                        self.ensemble_predictor = ep
                        logger.info("Ensemble predictor loaded successfully")
                    else:
                        logger.warning("Failed to load ensemble models")
                        self.ensemble_predictor = None
            except Exception as e:
                logger.error(f"Error initializing ensemble predictor: {e}")
                self.ensemble_predictor = None
        
        # Fallback to individual models if ensemble not available
        if not self.ensemble_predictor and HAS_ML_LIBS:
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback XGBoost models"""
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.gdp_model = xgb.XGBRegressor(**xgb_params)
        self.job_model = xgb.XGBRegressor(**xgb_params)
        self.roi_model = xgb.XGBRegressor(**xgb_params)
        
        # Load pre-trained models if available
        self._load_fallback_models()
    
    def enhance_clusters(self, strategic_clusters: List[Dict], 
                        historical_data: Optional[pd.DataFrame] = None,
                        kc_enhanced_data: Optional[pd.DataFrame] = None) -> Tuple[List[Dict], Dict]:
        """
        Enhance strategic clusters with ML predictions and insights
        
        Args:
            strategic_clusters: List of clusters from strategic former
            historical_data: Historical performance data for training
            kc_enhanced_data: Business data with KC features (if available)
            
        Returns:
            Enhanced clusters and explanations
        """
        if not (HAS_ML_LIBS or self.ensemble_predictor):
            logger.warning("No ML capabilities available. Returning original clusters.")
            return strategic_clusters, {}
        
        # Check if KC data is available
        self.kc_data_available = kc_enhanced_data is not None and not kc_enhanced_data.empty
        
        enhanced_clusters = []
        explanations = {}
        
        for cluster in strategic_clusters:
            # Extract features from cluster
            features = self._extract_cluster_features(cluster, kc_enhanced_data)
            
            # Make predictions using ensemble if available
            if self.ensemble_predictor:
                predictions = self._make_ensemble_predictions(features)
            else:
                predictions = self._predict_outcomes(features)
            
            # Enhance with network analysis if available
            if HAS_NETWORK_LIBS and self.network_analyzer:
                network_metrics = self.network_analyzer.analyze_cluster_network(
                    cluster.get('businesses', [])
                )
                cluster['network_metrics'] = network_metrics
            
            # Add ML predictions to cluster
            enhanced_cluster = cluster.copy()
            enhanced_cluster['ml_predictions'] = predictions

            # Normalize ROI for UI/consumers:
            # - Models output 'roi_percentage' as a percentage value (e.g., 23.5)
            # - UI expects a fraction for 'expected_roi' and 'cluster.roi' (e.g., 0.235)
            try:
                roi_pct = float(predictions.get('roi_percentage', 0))
            except Exception:
                roi_pct = 0.0
            expected_roi = float(predictions.get('expected_roi', roi_pct / 100.0))
            predictions['expected_roi'] = expected_roi
            # Provide a top-level alias for simpler UI access
            enhanced_cluster['roi'] = expected_roi

            # Ensure a cluster_score alias exists for legacy consumers
            if 'total_score' in enhanced_cluster and 'cluster_score' not in enhanced_cluster:
                enhanced_cluster['cluster_score'] = enhanced_cluster.get('total_score', 0)
            
            # Add sustainability prediction
            enhanced_cluster['sustainability_score'] = self.predict_sustainability(cluster)
            
            # Generate explanations
            explanation = self._generate_explanation(features, predictions)
            # Attach per-cluster feature importance via SHAP; fall back to model-level if needed
            fi = self._compute_feature_importance_shap(features)
            if not fi and self.ensemble_predictor:
                try:
                    fi = self.ensemble_predictor.get_feature_importance('gdp_impact')
                except Exception:
                    fi = {}
            if fi:
                # Keep raw importances for audit, and provide de-biased version
                explanation['feature_importance_raw'] = dict(fi)
                explanation['feature_importance'] = self._debias_importance(fi, target='gdp_impact')
            explanations[cluster.get('name', 'Unknown')] = explanation
            
            # Calculate confidence scores
            enhanced_cluster['confidence_score'] = self._calculate_confidence(
                features, predictions
            )
            
            # Add KC enhancement metadata if applicable
            if self.kc_data_available:
                enhanced_cluster['kc_enhanced'] = True
                enhanced_cluster['kc_data_quality'] = predictions.get('kc_data_quality', 0)
            
            enhanced_clusters.append(enhanced_cluster)
        
        # Re-rank clusters based on ML insights
        enhanced_clusters = self._rerank_clusters(enhanced_clusters)
        
        return enhanced_clusters, explanations
    
    def _extract_cluster_features(self, cluster: Dict, kc_data: Optional[pd.DataFrame] = None) -> Dict:
        """Extract ML features from cluster data - supports both 13 and 18 features"""
        
        features = {}
        
        # Core business metrics (1-4)
        features['business_count'] = cluster.get('business_count', 0)
        features['total_employees'] = cluster.get('metrics', {}).get('total_employees', 0)
        features['total_revenue'] = cluster.get('metrics', {}).get('total_revenue', 0)
        
        businesses = cluster.get('businesses', [])
        avg_age = cluster.get('metrics', {}).get('avg_business_age', 0)
        if avg_age == 0 and businesses:
            avg_age = self._calculate_avg_business_age(businesses)
        features['avg_business_age'] = avg_age
        
        # Performance scores (5-6)
        features['strategic_score'] = cluster.get('strategic_score', 0)
        features['innovation_score'] = cluster.get('metrics', {}).get('innovation_score', 0)
        
        # Critical Mass and sustainability metrics (7-13)
        features['critical_mass'] = cluster.get('critical_mass', 50.0)
        features['supply_chain_completeness'] = self._calculate_supply_chain_completeness(cluster)
        features['geographic_density'] = self._calculate_geographic_density(cluster)
        features['workforce_diversity'] = self._calculate_workforce_diversity(cluster)
        features['natural_community_score'] = cluster.get('natural_community_score', 0.5)
        
        # Additional features for better predictions - check if these exist first
        features['cluster_synergy'] = cluster.get('synergy_score',
                                          cluster.get('cluster_synergy',
                                          cluster.get('metrics', {}).get('synergy_score', 60.0)))
        features['market_position'] = cluster.get('market_position',
                                           cluster.get('synergy_score',
                                           cluster.get('metrics', {}).get('market_position', 50.0)))
        
        # Add KC features if available (14-18)
        if self.kc_data_available and kc_data is not None:
            kc_features = self._extract_kc_features(cluster, kc_data)
            features.update(kc_features)
        
        # Store cluster metadata (not used in prediction)
        features['cluster_name'] = cluster.get('name', 'Unknown')
        features['cluster_type'] = cluster.get('type', 'mixed')
        features['county'] = self._get_primary_county(businesses)
        
        return features
    
    def _extract_kc_features(self, cluster: Dict, kc_data: pd.DataFrame) -> Dict:
        """Extract KC-specific features for cluster"""
        kc_features = {}
        businesses = cluster.get('businesses', [])
        
        if not businesses:
            # Return defaults if no businesses
            return {
                'kc_crime_safety': 70,
                'kc_development_activity': 60,
                'kc_demographic_strength': 65,
                'kc_infrastructure_score': 70,
                'kc_market_access': 67
            }
        
        # Get business IDs to match with KC data
        business_ids = [b.get('business_id') for b in businesses if 'business_id' in b]
        
        # Filter KC data for cluster businesses
        if business_ids and 'business_id' in kc_data.columns:
            cluster_kc_data = kc_data[kc_data['business_id'].isin(business_ids)]
        else:
            # Try matching by name if no IDs
            business_names = [b.get('name') for b in businesses if 'name' in b]
            if business_names and 'name' in kc_data.columns:
                cluster_kc_data = kc_data[kc_data['name'].isin(business_names)]
            else:
                cluster_kc_data = pd.DataFrame()
        
        # Calculate average KC features for cluster
        kc_feature_cols = [
            'kc_crime_safety',
            'kc_development_activity',
            'kc_demographic_strength',
            'kc_infrastructure_score',
            'kc_market_access'
        ]
        
        for col in kc_feature_cols:
            if not cluster_kc_data.empty and col in cluster_kc_data.columns:
                # Use actual KC data
                kc_features[col] = cluster_kc_data[col].mean()
            else:
                # Use intelligent defaults based on county
                county = self._get_primary_county(businesses)
                kc_features[col] = self._get_kc_default(col, county)
        
        # Check enhancement quality
        if not cluster_kc_data.empty and 'kc_enhancement_source' in cluster_kc_data.columns:
            direct_pct = (cluster_kc_data['kc_enhancement_source'] == 'direct').mean()
            zip_pct = (cluster_kc_data['kc_enhancement_source'] == 'zip').mean()
            kc_features['kc_data_quality'] = direct_pct * 1.0 + zip_pct * 0.7
        else:
            kc_features['kc_data_quality'] = 0.0
        
        return kc_features
    
    def _get_primary_county(self, businesses: List[Dict]) -> str:
        """Get the primary county for a cluster"""
        if not businesses:
            return 'Unknown'
        
        counties = {}
        for b in businesses:
            county = b.get('county', 'Unknown')
            counties[county] = counties.get(county, 0) + 1
        
        return max(counties, key=counties.get) if counties else 'Unknown'
    
    def _get_kc_default(self, feature: str, county: str) -> float:
        """Get intelligent KC feature defaults based on county"""
        county_defaults = {
            'Johnson': {
                'kc_crime_safety': 85,
                'kc_development_activity': 75,
                'kc_demographic_strength': 85,
                'kc_infrastructure_score': 85,
                'kc_market_access': 83
            },
            'Jackson': {
                'kc_crime_safety': 75,
                'kc_development_activity': 65,
                'kc_demographic_strength': 70,
                'kc_infrastructure_score': 75,
                'kc_market_access': 72
            },
            'Wyandotte': {
                'kc_crime_safety': 65,
                'kc_development_activity': 55,
                'kc_demographic_strength': 60,
                'kc_infrastructure_score': 65,
                'kc_market_access': 61
            },
            'Clay': {
                'kc_crime_safety': 80,
                'kc_development_activity': 60,
                'kc_demographic_strength': 75,
                'kc_infrastructure_score': 70,
                'kc_market_access': 71
            }
        }
        
        # Find matching county
        for county_name, defaults in county_defaults.items():
            if county_name.lower() in county.lower():
                return defaults.get(feature, 65)
        
        # Generic defaults
        generic_defaults = {
            'kc_crime_safety': 70,
            'kc_development_activity': 60,
            'kc_demographic_strength': 65,
            'kc_infrastructure_score': 70,
            'kc_market_access': 67
        }
        
        return generic_defaults.get(feature, 65)
    
    def _make_ensemble_predictions(self, features: Dict) -> Dict:
        """Make predictions using ensemble model"""
        
        # Use ensemble predictor
        result = self.ensemble_predictor.predict(features, target='all')
        
        # Format predictions for consistency
        predictions = {
            'gdp_impact': result['gdp_impact']['prediction'],
            'job_creation': result['job_creation']['prediction'],
            'roi_percentage': result['roi_percentage']['prediction'],
            'confidence': result['ensemble_metadata']['confidence'],
            'model_used': result['ensemble_metadata']['model_selection'],
            'kc_data_quality': result['ensemble_metadata']['kc_data_quality']
        }
        
        # Add component predictions for transparency
        predictions['model_components'] = {}
        for target in ['gdp_impact', 'job_creation', 'roi_percentage']:
            if 'components' in result[target]:
                predictions['model_components'][target] = result[target]['components']
        
        return predictions
    
    def _predict_outcomes(self, features: Dict) -> Dict:
        """Fallback prediction method using heuristics"""
        predictions = {}
        
        # Heuristic predictions based on paper
        revenue = features.get('total_revenue', 0)
        employees = features.get('total_employees', 0)
        strategic_score = features.get('strategic_score', 60)
        critical_mass = features.get('critical_mass', 50)
        
        # GDP Impact
        base_gdp = revenue * 1.85
        mass_boost = 1 + (critical_mass / 100) * 0.2
        predictions['gdp_impact'] = base_gdp * mass_boost * 0.8
        
        # Job Creation
        base_jobs = employees * 2.2
        mass_factor = 0.95 + (critical_mass / 100) * 0.25
        predictions['job_creation'] = int(base_jobs * mass_factor * 0.8)
        
        # ROI
        base_roi = 5 + (strategic_score / 100) * 20
        predictions['roi_percentage'] = base_roi
        
        predictions['confidence'] = 0.5
        predictions['model_used'] = 'heuristic'
        
        return predictions
    
    def _calculate_avg_business_age(self, businesses: List[Dict]) -> float:
        """Calculate average business age from business list"""
        if not businesses:
            return 0
        
        current_year = datetime.now().year
        ages = []
        for business in businesses:
            year_established = business.get('year_established', current_year)
            age = current_year - year_established
            ages.append(age)
        
        return np.mean(ages) if ages else 0
    
    def _calculate_supply_chain_completeness(self, cluster: Dict) -> float:
        """Calculate how complete the supply chain ecosystem is within the cluster"""
        businesses = cluster.get('businesses', [])
        if not businesses:
            return 0.0
        
        # Get NAICS codes to analyze supply chain relationships
        naics_codes = set([b.get('naics_code', '')[:3] for b in businesses if b.get('naics_code')])
        
        # Define supply chain relationships
        supply_chains = {
            'logistics': {'484', '493', '488'},
            'manufacturing': {'332', '333', '336'},
            'biotech': {'325', '3254', '541'},
            'tech': {'511', '518', '541'},
        }
        
        # Calculate completeness for each supply chain
        max_completeness = 0
        for chain_name, chain_codes in supply_chains.items():
            if naics_codes.intersection(chain_codes):
                completeness = len(naics_codes.intersection(chain_codes)) / len(chain_codes)
                max_completeness = max(max_completeness, completeness)
        
        return max_completeness
    
    def _calculate_geographic_density(self, cluster: Dict) -> float:
        """Calculate geographic concentration of the cluster"""
        businesses = cluster.get('businesses', [])
        if not businesses:
            return 0.0
        
        # Count businesses per county
        county_counts = {}
        for b in businesses:
            county = b.get('county', 'Unknown')
            county_counts[county] = county_counts.get(county, 0) + 1
        
        # Calculate concentration (Herfindahl index)
        total = len(businesses)
        concentration = sum((count/total)**2 for count in county_counts.values())
        
        return concentration
    
    def _calculate_workforce_diversity(self, cluster: Dict) -> float:
        """Calculate workforce size diversity"""
        businesses = cluster.get('businesses', [])
        if not businesses:
            return 0.0
        
        # Categorize by size
        size_counts = {'small': 0, 'medium': 0, 'large': 0}
        for b in businesses:
            employees = b.get('employees', 0)
            if employees < 50:
                size_counts['small'] += 1
            elif employees < 500:
                size_counts['medium'] += 1
            else:
                size_counts['large'] += 1
        
        # Calculate Shannon diversity index
        total = len(businesses)
        diversity = 0
        for count in size_counts.values():
            if count > 0:
                p = count / total
                diversity -= p * np.log(p)
        
        # Normalize to 0-1 range
        return diversity / 1.099
    
    def predict_sustainability(self, cluster: Dict) -> float:
        """Predict long-term sustainability of cluster"""
        # Sustainability primarily based on Critical Mass
        critical_mass = cluster.get('critical_mass', 50)
        
        # Additional factors
        supply_chain = self._calculate_supply_chain_completeness(cluster) * 100
        density = self._calculate_geographic_density(cluster) * 100
        diversity = self._calculate_workforce_diversity(cluster) * 100
        
        # Weighted calculation
        sustainability = (
            critical_mass * 0.5 +
            supply_chain * 0.2 +
            density * 0.15 +
            diversity * 0.15
        )
        
        # Scale to percentage
        return min(100, sustainability * 1.2)
    
    def _generate_explanation(self, features: Dict, predictions: Dict) -> Dict:
        """Generate human-readable explanations for predictions"""
        explanation = {
            'key_drivers': [],
            'strengths': [],
            'risks': [],
            'recommendations': []
        }
        
        # Analyze key drivers
        if features.get('critical_mass', 0) > 70:
            explanation['key_drivers'].append("High critical mass ensures ecosystem sustainability")
        
        if features.get('strategic_score', 0) > 80:
            explanation['key_drivers'].append("Strong strategic positioning drives growth potential")
        
        # KC features impact
        if features.get('kc_market_access', 0) > 80:
            explanation['key_drivers'].append("Excellent KC market access enhances opportunities")
        
        if features.get('kc_infrastructure_score', 0) > 85:
            explanation['key_drivers'].append("Superior infrastructure supports scalability")
        
        # Identify strengths
        if predictions.get('roi_percentage', 0) > 20:
            explanation['strengths'].append(f"High ROI potential: {predictions['roi_percentage']:.1f}%")
        
        if predictions.get('job_creation', 0) > 10000:
            explanation['strengths'].append(f"Significant job creation: {predictions['job_creation']:,}")
        
        # Identify risks
        if features.get('critical_mass', 0) < 50:
            explanation['risks'].append("Low critical mass may impact long-term viability")
        
        if features.get('kc_crime_safety', 0) < 60:
            explanation['risks'].append("Safety concerns may affect business attraction")
        
        # Generate recommendations
        if features.get('supply_chain_completeness', 0) < 0.5:
            explanation['recommendations'].append("Strengthen supply chain partnerships")
        
        if features.get('workforce_diversity', 0) < 0.5:
            explanation['recommendations'].append("Diversify business sizes for resilience")
        
        return explanation

    def _debias_importance(self, mapping: Dict[str, float], target: str = 'gdp_impact') -> Dict[str, float]:
        """Return a version of feature importance with scale-dominant variables removed.

        Rationale: Our training targets for GDP and jobs are proportional to
        `total_revenue` and `total_employees` respectively. Raw importances and
        SHAP therefore concentrate almost entirely on these scale features. For
        interpretability, we remove those base-scale variables and re-normalize
        the remainder to show which factors drive the multiplier effects.
        """
        if not mapping:
            return {}
        # Remove pure scale drivers so the chart reflects multiplier drivers
        drop = {'business_count'}
        if target == 'gdp_impact':
            drop.update({'total_revenue'})
        elif target == 'job_creation':
            drop.update({'total_employees'})
        elif target == 'roi_percentage':
            # ROI is a ratio; guard against any residual scale dominance
            drop.update({'total_revenue', 'total_employees'})
        filtered = {k: v for k, v in mapping.items() if k not in drop}
        s = sum(abs(v) for v in filtered.values())
        if s <= 0:
            return mapping  # fall back to raw if nothing remains
        return {k: float(abs(v) / s) for k, v in filtered.items()}

    def _compute_feature_importance_shap(self, features_dict: Dict) -> Dict[str, float]:
        """Compute per-cluster feature importance using SHAP on the active model.

        Prefers the 18-feature model; falls back to 13-feature. Returns a
        normalized mapping feature -> importance (sum to ~1). If SHAP is not
        available or fails, returns an empty dict.
        """
        try:
            if not self.ensemble_predictor:
                return {}
            ep = self.ensemble_predictor
            import numpy as _np
            import pandas as _pd
            import shap as _shap

            # Choose target and model (GDP impact importance is most interpretable)
            target = 'gdp_impact'
            use_18 = target in ep.models_18
            model = ep.models_18.get(target) if use_18 else ep.models_13.get(target)
            if model is None:
                return {}

            feature_names = ep.feature_names_18 if use_18 else ep.feature_names_13
            scaler = (ep.scalers_18.get(target) if use_18 else ep.scalers_13.get(target))

            # Build one-row DataFrame in expected feature order
            row = [features_dict.get(name, 0) for name in feature_names]
            X = _pd.DataFrame([row], columns=feature_names)
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values

            # SHAP explainer (Tree-based models)
            try:
                explainer = _shap.Explainer(model)
                sv = explainer(_np.array(X_scaled))
                shap_vals = sv.values[0]
            except Exception:
                # Fallback to TreeExplainer path if generic fails
                explainer = _shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(_np.array(X_scaled))[0]

            # Normalize absolute shap values
            abs_vals = _np.abs(_np.array(shap_vals, dtype=float))
            denom = float(abs_vals.sum()) or 1.0
            norm = abs_vals / denom
            n = min(len(feature_names), len(norm))
            mapping = {feature_names[i]: float(norm[i]) for i in range(n)}
            # Keep top-N to reduce payload size
            top = dict(sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)[:20])
            return top
        except Exception:
            return {}
    
    def _calculate_confidence(self, features: Dict, predictions: Dict) -> float:
        """Calculate confidence score for predictions"""
        base_confidence = predictions.get('confidence', 0.5)
        
        # Adjust based on data completeness
        feature_completeness = sum(1 for v in features.values() if v and v != 0) / len(features)
        
        # Adjust based on KC data quality if available
        kc_quality = predictions.get('kc_data_quality', 0)
        
        # Weighted confidence
        confidence = (
            base_confidence * 0.6 +
            feature_completeness * 0.3 +
            kc_quality * 0.1
        )
        
        return min(1.0, confidence)
    
    def _rerank_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Re-rank clusters based on ML insights and sustainability"""
        
        for cluster in clusters:
            predictions = cluster.get('ml_predictions', {})
            
            # Calculate composite score
            gdp_score = predictions.get('gdp_impact', 0) / 1e9  # Billions
            job_score = predictions.get('job_creation', 0) / 1000  # Thousands
            roi_score = predictions.get('roi_percentage', 0)
            sustainability = cluster.get('sustainability_score', 0)
            
            # Weighted composite
            composite = (
                gdp_score * 0.25 +
                job_score * 0.25 +
                roi_score * 0.25 +
                sustainability * 0.25
            )
            
            cluster['ml_composite_score'] = composite
        
        # Sort by composite score
        clusters.sort(key=lambda x: x.get('ml_composite_score', 0), reverse=True)
        
        return clusters
    
    def _load_fallback_models(self):
        """Load pre-trained fallback models"""
        try:
            # Try loading conservative models
            gdp_path = os.path.join(self.model_path, 'gdp_model_conservative.pkl')
            job_path = os.path.join(self.model_path, 'job_model_conservative.pkl')
            roi_path = os.path.join(self.model_path, 'roi_model_conservative.pkl')
            scaler_path = os.path.join(self.model_path, 'scaler_conservative.pkl')
            
            if os.path.exists(gdp_path):
                with open(gdp_path, 'rb') as f:
                    self.gdp_model = pickle.load(f)
                logger.info("Loaded GDP model")
            
            if os.path.exists(job_path):
                with open(job_path, 'rb') as f:
                    self.job_model = pickle.load(f)
                logger.info("Loaded job model")
            
            if os.path.exists(roi_path):
                with open(roi_path, 'rb') as f:
                    self.roi_model = pickle.load(f)
                logger.info("Loaded ROI model")
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded scaler")
                
        except Exception as e:
            logger.warning(f"Could not load fallback models: {e}")
    
    def validate_predictions(self, test_clusters: List[Dict]) -> Dict:
        """Validate predictions against test data"""
        results = {
            'predictions': [],
            'metrics': {}
        }
        
        for cluster in test_clusters:
            features = self._extract_cluster_features(cluster)
            
            if self.ensemble_predictor:
                predictions = self._make_ensemble_predictions(features)
            else:
                predictions = self._predict_outcomes(features)
            
            results['predictions'].append({
                'cluster': cluster.get('name', 'Unknown'),
                'predictions': predictions,
                'confidence': predictions.get('confidence', 0)
            })
        
        # Calculate aggregate metrics
        avg_confidence = np.mean([p['confidence'] for p in results['predictions']])
        results['metrics']['average_confidence'] = avg_confidence
        
        return results


# Create a singleton instance for backward compatibility
_ml_enhancer_instance = None

def get_ml_enhancer(config=None):
    """Get or create ML enhancer instance"""
    global _ml_enhancer_instance
    if _ml_enhancer_instance is None:
        _ml_enhancer_instance = MLClusterEnhancerV2(config)
    return _ml_enhancer_instance

# Alias for backward compatibility
MLClusterEnhancer = MLClusterEnhancerV2
