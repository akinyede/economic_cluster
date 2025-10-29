"""Scenario comparison and management module for KC Cluster Prediction Tool"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ScenarioManager:
    """Manage and compare multiple analysis scenarios"""
    
    def __init__(self, storage_path: str = "scenarios"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
    def save_scenario(self, scenario_name: str, params: Dict, results: Dict) -> str:
        """Save a scenario for later comparison"""
        scenario_id = f"{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        scenario_data = {
            "id": scenario_id,
            "name": scenario_name,
            "timestamp": datetime.now().isoformat(),
            "parameters": params,
            "results": results
        }
        
        # Save to file
        file_path = os.path.join(self.storage_path, f"{scenario_id}.json")
        with open(file_path, 'w') as f:
            json.dump(scenario_data, f, indent=2, default=str)
        
        logger.info(f"Saved scenario: {scenario_id}")
        return scenario_id
    
    def load_scenario(self, scenario_id: str) -> Optional[Dict]:
        """Load a saved scenario"""
        file_path = os.path.join(self.storage_path, f"{scenario_id}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        
        logger.warning(f"Scenario not found: {scenario_id}")
        return None
    
    def list_scenarios(self) -> List[Dict]:
        """List all saved scenarios"""
        scenarios = []
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.storage_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    scenarios.append({
                        "id": data.get("id"),
                        "name": data.get("name"),
                        "timestamp": data.get("timestamp"),
                        "params_summary": self._summarize_params(data.get("parameters", {}))
                    })
        
        # Sort by timestamp descending
        scenarios.sort(key=lambda x: x['timestamp'], reverse=True)
        return scenarios
    
    def compare_scenarios(self, scenario_ids: List[str]) -> Dict:
        """Compare multiple scenarios side by side"""
        scenarios = []
        
        for scenario_id in scenario_ids:
            scenario = self.load_scenario(scenario_id)
            if scenario:
                scenarios.append(scenario)
        
        if len(scenarios) < 2:
            return {"error": "Need at least 2 scenarios to compare"}
        
        comparison = {
            "scenarios": [],
            "parameter_differences": self._compare_parameters(scenarios),
            "result_comparison": self._compare_results(scenarios),
            "visualizations": self._create_comparison_visualizations(scenarios)
        }
        
        # Add scenario summaries
        for scenario in scenarios:
            comparison["scenarios"].append({
                "id": scenario["id"],
                "name": scenario["name"],
                "timestamp": scenario["timestamp"]
            })
        
        return comparison
    
    def _summarize_params(self, params: Dict) -> str:
        """Create a brief summary of scenario parameters"""
        summary_parts = []
        
        if "economic_targets" in params:
            targets = params["economic_targets"]
            if "gdp_growth" in targets:
                summary_parts.append(f"GDP: ${targets['gdp_growth']/1e9:.1f}B")
            if "direct_jobs" in targets:
                summary_parts.append(f"Jobs: {targets['direct_jobs']}")
        
        if "geographic_scope" in params:
            geo = params["geographic_scope"]
            county_count = len(geo.get("kansas_counties", [])) + len(geo.get("missouri_counties", []))
            summary_parts.append(f"Counties: {county_count}")
        
        if "algorithm_params" in params:
            algo = params["algorithm_params"]
            if "num_clusters" in algo:
                summary_parts.append(f"Clusters: {algo['num_clusters']}")
        
        return ", ".join(summary_parts)
    
    def _compare_parameters(self, scenarios: List[Dict]) -> Dict:
        """Compare parameters across scenarios"""
        differences = {
            "all_same": True,
            "differences": []
        }
        
        # Get all parameter keys
        all_keys = set()
        for scenario in scenarios:
            all_keys.update(self._flatten_dict(scenario.get("parameters", {})).keys())
        
        # Compare each parameter
        for key in sorted(all_keys):
            values = []
            for scenario in scenarios:
                flat_params = self._flatten_dict(scenario.get("parameters", {}))
                values.append(flat_params.get(key, "Not set"))
            
            # Check if all values are the same
            if len(set(str(v) for v in values)) > 1:
                differences["all_same"] = False
                differences["differences"].append({
                    "parameter": key,
                    "values": {scenarios[i]["id"]: values[i] for i in range(len(scenarios))}
                })
        
        return differences
    
    def _compare_results(self, scenarios: List[Dict]) -> Dict:
        """Compare key results across scenarios"""
        comparison = {
            "economic_impact": [],
            "clusters": [],
            "businesses": [],
            "performance": []
        }
        
        for scenario in scenarios:
            results = scenario.get("results", {})
            impact = results.get("economic_impact", {})
            
            # Economic impact comparison
            comparison["economic_impact"].append({
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "gdp_impact": impact.get("projected_gdp_impact", 0),
                "direct_jobs": impact.get("projected_direct_jobs", 0),
                "total_jobs": impact.get("projected_total_jobs", 0),
                "meets_targets": impact.get("meets_targets", False),
                "gdp_achievement": impact.get("gdp_target_achievement", 0),
                "jobs_achievement": impact.get("jobs_target_achievement", 0)
            })
            
            # Cluster comparison
            clusters = results.get("steps", {}).get("cluster_optimization", {}).get("clusters", [])
            comparison["clusters"].append({
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "num_clusters": len(clusters),
                "top_cluster": clusters[0].get("name") if clusters else "None",
                "top_cluster_score": clusters[0].get("total_score", 0) if clusters else 0,
                "cluster_types": [c.get("type") for c in clusters[:3]]
            })
            
            # Business analysis comparison
            business_data = results.get("steps", {}).get("business_scoring", {})
            comparison["businesses"].append({
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "total_businesses": business_data.get("total_businesses", 0),
                "passing_threshold": business_data.get("passing_threshold", 0),
                "avg_score": business_data.get("avg_composite_score", 0)
            })
        
        # Calculate performance rankings
        comparison["performance"] = self._rank_scenarios(comparison)
        
        return comparison
    
    def _create_comparison_visualizations(self, scenarios: List[Dict]) -> Dict:
        """Create visualization data for scenario comparison"""
        viz = {
            "gdp_comparison": {
                "labels": [],
                "values": []
            },
            "jobs_comparison": {
                "labels": [],
                "direct": [],
                "total": []
            },
            "cluster_scores": {
                "labels": [],
                "scores": []
            },
            "target_achievement": {
                "labels": [],
                "gdp": [],
                "jobs": []
            }
        }
        
        for scenario in scenarios:
            label = f"{scenario['name']} ({scenario['id'][:8]})"
            viz["gdp_comparison"]["labels"].append(label)
            viz["jobs_comparison"]["labels"].append(label)
            viz["cluster_scores"]["labels"].append(label)
            viz["target_achievement"]["labels"].append(label)
            
            # Extract data
            impact = scenario.get("results", {}).get("economic_impact", {})
            viz["gdp_comparison"]["values"].append(impact.get("projected_gdp_impact", 0) / 1e9)  # Billions
            viz["jobs_comparison"]["direct"].append(impact.get("projected_direct_jobs", 0))
            viz["jobs_comparison"]["total"].append(impact.get("projected_total_jobs", 0))
            viz["target_achievement"]["gdp"].append(impact.get("gdp_target_achievement", 0))
            viz["target_achievement"]["jobs"].append(impact.get("jobs_target_achievement", 0))
            
            # Top cluster score
            clusters = scenario.get("results", {}).get("steps", {}).get("cluster_optimization", {}).get("clusters", [])
            top_score = clusters[0].get("total_score", 0) if clusters else 0
            viz["cluster_scores"]["scores"].append(top_score)
        
        return viz
    
    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """Flatten nested dictionary for comparison"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _rank_scenarios(self, comparison: Dict) -> List[Dict]:
        """Rank scenarios by overall performance"""
        rankings = []
        
        for i, impact in enumerate(comparison["economic_impact"]):
            score = 0
            
            # GDP impact weight: 40%
            gdp_scores = [e["gdp_impact"] for e in comparison["economic_impact"]]
            if max(gdp_scores) > 0:
                score += 40 * (impact["gdp_impact"] / max(gdp_scores))
            
            # Jobs impact weight: 30%
            job_scores = [e["total_jobs"] for e in comparison["economic_impact"]]
            if max(job_scores) > 0:
                score += 30 * (impact["total_jobs"] / max(job_scores))
            
            # Target achievement weight: 20%
            gdp_achieve = impact["gdp_achievement"] / 100 if impact["gdp_achievement"] < 100 else 1
            jobs_achieve = impact["jobs_achievement"] / 100 if impact["jobs_achievement"] < 100 else 1
            score += 10 * gdp_achieve + 10 * jobs_achieve
            
            # Cluster quality weight: 10%
            cluster_scores = [c["top_cluster_score"] for c in comparison["clusters"]]
            if max(cluster_scores) > 0:
                score += 10 * (comparison["clusters"][i]["top_cluster_score"] / max(cluster_scores))
            
            rankings.append({
                "scenario_id": impact["scenario_id"],
                "scenario_name": impact["scenario_name"],
                "overall_score": round(score, 1),
                "rank": 0  # Will be set after sorting
            })
        
        # Sort and assign ranks
        rankings.sort(key=lambda x: x["overall_score"], reverse=True)
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
        
        return rankings
    
    def generate_comparison_report(self, comparison: Dict) -> str:
        """Generate a text report comparing scenarios"""
        report = "# Scenario Comparison Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Scenarios compared
        report += "## Scenarios Compared\n"
        for scenario in comparison["scenarios"]:
            report += f"- **{scenario['name']}** (ID: {scenario['id']})\n"
        
        # Parameter differences
        report += "\n## Parameter Differences\n"
        if comparison["parameter_differences"]["all_same"]:
            report += "All parameters are identical across scenarios.\n"
        else:
            for diff in comparison["parameter_differences"]["differences"][:10]:  # Top 10
                report += f"\n**{diff['parameter']}:**\n"
                for scenario_id, value in diff["values"].items():
                    report += f"  - {scenario_id[:8]}: {value}\n"
        
        # Performance rankings
        report += "\n## Performance Rankings\n"
        rankings = comparison["result_comparison"]["performance"]
        for rank in rankings:
            report += f"{rank['rank']}. **{rank['scenario_name']}** - Score: {rank['overall_score']}/100\n"
        
        # Economic impact comparison
        report += "\n## Economic Impact Comparison\n"
        report += "| Scenario | GDP Impact | Total Jobs | Targets Met |\n"
        report += "|----------|------------|------------|-------------|\n"
        for impact in comparison["result_comparison"]["economic_impact"]:
            meets = "✓" if impact["meets_targets"] else "✗"
            report += f"| {impact['scenario_name']} | ${impact['gdp_impact']/1e9:.2f}B | {impact['total_jobs']:,} | {meets} |\n"
        
        # Cluster comparison
        report += "\n## Cluster Analysis Comparison\n"
        report += "| Scenario | # Clusters | Top Cluster | Score | Types |\n"
        report += "|----------|------------|-------------|-------|-------|\n"
        for cluster in comparison["result_comparison"]["clusters"]:
            types = ", ".join(cluster["cluster_types"])
            report += f"| {cluster['scenario_name']} | {cluster['num_clusters']} | {cluster['top_cluster']} | {cluster['top_cluster_score']:.1f} | {types} |\n"
        
        return report