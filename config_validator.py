"""Configuration validation using Pydantic (v2)"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class EconomicTargets(BaseModel):
    gdp_growth: float = Field(gt=0, description="GDP growth target in dollars")
    direct_jobs: int = Field(gt=0, description="Direct jobs target")
    indirect_jobs: int = Field(gt=0, description="Indirect jobs target")
    wage_growth: float = Field(ge=0, le=1, description="Wage growth rate (0-1)")
    time_horizon: int = Field(gt=0, le=10, description="Time horizon in years")
    min_roi: float = Field(ge=0, description="Minimum ROI threshold")


class BusinessFilters(BaseModel):
    min_employees: int = Field(ge=0, default=2)
    max_employees: int = Field(gt=0, default=50000)
    min_revenue: float = Field(ge=0, default=25000)
    min_age: int = Field(ge=0, default=0)
    excluded_naics: List[str] = Field(default_factory=list)
    require_patents: bool = False
    require_sbir: bool = False
    min_growth_rate: float = Field(ge=-1, le=10, default=0)

    @model_validator(mode='after')
    def _validate_employee_range(self):
        if self.min_employees and self.max_employees and self.min_employees > self.max_employees:
            raise ValueError('min_employees cannot be greater than max_employees')
        return self


class GeographicScope(BaseModel):
    kansas_counties: List[str] = Field(default_factory=list)
    missouri_counties: List[str] = Field(default_factory=list)
    focus: str = Field(pattern=r'^(urban|suburban|rural|both|all)$', default='both')

    @field_validator('kansas_counties', 'missouri_counties')
    @classmethod
    def validate_counties(cls, v: List[str]):
        valid = {
            "Johnson County", "Leavenworth County", "Linn County",
            "Miami County", "Wyandotte County", "Bates County",
            "Caldwell County", "Cass County", "Clay County",
            "Clinton County", "Jackson County", "Lafayette County",
            "Platte County", "Ray County"
        }
        for c in v:
            normalized = c.replace(', KS', '').replace(', MO', '')
            if normalized not in valid:
                raise ValueError(f"Invalid county: {c}")
        return v


class AlgorithmParams(BaseModel):
    num_clusters: Optional[int] = Field(None, ge=1, le=20)
    cluster_size: Optional[Dict[str, int]] = None
    optimization_focus: str = Field(pattern=r'^(balanced|gdp|jobs|innovation)$', default='balanced')
    population_size: int = Field(ge=10, le=500, default=100)
    generations: int = Field(ge=10, le=200, default=50)
    crossover_prob: float = Field(ge=0, le=1, default=0.7)
    mutation_prob: float = Field(ge=0, le=1, default=0.2)

    @model_validator(mode='after')
    def _validate_cluster_size(self):
        cs = self.cluster_size
        if cs:
            if 'min' not in cs or 'max' not in cs:
                raise ValueError('cluster_size must have min and max')
            if cs['min'] > cs['max']:
                raise ValueError('cluster_size min cannot be greater than max')
            if cs['min'] < 5:
                raise ValueError('cluster_size min should be at least 5')
        return self


class AnalysisParams(BaseModel):
    economic_targets: Optional[EconomicTargets] = None
    business_filters: Optional[BusinessFilters] = None
    geographic_scope: Optional[GeographicScope] = None
    algorithm_params: Optional[AlgorithmParams] = None
    quick_mode: bool = False
    sample_size: Optional[int] = Field(None, gt=100, le=100000)
    use_ml_enhancement: bool = True
    use_kc_features: bool = True
    # Make optional so we can distinguish between 'not provided' and explicit False
    skip_patents: Optional[bool] = None
    data_sources: List[str] = Field(default_factory=list)

    @field_validator('data_sources')
    @classmethod
    def validate_data_sources(cls, v: List[str]):
        valid = {
            'state_registrations', 'bls_employment', 'uspto_patents',
            'sbir_awards', 'university_partners', 'infrastructure_assets'
        }
        for s in v:
            if s not in valid:
                raise ValueError(f"Invalid data source: {s}")
        return v

    @model_validator(mode='after')
    def _validate_quick_mode(self):
        # Default sampling only if not explicitly provided
        if self.quick_mode and self.sample_size is None:
            self.sample_size = 10000
        # Respect explicit skip_patents=False; only set True when not provided at all
        if self.quick_mode and self.skip_patents is None:
            self.skip_patents = True
        return self


def validate_analysis_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize analysis parameters"""
    model = AnalysisParams(**(params or {}))
    # Exclude fields not provided (e.g., skip_patents when None)
    return model.model_dump(exclude_none=True)
