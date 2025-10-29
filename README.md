# Kansas City Economic Cluster Prediction Tool

A comprehensive Python-based tool for predicting and optimizing economic development clusters in the Kansas City Metropolitan Statistical Area (MSA). This tool implements advanced data collection, business scoring, and multi-objective optimization to identify high-probability economic clusters.

## Features

### Core Capabilities
- **Automated Data Collection**: Scrapes business data from public sources (state registries, BLS, USPTO, SBIR)
- **Business-Level Analysis**: Scores individual businesses on innovation, market potential, and competitive position
- **Cluster Optimization**: Uses NSGA-II genetic algorithm for multi-objective cluster formation
- **Economic Impact Projection**: Estimates GDP impact, job creation, and wage growth
- **Stakeholder Recommendations**: Provides actionable insights for entrepreneurs, investors, universities, and policymakers

### Key Improvements Over Original Framework
1. **Enhanced Data Collection**
   - Concurrent scraping with thread pools for faster data gathering
   - Cross-validation across multiple sources for 95% confidence
   - Real-time market trend integration via news APIs
   - Automated inference for missing data from comparable regions

2. **Advanced Business Scoring**
   - Dynamic market growth rates updated from real-time data
   - Resource fit scoring based on infrastructure alignment
   - Revenue estimation using industry-specific models
   - Patent and SBIR/STTR integration for innovation metrics

3. **Sophisticated Cluster Optimization**
   - Multi-objective optimization balancing 5 key factors
   - Synergy scoring for supply chain linkages
   - Risk assessment for cluster resilience
   - Pareto front generation for optimal trade-offs

4. **Comprehensive Analysis**
   - Longevity scoring (0-10 scale) for long-term viability
   - Target alignment validation against economic goals
   - Industry-specific multipliers for accurate projections
   - Geographic concentration analysis

5. **Actionable Outputs**
   - Specific business opportunities by cluster type
   - ROI projections for investors
   - Research direction for universities
   - Policy recommendations with priority levels

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd cluster_prediction_tool

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

Edit `config.py` to customize:
- Data source URLs and API keys
- Scoring weights and thresholds
- Economic targets (GDP, jobs, wages)
- Geographic boundaries
- Optimization parameters

## Usage

### Basic Usage

```python
from main import ClusterPredictionTool

# Initialize tool
tool = ClusterPredictionTool()

# Run full analysis
results = tool.run_full_analysis()

# Generate report
report = tool.generate_report(results)
print(report)
```

### Command Line

```bash
# Run complete analysis
python main.py

# Output files:
# - cluster_analysis_results.json (detailed results)
# - cluster_analysis_report.md (formatted report)
```

## Architecture

```
cluster_prediction_tool/
├── config.py              # Configuration settings
├── models.py              # SQLAlchemy data models
├── main.py               # Main application
├── data_collection/      # Data scraping modules
│   └── scraper.py       # Web scraping implementation
├── analysis/            # Analysis modules
│   ├── business_scorer.py    # Business evaluation
│   └── cluster_optimizer.py  # NSGA-II optimization
└── api/                 # REST API (future)
```

## Data Sources

### Primary (Free/Public)
- **Business Registries**: Kansas and Missouri Secretary of State
- **Employment Data**: Bureau of Labor Statistics (BLS)
- **Innovation Data**: USPTO, SBIR.gov
- **Economic Data**: Bureau of Economic Analysis (BEA)
- **Infrastructure**: Federal Railroad Administration, FCC

### APIs for Enhanced Analysis
- BLS API: `https://api.bls.gov/publicAPI/v2/`
- USPTO API: `https://developer.uspto.gov/ibd-api/v1/`
- Census API: For demographic data
- News APIs: For real-time market trends

## Economic Targets

The tool optimizes clusters to achieve:
- **GDP Growth**: Minimum $2.87 billion (0.5% of Kansas + Missouri)
- **Job Creation**: 1,000 direct + 2,000 indirect/induced jobs
- **Wage Growth**: Increase median wages in urban core

## Cluster Types

### 1. Logistics & Distribution (Score: 85/100)
- Leverages KC's Class-I rail network
- Focus: Last-mile delivery, cold chain, warehousing
- Job potential: 500-700 per major facility

### 2. Biosciences & Healthcare (Score: 75/100)
- Anchored by research institutions
- Focus: Pharmaceuticals, medical devices, digital health
- High-wage jobs: $80,000-$100,000 average

### 3. Technology & Innovation (Score: 76/100)
- Growing startup ecosystem
- Focus: FinTech, AgTech, cybersecurity
- Rapid growth potential

### 4. Manufacturing (Score: 80/100)
- Existing industrial base
- Focus: Advanced manufacturing, food processing
- Stable employment growth

### 5. Animal Health (Score: 68/100)
- National leadership position
- Focus: Veterinary pharmaceuticals, nutrition
- Niche market advantages

## Output Examples

### Business Ranking
```
Top Businesses by Composite Score:
1. Logistics Co A - Score: 73.3 (Innovation: 50, Market: 90, Competition: 80)
2. BioTech Co B - Score: 71.2 (Innovation: 80, Market: 75, Competition: 60)
3. Tech Startup C - Score: 70.0 (Innovation: 70, Market: 75, Competition: 65)
```

### Cluster Configuration
```
Logistics Cluster 1:
- Businesses: 45
- Total Employees: 3,250
- Projected GDP Impact: $875M
- Projected Jobs: 8,125 (3,250 direct + 4,875 indirect)
- Longevity Score: 8.5/10
```

### Recommendations
```
For Entrepreneurs:
- Last-mile delivery services (Investment: $100K-$500K)
- Cold chain logistics (Investment: $500K-$2M)

For Universities:
- Research: Supply chain optimization using AI/ML
- Partners: [List of high-potential companies]
```

## Performance Metrics

- Data Collection: 95% accuracy target
- Processing Time: <10 minutes for 5,000 businesses
- Optimization: 50 generations, 100 population size
- Memory Usage: <2GB RAM for typical analysis

## Future Enhancements

1. **Real-time Dashboard**: Web interface for interactive analysis
2. **Machine Learning**: Predictive models for cluster success
3. **Geographic Expansion**: Adapt framework for other MSAs
4. **API Integration**: RESTful API for third-party access
5. **Blockchain**: Supply chain transparency features

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Kansas City Development Corporation
- University of Missouri-Kansas City
- Kansas Department of Commerce
- Economic Development Administration

## Contact

For questions or collaboration opportunities, please contact the development team.