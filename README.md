# Supplier Risk Analyzer

An AI-powered tool for comprehensive supplier risk assessment, contract analysis, and performance evaluation.

## Overview

The Supplier Risk Analyzer is a Streamlit-based application that uses AI agents (powered by Google's Agent Development Kit and Gemini AI) to analyze supplier data across multiple dimensions:

1. **Performance Analysis**: Evaluates quality metrics, delivery performance, and cost efficiency
2. **Contract Analysis**: Reviews contract terms, identifies key clauses and risks, and recommends negotiation strategies
3. **Risk Assessment**: Identifies financial, supply chain, geopolitical, and compliance risks with mitigation strategies

## Features

- **Integrated AI Agents**: Sequential AI agents that work together to provide comprehensive analysis
- **Interactive Visualizations**: Dynamic charts and visual representations of supplier metrics and risks
- **Multiple Data Input Methods**: Use sample data (Apple Inc.), upload JSON, or manually enter supplier information
- **Detailed Recommendations**: Actionable insights for performance improvement and risk mitigation
- **User-Friendly Interface**: Clean, tab-based interface with expandable sections for detailed information

## Key Components

- **Performance Metrics**: Quality, delivery, and cost metrics with trend analysis
- **Contract Analysis**: Risk assessment of contract clauses, negotiation tactics, and pricing targets
- **Risk Radar**: Comprehensive view of financial, supply chain, geopolitical, and compliance risks
- **Mitigation Strategies**: Specific actions to address identified risks with implementation timelines

## Requirements

- Python 3.8+
- Streamlit
- Google ADK (Agent Development Kit)
- Plotly for visualizations
- Pandas for data handling
- Python-dotenv for environment variables

## Installation

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your Google API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

1. Run the application: `streamlit run supplier_risk_analyzer.py`
2. Select data input method (sample data, upload JSON, or manual entry)
3. Review and optionally edit the supplier data
4. Click "Analyze Supplier" to run the AI analysis
5. Navigate through the results tabs to view the comprehensive supplier assessment

## Sample Data

The application includes a detailed sample profile for Apple Inc. as a supplier, with realistic data about:
- Performance metrics
- Contract terms and clauses
- Financial stability
- Geographic manufacturing distribution
- Supply chain dependencies
- ESG (Environmental, Social, Governance) factors
- Dependency relationships

## License

MIT License
