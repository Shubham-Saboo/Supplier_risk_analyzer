import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.tools import google_search

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "supplier_management"
USER_ID = "default_user"

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Pydantic models for data schemas
class PerformanceMetric(BaseModel):
    metric_name: str = Field(..., description="Name of the performance metric")
    score: float = Field(..., description="Current score (0-100)")
    trend: str = Field(..., description="Trend direction: Improving, Stable, Declining")
    analysis: str = Field(..., description="Analysis of this metric's performance")

class PerformanceRecommendation(BaseModel):
    area: str = Field(..., description="Area for improvement")
    recommendation: str = Field(..., description="Specific recommendation")
    potential_impact: str = Field(..., description="Estimated business impact")

class SupplierPerformanceAnalysis(BaseModel):
    overall_score: float = Field(..., description="Overall supplier performance score (0-100)")
    quality_metrics: List[PerformanceMetric] = Field(..., description="Quality-related metrics")
    delivery_metrics: List[PerformanceMetric] = Field(..., description="Delivery-related metrics") 
    cost_metrics: List[PerformanceMetric] = Field(..., description="Cost-related metrics")
    recommendations: List[PerformanceRecommendation] = Field(..., description="Performance improvement recommendations")

class ContractClause(BaseModel):
    clause_name: str = Field(..., description="Name of the contract clause")
    risk_level: str = Field(..., description="Risk level: Low, Medium, High")
    description: str = Field(..., description="Description of the clause")
    expiration_date: Optional[str] = Field(None, description="Date when this clause expires")

class NegotiationTactic(BaseModel):
    tactic_name: str = Field(..., description="Name of negotiation tactic")
    description: str = Field(..., description="Description of how to use this tactic")
    when_to_use: str = Field(..., description="When this tactic is most effective")

class PricingTarget(BaseModel):
    category: str = Field(..., description="Category of goods/services")
    current_price: float = Field(..., description="Current price")
    target_price: float = Field(..., description="Target price to achieve")
    justification: str = Field(..., description="Justification for the target price")

class ContractAnalysis(BaseModel):
    overall_risk_level: str = Field(..., description="Overall contract risk level: Low, Medium, High")
    key_clauses: List[ContractClause] = Field(..., description="Key clauses in the contract")
    expiration_date: Optional[str] = Field(None, description="Contract expiration date")
    renewal_notice_period: Optional[int] = Field(None, description="Days required for renewal notice")
    negotiation_approach: str = Field(..., description="Recommended negotiation approach")
    key_leverage_points: List[str] = Field(..., description="Key points of leverage for negotiations")
    tactics: List[NegotiationTactic] = Field(..., description="Recommended negotiation tactics")
    pricing_targets: List[PricingTarget] = Field(..., description="Target pricing for negotiation")

class RiskFactor(BaseModel):
    name: str = Field(..., description="Name of risk factor")
    level: str = Field(..., description="Current risk level: Low, Medium, High")
    trend: str = Field(..., description="Risk trend: Increasing, Stable, Decreasing")
    impact: str = Field(..., description="Potential business impact")

class MitigationStrategy(BaseModel):
    risk_name: str = Field(..., description="Name of risk being mitigated")
    strategy: str = Field(..., description="Mitigation strategy description")
    implementation_time: str = Field(..., description="Estimated implementation time")
    expected_result: str = Field(..., description="Expected outcome after implementation")

class SupplierRiskAssessment(BaseModel):
    overall_risk_score: float = Field(..., description="Overall risk score (0-100)")
    financial_risks: List[RiskFactor] = Field(..., description="Financial risk factors")
    supply_chain_risks: List[RiskFactor] = Field(..., description="Supply chain risk factors")
    geopolitical_risks: List[RiskFactor] = Field(..., description="Geopolitical risk factors")
    compliance_risks: List[RiskFactor] = Field(..., description="Compliance/regulatory risk factors")
    mitigation_strategies: List[MitigationStrategy] = Field(..., description="Risk mitigation strategies")

class SupplierManagementSystem:
    """
    AI-powered system for analyzing supplier data and providing actionable insights
    """
    
    def __init__(self):
        """Initialize the supplier management system"""
        self.session_service = InMemorySessionService()
        
        self.supplier_performance_agent = LlmAgent(
            name="SupplierPerformanceAgent",
            model="gemini-2.0-flash-exp",
            description="Analyzes supplier performance metrics and provides actionable insights",
            instruction="""You are a Supplier Performance Analysis Agent specialized in evaluating supplier metrics.
You are the first agent in a sequence of three supplier management agents.

Your tasks:
1. Analyze quality metrics (defect rates, returns, compliance)
2. Evaluate delivery performance (on-time rate, lead times, fulfillment)
3. Assess cost metrics (price trends, cost savings)
4. Identify performance trends and improvement areas
5. Create a performance scorecard with actionable insights

Consider:
- Historical performance trends over time
- Industry benchmarks for similar suppliers
- Critical vs. non-critical suppliers
- The impact of performance issues on production

IMPORTANT: Store your analysis in state['performance_analysis'] for use by subsequent agents.""",
            output_schema=SupplierPerformanceAnalysis,
            output_key="performance_analysis"
        )
        
        self.contract_analysis_agent = LlmAgent(
            name="ContractAnalysisAgent",
            model="gemini-2.0-flash-exp",
            description="Analyzes supplier contracts and develops negotiation strategies",
            instruction="""You are a Contract Analysis & Negotiation Agent specialized in reviewing contracts and developing negotiation strategies.
You are the second agent in a sequence of three supplier management agents. READ state['performance_analysis'] first.

Your tasks:
1. Identify key contract clauses and their implications
2. Assess risk levels for different contract components
3. Track important dates (renewals, expirations, etc.)
4. Develop tailored negotiation approaches based on contract terms
5. Identify key leverage points based on supplier dependencies
6. Recommend specific negotiation tactics and timing
7. Set target pricing goals with justifications

Consider:
- Pricing terms and mechanisms
- Service level agreements and penalties
- Liability and indemnification clauses
- Performance issues identified by the Performance Analysis Agent
- Supplier's financial health and market position
- Your organization's bargaining power
- Market conditions and alternative suppliers

IMPORTANT: Store your analysis in state['contract_analysis'] for use by the Risk Assessment Agent.""",
            output_schema=ContractAnalysis,
            output_key="contract_analysis"
        )
        
        self.supplier_risk_agent = SupplierRiskAgent()
        
        self.coordinator_agent = SequentialAgent(
            name="SupplierManagementCoordinator",
            description="Coordinates specialized supplier management agents to optimize supplier relationships",
            sub_agents=[
                self.supplier_performance_agent,
                self.contract_analysis_agent,
                self.supplier_risk_agent
            ]
        )
        
        self.runner = Runner(
            agent=self.coordinator_agent,
            app_name="supplier_management",
            session_service=self.session_service
        )

    async def analyze_supplier(self, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze supplier data using specialized agents
        
        Args:
            supplier_data: Dictionary containing supplier information
            
        Returns:
            Dictionary with analysis results from all agents
        """
        session_id = f"supplier_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            initial_state = {
                "supplier_name": supplier_data.get("supplier_name", "Unknown Supplier"),
                "spend": supplier_data.get("spend", 0),
                "category": supplier_data.get("category", "Uncategorized"),
                "performance_metrics": supplier_data.get("performance_metrics", {}),
                "contract_data": supplier_data.get("contract_data", {}),
                "financial_data": supplier_data.get("financial_data", {})
            }
            
            session = self.session_service.create_session(
                app_name="supplier_management",
                user_id="default_user",
                session_id=session_id,
                state=initial_state
            )
            
            default_results = self._create_default_results(supplier_data)
            
            user_content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(supplier_data))]
            )
            
            async for event in self.runner.run_async(
                user_id="default_user",
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    break
            
            updated_session = self.session_service.get_session(
                app_name="supplier_management",
                user_id="default_user",
                session_id=session_id
            )
            
            results = {}
            for key in ["performance_analysis", "contract_analysis", "risk_assessment"]:
                value = updated_session.state.get(key)
                results[key] = parse_json_safely(value, default_results[key]) if value else default_results[key]
            
            return results
            
        except Exception as e:
            logger.exception(f"Error during supplier analysis: {str(e)}")
            raise
        finally:
            self.session_service.delete_session(
                app_name="supplier_management",
                user_id="default_user",
                session_id=session_id
            )
    
    def _create_default_results(self, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create default results if AI analysis fails"""
        supplier_name = supplier_data.get("supplier_name", "Unknown Supplier")
        spend = supplier_data.get("spend", 0)
        
        default_performance_metrics = {
            "quality_score": 75,
            "delivery_score": 80,
            "cost_score": 70
        }
        
        performance_metrics = supplier_data.get("performance_metrics", default_performance_metrics)
        
        return {
            "performance_analysis": {
                "overall_score": sum(performance_metrics.values()) / len(performance_metrics) if performance_metrics else 75,
                "quality_metrics": [
                    {"metric_name": "Defect Rate", "score": 70, "trend": "Stable", "analysis": "Defect rate is within acceptable range"}
                ],
                "delivery_metrics": [
                    {"metric_name": "On-Time Delivery", "score": 80, "trend": "Improving", "analysis": "On-time delivery rates have improved consistently"}
                ],
                "cost_metrics": [
                    {"metric_name": "Price Competitiveness", "score": 65, "trend": "Stable", "analysis": "Prices remain competitive but slightly above market average"}
                ],
                "recommendations": [
                    {"area": "Quality Control", "recommendation": "Implement joint quality reviews", "potential_impact": "Could reduce defect rates by 5-10%"}
                ]
            },
            "contract_analysis": {
                "overall_risk_level": "Medium",
                "key_clauses": [
                    {"clause_name": "Pricing Terms", "risk_level": "Medium", "description": "Fixed pricing with annual adjustments", "expiration_date": None}
                ],
                "expiration_date": "2024-12-31",
                "renewal_notice_period": 90,
                "negotiation_approach": "Collaborative with firm boundaries",
                "key_leverage_points": ["High volume of business", "Long-term relationship"],
                "tactics": [
                    {"tactic_name": "Bundled Ordering", "description": "Combine multiple orders for better pricing", "when_to_use": "When negotiating annual contracts"}
                ],
                "pricing_targets": [
                    {"category": "Raw Materials", "current_price": 100, "target_price": 95, "justification": "Market prices have decreased by 5%"}
                ]
            },
            "risk_assessment": {
                "overall_risk_score": 40,
                "financial_risks": [
                    {"name": "Financial Stability", "level": "Low", "trend": "Stable", "impact": "Low risk of supply disruption due to financial issues"}
                ],
                "supply_chain_risks": [
                    {"name": "Single Source Component", "level": "High", "trend": "Increasing", "impact": "Critical components have no alternative sources"}
                ],
                "geopolitical_risks": [
                    {"name": "Political Instability", "level": "Low", "trend": "Stable", "impact": "Operations not in politically volatile regions"}
                ],
                "compliance_risks": [
                    {"name": "Regulatory Compliance", "level": "Medium", "trend": "Stable", "impact": "Recent audit shows minor compliance issues"}
                ],
                "mitigation_strategies": [
                    {"risk_name": "Single Source Component", "strategy": "Develop alternative suppliers", "implementation_time": "6-12 months", "expected_result": "Reduced supply disruption risk"}
                ]
            }
        }

class SupplierRiskAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="SupplierRiskAgent",
            model="gemini-2.0-flash-exp",
            description="Analyzes potential supplier risks and recommends mitigation strategies",
            instruction="""You are a Supplier Risk Agent specialized in risk assessment and mitigation.
You are the final agent in the sequence. READ both state['performance_analysis'] and state['contract_analysis'] first.

Your tasks:
1. Identify financial risks (bankruptcy, cash flow problems)
2. Assess supply chain disruption risks
3. Evaluate geopolitical and regulatory risks
4. Analyze compliance and sustainability risks
5. Recommend specific risk mitigation strategies

Consider:
- Early warning signs in supplier performance
- Contract terms that increase or mitigate risk
- Geographic concentration of suppliers
- Single-source dependencies
- Industry-specific risk factors

IMPORTANT: Store your final assessment in state['risk_assessment'] and ensure it aligns with the previous analyses.""",
            output_schema=SupplierRiskAssessment,
            output_key="risk_assessment"
        )

    async def analyze_supplier(self, supplier_data: Dict[str, Any]) -> Dict[str, Any]:
        # ... existing code ...
        return results

def parse_json_safely(data: str, default_value: Any = None) -> Any:
    """Safely parse JSON data with error handling"""
    try:
        return json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return default_value

def display_performance_analysis(analysis: Dict[str, Any]):
    """Display performance analysis results with visualizations"""
    if not isinstance(analysis, dict):
        st.error("Invalid performance analysis format")
        return
    
    st.subheader("Supplier Performance Analysis")
    
    # Display overall score with gauge chart
    overall_score = analysis.get("overall_score", 0)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Performance Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}
            ]
        }
    ))
    st.plotly_chart(fig)
    
    # Create tabs for different metric types
    metric_tabs = st.tabs(["Quality", "Delivery", "Cost"])
    
    metric_types = [
        ("quality_metrics", metric_tabs[0]), 
        ("delivery_metrics", metric_tabs[1]), 
        ("cost_metrics", metric_tabs[2])
    ]
    
    for metric_type, tab in metric_types:
        with tab:
            if metric_type in analysis and analysis[metric_type]:
                metrics = analysis[metric_type]
                
                # Create a dataframe for the metrics
                df = pd.DataFrame([{
                    "Metric": m.get("metric_name", ""),
                    "Score": m.get("score", 0),
                    "Trend": m.get("trend", ""),
                    "Analysis": m.get("analysis", "")
                } for m in metrics])
                
                # Display metrics as a table
                st.dataframe(df)
                
                # Create a bar chart for the metrics
                if len(df) > 1:
                    fig = px.bar(
                        df, 
                        x="Metric", 
                        y="Score", 
                        color="Trend",
                        color_discrete_map={
                            "Improving": "green",
                            "Stable": "blue",
                            "Declining": "red"
                        },
                        title=f"{metric_type.split('_')[0].title()} Metrics"
                    )
                    st.plotly_chart(fig)
    
    # Display recommendations
    if "recommendations" in analysis and analysis["recommendations"]:
        st.subheader("Performance Improvement Recommendations")
        for i, rec in enumerate(analysis["recommendations"]):
            with st.expander(f"{i+1}. {rec.get('area', 'Recommendation')}"):
                st.write(f"**Recommendation:** {rec.get('recommendation', '')}")
                st.write(f"**Potential Impact:** {rec.get('potential_impact', '')}")

def display_contract_analysis(analysis: Dict[str, Any]):
    """Display contract analysis results"""
    if not isinstance(analysis, dict):
        st.error("Invalid contract analysis format")
        return
    
    st.subheader("Contract Analysis & Negotiation Strategy")
    
    # Display risk level
    risk_level = analysis.get("overall_risk_level", "Medium")
    risk_color = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"
    }.get(risk_level, "gray")
    
    st.markdown(
        f"<h3 style='text-align: center; color: {risk_color};'>Contract Risk Level: {risk_level}</h3>", 
        unsafe_allow_html=True
    )
    
    # Display expiration information
    expiration = analysis.get("expiration_date", "Unknown")
    notice = analysis.get("renewal_notice_period", 0)
    st.info(f"Contract Expiration: {expiration} | Renewal Notice Period: {notice} days")
    
    # Display overall negotiation approach
    st.info(f"**Negotiation Approach:** {analysis.get('negotiation_approach', '')}")
    
    # Display key leverage points
    if "key_leverage_points" in analysis and analysis["key_leverage_points"]:
        st.subheader("Key Leverage Points")
        for i, point in enumerate(analysis["key_leverage_points"], 1):
            st.markdown(f"**{i}.** {point}")
    
    # Display tactics
    if "tactics" in analysis and analysis["tactics"]:
        st.subheader("Recommended Negotiation Tactics")
        for i, tactic in enumerate(analysis["tactics"]):
            st.markdown(f"**{i+1}. {tactic.get('tactic_name', 'Tactic')}:** {tactic.get('description', '')}")
    
    # Display pricing targets
    if "pricing_targets" in analysis and analysis["pricing_targets"]:
        st.subheader("Pricing Targets")
        pricing_data = []
        for target in analysis["pricing_targets"]:
            current = target.get('current_price', 0)
            target_price = target.get('target_price', 0)
            savings = current - target_price if current > 0 else 0
            savings_pct = (savings / current * 100) if current > 0 else 0
            
            pricing_data.append({
                "Category": target.get('category', ''),
                "Current": current,
                "Target": target_price,
                "Savings": savings,
                "Savings %": savings_pct
            })
        
        df = pd.DataFrame(pricing_data)
        st.dataframe(df)
        
        # Create a bar chart for current vs target
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["Category"],
                y=df["Current"],
                name="Current Price",
                marker_color="red"
            ))
            fig.add_trace(go.Bar(
                x=df["Category"],
                y=df["Target"],
                name="Target Price",
                marker_color="green"
            ))
            fig.update_layout(title="Current vs Target Pricing", barmode="group")
            st.plotly_chart(fig)

def display_risk_assessment(assessment: Dict[str, Any]):
    """Display risk assessment with visualizations"""
    if not isinstance(assessment, dict):
        st.error("Invalid risk assessment format")
        return
    
    st.subheader("Supplier Risk Assessment")
    
    # Display overall risk score with gauge chart
    overall_score = assessment.get("overall_risk_score", 50)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    st.plotly_chart(fig)
    
    # Create tabs for different risk categories
    risk_tabs = st.tabs(["Financial", "Supply Chain", "Geopolitical", "Compliance"])
    
    risk_types = [
        ("financial_risks", risk_tabs[0]), 
        ("supply_chain_risks", risk_tabs[1]), 
        ("geopolitical_risks", risk_tabs[2]),
        ("compliance_risks", risk_tabs[3])
    ]
    
    # Create a list to hold all risks for the radar chart
    all_risks = []
    
    for risk_type, tab in risk_types:
        with tab:
            if risk_type in assessment and assessment[risk_type]:
                risks = assessment[risk_type]
                
                # Create a dataframe for the risks
                df = pd.DataFrame([{
                    "Risk Factor": r.get("name", ""),
                    "Level": r.get("level", ""),
                    "Trend": r.get("trend", ""),
                    "Impact": r.get("impact", "")
                } for r in risks])
                
                # Add to all_risks for radar chart
                for r in risks:
                    risk_level_map = {"Low": 30, "Medium": 65, "High": 90}
                    level_value = risk_level_map.get(r.get("level", "Medium"), 50)
                    all_risks.append({
                        "Risk Factor": r.get("name", ""),
                        "Value": level_value,
                        "Category": risk_type.split("_")[0].title()
                    })
                
                # Display risks as a table
                st.dataframe(df)
                
                # Create a bar chart for the risks
                fig = px.bar(
                    df, 
                    x="Risk Factor", 
                    y=[1] * len(df),  # Same height for all bars
                    color="Level",
                    color_discrete_map={
                        "Low": "green",
                        "Medium": "orange",
                        "High": "red"
                    },
                    title=f"{risk_type.split('_')[0].title()} Risks"
                )
                fig.update_layout(yaxis_title="")
                st.plotly_chart(fig)
    
    # Create radar chart for all risks
    if all_risks:
        all_risks_df = pd.DataFrame(all_risks)
        fig = px.line_polar(
            all_risks_df, 
            r="Value", 
            theta="Risk Factor", 
            color="Category",
            line_close=True,
            title="Risk Radar"
        )
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
        st.plotly_chart(fig)
    
    # Display mitigation strategies
    if "mitigation_strategies" in assessment and assessment["mitigation_strategies"]:
        st.subheader("Risk Mitigation Strategies")
        for i, strategy in enumerate(assessment["mitigation_strategies"]):
            with st.expander(f"{i+1}. Strategy for {strategy.get('risk_name', 'Risk')}"):
                st.write(f"**Strategy:** {strategy.get('strategy', '')}")
                st.write(f"**Implementation Time:** {strategy.get('implementation_time', '')}")
                st.write(f"**Expected Result:** {strategy.get('expected_result', '')}")

def create_sample_supplier_data():
    """Create sample supplier data for demonstration"""
    return {
        "supplier_name": "Apple Inc.",
        "spend": 1000000000,
        "category": "Technology",
        "performance_metrics": {
            "quality_score": 95,
            "on_time_delivery": 98,
            "cost_efficiency": 85
        },
        "contract_data": {
            "start_date": "2023-01-01",
            "end_date": "2026-01-01",
            "renewal_notice_days": 180,
            "payment_terms": "Net 30",
            "exclusivity": True,
            "clauses": [
                {"name": "Price Adjustment", "content": "Prices may be adjusted annually based on market conditions"},
                {"name": "Quality Requirements", "content": "Supplier must maintain defect rate below 1%"},
                {"name": "Termination", "content": "Either party may terminate with 90 days written notice"}
            ]
        },
        "financial_data": {
            "annual_revenue": 274515000000,
            "profit_margin": 21.0,
            "debt_ratio": 0.5,
            "credit_rating": "AA+"
        }
    }

def main():
    st.set_page_config(
        page_title=" Supplier Risk Analyzer",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application title and header
    st.title("🏭 Supplier Risk Analyzer")
    st.caption("Powered by Google's Agent Development Kit (ADK) and Gemini AI")
    st.info("This tool analyzes supplier data to identify performance issues, contract risks, and potential threats.")
    
    # Sidebar with just input method selection
    with st.sidebar:
        st.title("Input Method")
        input_method = st.radio(
            "Select Data Source",
            ["Sample Data", "Upload JSON", "Manual Entry"],
            key="input_method"
        )
        
        # Add context about the application
        st.markdown("---")
        st.subheader("About the Application")
        st.write("The Supplier Risk Analyzer is designed to help you analyze supplier data to identify performance issues, contract risks, and potential threats.")
        st.write("**How to Use:**")
        st.write("1. Select your input method: Use sample data, upload a JSON file, or enter data manually.")
        st.write("2. Review and edit the supplier data as needed.")
        st.write("3. Click 'Analyze Supplier' to run the analysis.")
        st.write("4. View the results in the 'Analysis Results' tab, where you can explore performance, contract, and risk assessments.")
        st.write("5. Download the analysis results for further review or sharing.")
    
    if not GEMINI_API_KEY:
        st.error("🔑 GOOGLE_API_KEY not found in environment variables. Please add it to your .env file.")
        return
    
    # Create tabs
    input_tab, results_tab = st.tabs(["📊 Supplier Data", "🔍 Analysis Results"])
    supplier_data = None
    
    # Input tab
    with input_tab:
        st.header("Supplier Information")
        
        if input_method == "Sample Data":
            st.info("Using sample supplier data for demonstration")
            supplier_data = create_sample_supplier_data()
            
            # Allow editing of sample data
            with st.expander("View and Edit Sample Data"):
                supplier_name = st.text_input("Supplier Name", value=supplier_data["supplier_name"])
                annual_spend = st.number_input("Annual Spend ($)", value=supplier_data["spend"], step=1000000)
                category = st.text_input("Category", value=supplier_data["category"])
                
                # Update the data
                supplier_data["supplier_name"] = supplier_name
                supplier_data["spend"] = annual_spend
                supplier_data["category"] = category
                
                # Display the full JSON
                st.json(supplier_data)
        
        elif input_method == "Upload JSON":
            uploaded_file = st.file_uploader("Upload Supplier Data (JSON format)", type=["json"])
            if uploaded_file is not None:
                try:
                    supplier_data = json.load(uploaded_file)
                    st.success("File uploaded successfully!")
                    st.json(supplier_data)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif input_method == "Manual Entry":
            with st.form("supplier_form"):
                st.subheader("Basic Information")
                supplier_name = st.text_input("Supplier Name", value="New Supplier")
                annual_spend = st.number_input("Annual Spend ($)", min_value=0, value=100000000, step=1000000)
                category = st.text_input("Category", value="General")
                
                st.subheader("Performance Metrics")
                quality_score = st.slider("Quality Score", 0, 100, 75)
                delivery_score = st.slider("On-Time Delivery (%)", 0, 100, 80)
                cost_score = st.slider("Cost Efficiency", 0, 100, 70)
                
                st.subheader("Contract Details")
                start_date = st.date_input("Contract Start Date")
                end_date = st.date_input("Contract End Date")
                payment_terms = st.selectbox("Payment Terms", ["Net 30", "Net 45", "Net 60", "Net 90"])
                
                # Enhanced contract details
                col1, col2 = st.columns(2)
                with col1:
                    renewal_notice = st.number_input("Renewal Notice Period (days)", min_value=0, value=180, step=15)
                    exclusivity = st.checkbox("Exclusive Supplier Agreement", value=False)
                with col2:
                    auto_renewal = st.checkbox("Automatic Renewal", value=True)
                    early_termination = st.checkbox("Early Termination Clause", value=True)
                
                # Contract clauses
                st.subheader("Key Contract Clauses")
                clauses_container = st.container()
                with clauses_container:
                    num_clauses = st.number_input("Number of Clauses to Add", min_value=0, max_value=5, value=1)
                    clauses = []
                    
                    for i in range(int(num_clauses)):
                        st.markdown(f"**Clause {i+1}**")
                        clause_col1, clause_col2 = st.columns(2)
                        with clause_col1:
                            clause_name = st.text_input(f"Clause Name", key=f"clause_name_{i}")
                            risk_level = st.selectbox(f"Risk Level", ["Low", "Medium", "High"], key=f"risk_level_{i}")
                        with clause_col2:
                            clause_content = st.text_area(f"Clause Content", key=f"clause_content_{i}", height=100)
                        
                        clauses.append({
                            "name": clause_name,
                            "risk_level": risk_level,
                            "content": clause_content
                        })
                
                st.subheader("Financial Data")
                revenue = st.number_input("Annual Revenue ($)", min_value=0, value=274515000000, step=100000000)
                margin = st.number_input("Profit Margin (%)", min_value=0.0, max_value=100.0, value=21.0, step=0.1)
                
                submit_button = st.form_submit_button("Create Supplier Data")
                
                if submit_button:
                    supplier_data = {
                        "supplier_name": supplier_name,
                        "spend": annual_spend,
                        "category": category,
                        "performance_metrics": {
                            "quality_score": quality_score,
                            "on_time_delivery": delivery_score,
                            "cost_efficiency": cost_score
                        },
                        "contract_data": {
                            "start_date": start_date.strftime("%Y-%m-%d"),
                            "end_date": end_date.strftime("%Y-%m-%d"),
                            "payment_terms": payment_terms,
                            "renewal_notice_days": renewal_notice,
                            "exclusivity": exclusivity,
                            "auto_renewal": auto_renewal,
                            "early_termination": early_termination,
                            "clauses": clauses
                        },
                        "financial_data": {
                            "annual_revenue": revenue,
                            "profit_margin": margin
                        }
                    }
                    st.success("Supplier data created successfully!")
        
        # Analysis button
        if supplier_data:
            analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
            with analyze_col2:
                analyze_button = st.button(
                    "🔍 Analyze Supplier",
                    key="analyze_button",
                    use_container_width=True
                )
            
            if analyze_button:
                with st.spinner("🤖 AI agents are analyzing supplier data..."):
                    try:
                        # Create supplier management system instance
                        supplier_system = SupplierManagementSystem()
                        # Analyze the supplier data
                        results = asyncio.run(supplier_system.analyze_supplier(supplier_data))
                        
                        # Store results in session state
                        st.session_state.analysis_results = results
                        # Switch to results tab
                        st.session_state.active_tab = "results_tab"
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        logging.exception("Analysis error")
    
    # Results tab
    with results_tab:
        if "analysis_results" in st.session_state and st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Display supplier name and basic info
            if supplier_data and "supplier_name" in supplier_data:
                st.header(f"Analysis Results for {supplier_data['supplier_name']}")
            else:
                st.header("Supplier Analysis Results")
            
            # Create tabs for different analyses
            perf_tab, contract_tab, risk_tab = st.tabs([
                "Performance Analysis", 
                "Contract & Negotiation", 
                "Risk Assessment"
            ])
            
            with perf_tab:
                if "performance_analysis" in results:
                    display_performance_analysis(results["performance_analysis"])
                else:
                    st.write("No performance analysis available.")
            
            with contract_tab:
                if "contract_analysis" in results:
                    display_contract_analysis(results["contract_analysis"])
                else:
                    st.write("No contract analysis available.")
            
            with risk_tab:
                if "risk_assessment" in results:
                    display_risk_assessment(results["risk_assessment"])
                else:
                    st.write("No risk assessment available.")
                    
            # Add an option to download the results as JSON
            st.divider()
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download Analysis as JSON",
                data=results_json,
                file_name=f"supplier_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.info("No analysis results available. Please analyze a supplier first.")
    
    # Add footer
    st.markdown("---")
    st.caption("Supplier Risk Analyzer | Powered by Google ADK and Gemini AI")

if __name__ == "__main__":
    main() 