from pydantic import BaseModel, Field
from typing import List, Optional

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