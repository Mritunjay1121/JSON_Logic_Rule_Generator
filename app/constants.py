SAMPLE_STORE_KEYS= [
    {"value": "business.address.pincode", "label": "Business Pincode", "group": "business"},
    {"value": "business.address.state", "label": "Business State", "group": "business"},
    {"value": "business.vintage_in_years", "label": "Business Vintage In Years", "group": "business"},
    {"value": "business.commercial_cibil_score", "label": "Commercial Cibil Score", "group": "business"},
    {"value": "primary_applicant.age", "label": "Primary Applicant Age", "group": "primary_applicant"},
    {"value": "primary_applicant.monthly_income", "label": "Primary Applicant Monthly Income", "group": "primary_applicant"},
    {"value": "primary_applicant.tags", "label": "Primary Applicant Tags", "group": "primary_applicant"},
    {"value": "bureau.score", "label": "Bureau Score", "group": "bureau"},
    {"value": "bureau.is_ntc", "label": "Is New to Credit?", "group": "bureau"},
    {"value": "bureau.overdue_amount", "label": "Overdue Amount", "group": "bureau"},
    {"value": "bureau.dpd", "label": "DPD", "group": "bureau"},
    {"value": "bureau.active_accounts", "label": "Active Accounts", "group": "bureau"},
    {"value": "bureau.enquiries", "label": "Enquiries", "group": "bureau"},
    {"value": "bureau.suit_filed", "label": "Suit Filed", "group": "bureau"},
    {"value": "bureau.wilful_default", "label": "Wilful Default", "group": "bureau"},
    {"value": "banking.abb", "label": "ABB", "group": "banking"},
    {"value": "banking.avg_monthly_turnover", "label": "Avg Monthly Turnover", "group": "banking"},
    {"value": "banking.total_credits", "label": "Total Credits", "group": "banking"},
    {"value": "banking.total_debits", "label": "Total Debits", "group": "banking"},
    {"value": "banking.inward_bounces", "label": "Inward Bounces", "group": "banking"},
    {"value": "banking.outward_bounces", "label": "Outward Bounces", "group": "banking"},
    {"value": "gst.registration_age_months", "label": "Registration Age Months", "group": "gst"},
    {"value": "gst.place_of_supply_count", "label": "Place Of Supply Count", "group": "gst"},
    {"value": "gst.is_gstin", "label": "Is GSTIN", "group": "gst"},
    {"value": "gst.filing_amount", "label": "Filing Amount", "group": "gst"},
    {"value": "gst.missed_returns", "label": "Missed Returns", "group": "gst"},
    {"value": "gst.monthly_turnover_avg", "label": "Monthly Turnover Avg", "group": "gst"},
    {"value": "gst.turnover", "label": "Turnover", "group": "gst"},
    {"value": "gst.turnover_growth_rate", "label": "Turnover Growth Rate", "group": "gst"},
    {"value": "gst.output_tax_liability", "label": "Output Tax Liability", "group": "gst"},
    {"value": "gst.tax_paid_cash_vs_credit_ratio", "label": "Tax Paid Cash Vs Credit Ratio", "group": "gst"},
    {"value": "gst.high_risk_suppliers_count", "label": "High Risk Suppliers Count", "group": "gst"},
    {"value": "gst.supplier_concentration_ratio", "label": "Supplier Concentration Ratio", "group": "gst"},
    {"value": "gst.customer_concentration_ratio", "label": "Customer Concentration Ratio", "group": "gst"},
    {"value": "itr.years_filed", "label": "Years Filed", "group": "itr"},
    {"value": "foir", "label": "FOIR", "group": "metrics"},
    {"value": "debt_to_income", "label": "Debt To Income", "group": "metrics"},
]

# Policy documents for RAG
POLICIES = [
    "Minimum bureau score must be 600 for loan approval. Scores below 600 indicate high credit risk.",
    "Business vintage should be at least 2 years for standard loans. New businesses needs additional scrutiny.",
    "Applicants with wilful default or suit filed status are automatically rejected regardless of other parameters.",
    "High overdue amount greater than 50000 rupees flags application high risk and requires manual review.",
    "Primary applicant age must be between 21 and 65 years. Outside this range applications are not eligible.",
    "DPD (Days Past Due) greater than 90 days indicates serious payment default and leads to rejection.",
    "Monthly income below 25000 for primary applicant is insufficient for loan approval in most cases.",
    "New to Credit (NTC) applicants require bureau score of at least 650 instead of standard 600.",
    "GST registration age should be minimum 12 months for business loan eligibility verification.",
    "Banking average monthly turnover must exceed 100000 rupees for commercial lending approval.",
]

MOCK_STORE_SAMPLES = [

    {
        "bureau.score": 750,
        "business.vintage_in_years": 5,
        "primary_applicant.age": 35,
        "primary_applicant.monthly_income": 75000,
        "bureau.wilful_default": False,
        "bureau.suit_filed": False,
        "bureau.overdue_amount": 0,
        "bureau.dpd": 0,
        "primary_applicant.tags": ["regular", "salaried"],
    },
    {
        "bureau.score": 550,
        "business.vintage_in_years": 3,
        "primary_applicant.age": 40,
        "bureau.wilful_default": False,
        "bureau.overdue_amount": 10000,
    },
    # Sample 3: Wilful default - should fail
    {
        "bureau.score": 720,
        "business.vintage_in_years": 4,
        "primary_applicant.age": 45,
        "bureau.wilful_default": True,
        "bureau.overdue_amount": 5000,
    },
    # Sample 4: High overdue amount
    {
        "bureau.score": 680,
        "business.vintage_in_years": 3,
        "primary_applicant.age": 38,
        "bureau.wilful_default": False,
        "bureau.overdue_amount": 75000,
        "bureau.dpd": 120,
    },
    # Sample 5: Veteran tag with good income
    {
        "bureau.score": 710,
        "primary_applicant.age": 42,
        "primary_applicant.monthly_income": 150000,
        "primary_applicant.tags": ["veteran", "business_owner"],
        "business.vintage_in_years": 6,
    },
    # Sample 6: Edge case - minimum acceptable values
    {
        "bureau.score": 600,
        "business.vintage_in_years": 2,
        "primary_applicant.age": 25,
        "primary_applicant.monthly_income": 50000,
        "bureau.wilful_default": False,
        "bureau.overdue_amount": 0,
    },
    # Sample 7: NTC applicant
    {
        "bureau.score": 655,
        "bureau.is_ntc": True,
        "primary_applicant.age": 28,
        "business.vintage_in_years": 1.5,
    },
    # Sample 8: High DPD
    {
        "bureau.score": 640,
        "bureau.dpd": 95,
        "business.vintage_in_years": 4,
        "primary_applicant.age": 50,
    },
]

def get_key_by_value(value) :
    """Helper to find key object by value string"""
    for key in SAMPLE_STORE_KEYS:
        if key["value"] == value:
            return key
    return None

def build_key_search_text(key):
    """Build searchable text for a key (used in embeddings)"""
    return f"{key['label']} {key['value']} {key['group']}"
