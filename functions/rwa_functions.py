def assign_risk_weight(ltv_ratio):
    if ltv_ratio < 0.5:
        return 0.2
    elif 0.5 <= ltv_ratio < 0.6:
        return 0.25
    elif 0.6 <= ltv_ratio < 0.7:
        return 0.3
    elif 0.7 <= ltv_ratio < 0.8:
        return 0.3
    elif 0.8 <= ltv_ratio < 0.9:
        return 0.4
    elif 0.9 <= ltv_ratio <= 1.0:
        return 0.5
    elif ltv_ratio > 1.0:
        return 0.7
    else:
        return None
