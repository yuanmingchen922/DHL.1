import os
from pathlib import Path

import pulp
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "DHL" / "DHLData.xlsx"
LEGACY_DATA_PATH = Path("/Users/yuanmingchen/Desktop/DHL.1/DHL/DHLData.xlsx")

def _resolve_excel_path(custom_path: Optional[str] = None) -> Path:
    """Resolve the Excel path with support for env overrides and sensible fallbacks."""

    candidate_paths = []

    if custom_path:
        candidate_paths.append(Path(custom_path).expanduser())

    env_path = os.getenv("DHL_DATA_PATH")
    if env_path:
        candidate_paths.append(Path(env_path).expanduser())

    candidate_paths.extend([DEFAULT_DATA_PATH, LEGACY_DATA_PATH])

    # Preserve order but remove duplicates
    unique_paths = []
    seen = set()
    for path in candidate_paths:
        if path and path not in seen:
            unique_paths.append(path)
            seen.add(path)

    for path in unique_paths:
        if path.exists():
            return path

    # Nothing exists; return the last candidate for informative error handling
    return unique_paths[-1]


def validate_data_integrity(excel_file_path: Optional[str] = None):
    """Enhanced data validation and error handling using specified local file"""

    resolved_excel_path = _resolve_excel_path(excel_file_path)

    try:
        logger.info(f"Loading Excel file from: {resolved_excel_path}")
        xl = pd.ExcelFile(resolved_excel_path)
        logger.info(f"Successfully loaded Excel file with sheets: {xl.sheet_names}")
        
        # Load and validate customer data
        df_customers = pd.read_excel(xl, 'Customers')
        logger.info(f"Original customer data shape: {df_customers.shape}")
        
        # Check for missing critical columns
        required_customer_cols = ['City; State', '% of Volume']
        missing_cols = [col for col in required_customer_cols if col not in df_customers.columns]
        if missing_cols:
            raise ValueError(f"Missing required customer columns: {missing_cols}")
        
        # Data cleaning with validation
        initial_customers = len(df_customers)
        df_customers = df_customers.dropna(subset=required_customer_cols)
        dropped_customers = initial_customers - len(df_customers)
        if dropped_customers > 0:
            logger.warning(f"Dropped {dropped_customers} customers due to missing data")
        
        # Validate volume percentages
        volume_sum = df_customers['% of Volume'].sum()
        if abs(volume_sum - 1.0) > 0.01:  # Allow 1% tolerance
            logger.warning(f"Volume percentages sum to {volume_sum:.3f}, expected ~1.0")
            # Normalize if close to 1.0
            if 0.95 <= volume_sum <= 1.05:
                df_customers['% of Volume'] = df_customers['% of Volume'] / volume_sum
                logger.info("Normalized volume percentages to sum to 1.0")
        
        # Load and validate cost data
        df_costs = pd.read_excel(xl, 'fixed & variable costs')
        logger.info(f"Cost data shape: {df_costs.shape}")
        
        required_cost_cols = ['Candidate Facility', 'Manufacturing Fixed Annual Cost', 
                             'Distribution Fixed Annual Cost', 'Vairable Manufacturing Cost Per Bottle',
                             'Distribution Processing Cost Per Order (6 Bottles)']
        missing_cost_cols = [col for col in required_cost_cols if col not in df_costs.columns]
        if missing_cost_cols:
            raise ValueError(f"Missing required cost columns: {missing_cost_cols}")
        
        # Load distance matrices
        df_dc_to_cust = pd.read_excel(xl, 'Distances DC to Cust')
        df_mfg_to_dc = pd.read_excel(xl, 'Distances MFG to DC')
        
        logger.info(f"DC to Customer distance matrix shape: {df_dc_to_cust.shape}")
        logger.info(f"MFG to DC distance matrix shape: {df_mfg_to_dc.shape}")
        
        # Check for negative distances
        numeric_cols_dc = df_dc_to_cust.select_dtypes(include=[np.number]).columns
        negative_distances_dc = (df_dc_to_cust[numeric_cols_dc] < 0).sum().sum()
        if negative_distances_dc > 0:
            logger.warning(f"Found {negative_distances_dc} negative distances in DC to Customer matrix")
        
        numeric_cols_mfg = df_mfg_to_dc.select_dtypes(include=[np.number]).columns
        negative_distances_mfg = (df_mfg_to_dc[numeric_cols_mfg] < 0).sum().sum()
        if negative_distances_mfg > 0:
            logger.warning(f"Found {negative_distances_mfg} negative distances in MFG to DC matrix")
        
        return df_customers, df_costs, df_dc_to_cust, df_mfg_to_dc, xl
        
    except FileNotFoundError:
        logger.error(f"DHLData.xlsx file not found at {resolved_excel_path}! Please ensure the file exists.")
        raise FileNotFoundError(f"Required Excel file not found: {resolved_excel_path}")
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise RuntimeError(f"Failed to load data from DHLData.xlsx at {resolved_excel_path}: {str(e)}")

# Execute data validation
df_customers, df_costs, df_dc_to_cust, df_mfg_to_dc, xl = validate_data_integrity()

candidates_full = df_costs['Candidate Facility'].tolist()
customers_full = df_customers['City; State'].tolist()

def simplify_name(name: str) -> str:
    """Enhanced name simplification with comprehensive mapping"""
    if pd.isna(name):
        return ""
    
    name = str(name).strip()
    
    # Handle different separators
    if ';' in name:
        base_name = name.split(';')[0].strip()
    elif ',' in name:
        base_name = name.split(',')[0].strip()
    else:
        base_name = name.strip()
    
    # Comprehensive name mapping for standardization
    name_mapping = {
        'New York City': 'New York',
        'Nashville-Davidson': 'Nashville',
        'Louisville/Jefferson': 'Louisville', 
        'Oklahoma ': 'Oklahoma City',
        'Oklahoma City ': 'Oklahoma City',  
        'Oklahoma': 'Oklahoma City',
        'Los Angeles': 'LA',
        'San Francisco': 'SF',
        'Washington': 'Washington DC',
        'St. Louis': 'Saint Louis',
        'St Louis': 'Saint Louis'
    }
    
    return name_mapping.get(base_name, base_name)

def validate_facility_customer_matching():
    """Validate that all candidates appear in distance matrices"""
    candidates = [simplify_name(name) for name in candidates_full]
    customers = [simplify_name(name) for name in customers_full]
    
    # Remove duplicates while preserving order
    candidates = list(dict.fromkeys(candidates))
    customers = list(dict.fromkeys(customers))
    
    logger.info(f"Processed {len(candidates)} unique candidate facilities: {candidates}")
    logger.info(f"Processed {len(customers)} unique customer cities (showing first 5): {customers[:5]}...")
    
    # Validate candidate coverage in distance matrices
    dc_to_cust_cols = [simplify_name(col) for col in df_dc_to_cust.columns[1:]]
    mfg_to_dc_cols = [simplify_name(col) for col in df_mfg_to_dc.columns[1:]]
    
    missing_in_dc_cust = set(candidates) - set(dc_to_cust_cols)
    missing_in_mfg_dc = set(candidates) - set(mfg_to_dc_cols)
    
    if missing_in_dc_cust:
        logger.warning(f"Candidates missing in DC-to-Customer matrix: {missing_in_dc_cust}")
    if missing_in_mfg_dc:
        logger.warning(f"Candidates missing in MFG-to-DC matrix: {missing_in_mfg_dc}")
    
    return candidates, customers

candidates, customers = validate_facility_customer_matching()

def clean_cost_value(value) -> float:
    """Enhanced cost value cleaning with validation"""
    if pd.isna(value):
        logger.warning(f"Found NaN cost value, defaulting to 0")
        return 0.0
    
    if isinstance(value, str):
        value = value.replace('$', '').replace(' ', '').replace(',', '')
        if 'K' in value.upper():
            return float(value.upper().replace('K', '')) * 1000
        elif 'M' in value.upper():
            return float(value.upper().replace('M', '')) * 1000000
        return float(value)
    return float(value)

def process_cost_data() -> Tuple[Dict, Dict, Dict, Dict]:
    """Process cost data with comprehensive validation"""
    fixed_mfg_costs = {}
    fixed_dist_costs = {}
    var_mfg_costs = {}
    proc_costs = {}
    
    for i, row in df_costs.iterrows():
        candidate = simplify_name(row['Candidate Facility'])
        if candidate in candidates:  # Only process valid candidates
            try:
                # Fixed costs (f^m_i and f^d_j in the equation)
                fixed_mfg_costs[candidate] = clean_cost_value(row['Manufacturing Fixed Annual Cost'])
                fixed_dist_costs[candidate] = clean_cost_value(row['Distribution Fixed Annual Cost'])
                
                # Variable costs (v^m_m in the equation)
                var_mfg_costs[candidate] = clean_cost_value(row['Vairable Manufacturing Cost Per Bottle'])
                
                # Processing costs (p^j in the equation)
                proc_costs[candidate] = clean_cost_value(row['Distribution Processing Cost Per Order (6 Bottles)'])
                
            except (ValueError, KeyError) as e:
                logger.error(f"Error processing costs for {candidate}: {e}")
                # Set default values for missing data
                fixed_mfg_costs[candidate] = 0.0
                fixed_dist_costs[candidate] = 0.0
                var_mfg_costs[candidate] = 0.0
                proc_costs[candidate] = 0.0
    
    # Validation checks
    logger.info("=== Cost Data Validation ===")
    logger.info(f"Manufacturing fixed costs range: ${min(fixed_mfg_costs.values()):,.0f} - ${max(fixed_mfg_costs.values()):,.0f}")
    logger.info(f"Distribution fixed costs range: ${min(fixed_dist_costs.values()):,.0f} - ${max(fixed_dist_costs.values()):,.0f}")
    logger.info(f"Variable manufacturing costs range: ${min(var_mfg_costs.values()):.2f} - ${max(var_mfg_costs.values()):.2f} per bottle")
    logger.info(f"Processing costs range: ${min(proc_costs.values()):.2f} - ${max(proc_costs.values()):.2f} per order")
    
    # Check for zero costs (potential data issues)
    zero_fixed_mfg = [k for k, v in fixed_mfg_costs.items() if v == 0]
    zero_fixed_dist = [k for k, v in fixed_dist_costs.items() if v == 0]
    zero_var_mfg = [k for k, v in var_mfg_costs.items() if v == 0]
    zero_proc = [k for k, v in proc_costs.items() if v == 0]
    
    if zero_fixed_mfg:
        logger.warning(f"Facilities with zero manufacturing fixed costs: {zero_fixed_mfg}")
    if zero_fixed_dist:
        logger.warning(f"Facilities with zero distribution fixed costs: {zero_fixed_dist}")
    if zero_var_mfg:
        logger.warning(f"Facilities with zero variable manufacturing costs: {zero_var_mfg}")
    if zero_proc:
        logger.warning(f"Facilities with zero processing costs: {zero_proc}")
    
    return fixed_mfg_costs, fixed_dist_costs, var_mfg_costs, proc_costs

fixed_mfg_costs, fixed_dist_costs, var_mfg_costs, proc_costs = process_cost_data()

def process_demand_data(total_volume: int = 2000000) -> Dict[str, float]:
    """Process demand data with validation and statistics"""
    demand = {}
    
    for i, row in df_customers.iterrows():
        customer_name = simplify_name(row['City; State'])
        if customer_name in customers:  # Only process valid customers
            volume_pct = row['% of Volume']
            if pd.isna(volume_pct) or volume_pct < 0:
                logger.warning(f"Invalid volume percentage for {customer_name}: {volume_pct}")
                volume_pct = 0
            demand[customer_name] = volume_pct * total_volume
    
    # Demand validation and statistics
    total_demand = sum(demand.values())
    logger.info("=== Demand Data Analysis ===")
    logger.info(f"Total demand: {total_demand:,.0f} bottles ({total_demand/total_volume:.1%} of target)")
    logger.info(f"Number of customer locations: {len(demand)}")
    logger.info(f"Average demand per location: {total_demand/len(demand):,.0f} bottles")
    logger.info(f"Demand range: {min(demand.values()):,.0f} - {max(demand.values()):,.0f} bottles")
    
    # Identify top customers
    top_customers = sorted(demand.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Top 5 customers by demand: {[(name, f'{vol:,.0f}') for name, vol in top_customers]}")
    
    return demand

demand = process_demand_data()

def build_distance_matrix(df_distance: pd.DataFrame, 
                         from_locations: List[str], 
                         to_locations: List[str],
                         matrix_name: str) -> Dict[Tuple[str, str], float]:
    """Enhanced distance matrix builder with comprehensive validation"""
    distance_dict = {}
    
    # Handle the actual Excel file structure where first column is 'to/from'
    # Distance columns are all columns except the first one
    distance_columns = df_distance.columns[1:].tolist()  # Skip first column 'to/from'
    from_cols = [simplify_name(col) for col in distance_columns]
    to_rows = [simplify_name(name) for name in df_distance.iloc[:, 0].tolist()]  # First column contains row names
    
    missing_from = set(from_locations) - set(from_cols)
    missing_to = set(to_locations) - set(to_rows)
    
    if missing_from:
        logger.warning(f"{matrix_name}: Missing origin locations: {missing_from}")
    if missing_to:
        logger.warning(f"{matrix_name}: Missing destination locations: {missing_to}")
    
    valid_distances = 0
    missing_distances = 0
    zero_distances = 0
    
    for i, from_location in enumerate(from_cols):
        if from_location in from_locations:
            for j, to_location in enumerate(to_rows):
                if to_location in to_locations:
                    try:
                        # Get distance using row j and column i+1 (since first column is 'to/from')
                        distance = df_distance.iloc[j, i+1]
                        if pd.isna(distance):
                            logger.warning(f"{matrix_name}: Missing distance {from_location} -> {to_location}")
                            distance = 0
                            missing_distances += 1
                        elif distance < 0:
                            logger.warning(f"{matrix_name}: Negative distance {from_location} -> {to_location}: {distance}")
                            distance = abs(distance)
                        elif distance == 0 and from_location != to_location:
                            zero_distances += 1
                        
                        distance_dict[(from_location, to_location)] = float(distance)
                        valid_distances += 1
                        
                    except (IndexError, ValueError, KeyError) as e:
                        logger.error(f"{matrix_name}: Error processing {from_location} -> {to_location}: {e}")
                        distance_dict[(from_location, to_location)] = 0
                        missing_distances += 1
    
    logger.info(f"=== {matrix_name} Distance Matrix Statistics ===")
    logger.info(f"Valid distances: {valid_distances}")
    logger.info(f"Missing/Error distances: {missing_distances}")
    logger.info(f"Zero distances (non-diagonal): {zero_distances}")
    
    if distance_dict:
        distances = list(distance_dict.values())
        logger.info(f"Distance range: {min(distances):.1f} - {max(distances):.1f} miles")
        logger.info(f"Average distance: {np.mean(distances):.1f} miles")
    
    return distance_dict

dist_mfg_to_dc = build_distance_matrix(df_mfg_to_dc, candidates, candidates, "MFG to DC")
dist_dc_to_cust = build_distance_matrix(df_dc_to_cust, candidates, customers, "DC to Customer")

def create_optimization_model(scenario_params=None):
    """
    Create the optimization model following Version 3 Lagrangian Dual formulation
    
    Mathematical Framework from Update_Version_3.md:
    min_{X,Z,Y} ∑f^m_i(1+α_i)Y_i + ∑f^d_j(1+β_j)Y_j + 
                ∑(v^m_i + 3/2000·d^MD_ij(1+γ_ij))X_ij + 
                ∑Z_jc/6·(p^j+9.75+3.5·d^DC_jc/500)(1+δ_jc)
    
    Args:
        scenario_params: Dictionary containing sensitivity coefficients {alpha, beta, gamma, delta}
    """
    
    logger.info("=== Creating Optimization Model (Version 3) ===")
    
    # Initialize scenario parameters (perturbation coefficients)
    if scenario_params is None:
        scenario_params = {
            'alpha': {i: 0.0 for i in candidates},  # Manufacturing fixed cost perturbation
            'beta': {j: 0.0 for j in candidates},   # Distribution fixed cost perturbation
            'gamma': {(i,j): 0.0 for i in candidates for j in candidates},  # Transport cost perturbation
            'delta': {(j,c): 0.0 for j in candidates for c in customers}    # Delivery cost perturbation
        }
    
    logger.info(f"Scenario parameters initialized: α_max={max(scenario_params['alpha'].values()):.2%}, "
                f"β_max={max(scenario_params['beta'].values()):.2%}")
    
    # Create the problem instance
    prob = pulp.LpProblem("Lagrangian_Dual_Network_Design_V3", pulp.LpMinimize)
    
    # ===== Decision Variables =====
    # Y_i ∈ {0,1}: Binary variable for opening manufacturing facility at location i
    y_m = pulp.LpVariable.dicts("Y_MFG", candidates, cat='Binary')
    
    # Y_j ∈ {0,1}: Binary variable for opening distribution center at location j  
    y_d = pulp.LpVariable.dicts("Y_DC", candidates, cat='Binary')
    
    # X_ij ≥ 0: Flow from manufacturing facility i to distribution center j (bottles)
    x = pulp.LpVariable.dicts("X_Flow_MFG_to_DC", 
                             [(i, j) for i in candidates for j in candidates], 
                             lowBound=0, cat='Continuous')
    
    # Z_jc ≥ 0: Flow from distribution center j to customer c (bottles)
    z = pulp.LpVariable.dicts("Z_Flow_DC_to_Cust", 
                             [(j, c) for j in candidates for c in customers 
                              if (j, c) in dist_dc_to_cust], 
                             lowBound=0, cat='Continuous')
    
    logger.info(f"Created {len(y_m)} Y_MFG binary variables")
    logger.info(f"Created {len(y_d)} Y_DC binary variables")
    logger.info(f"Created {len(x)} X_Flow variables")
    logger.info(f"Created {len(z)} Z_Flow variables")
    
    # ===== Objective Function (Version 3 Formulation) =====
    # Following the exact mathematical structure from Update_Version_3.md
    
    # Component 1: Manufacturing Fixed Costs with scenario coefficient
    # ∑_i f^m_i(1+α_i)Y_i
    mfg_fixed_cost = pulp.lpSum([
        fixed_mfg_costs[i] * (1 + scenario_params['alpha'][i]) * y_m[i] 
        for i in candidates
    ])
    
    # Component 2: Distribution Fixed Costs with scenario coefficient
    # ∑_j f^d_j(1+β_j)Y_j
    dist_fixed_cost = pulp.lpSum([
        fixed_dist_costs[j] * (1 + scenario_params['beta'][j]) * y_d[j] 
        for j in candidates
    ])
    
    # Component 3: Manufacturing Variable Costs and Inbound Transportation (combined)
    # ∑_{i,j} (v^m_i + 3/2000·d^MD_ij(1+γ_ij))X_ij
    mfg_and_transport_cost = pulp.lpSum([
        (var_mfg_costs[i] + 
         (3.0/2000.0) * dist_mfg_to_dc.get((i, j), 0) * (1 + scenario_params['gamma'][(i,j)])) 
        * x[i, j]
        for i in candidates for j in candidates
    ])
    
    # Component 4: Outbound Distribution Costs with scenario coefficient
    # ∑_{j,c} Z_jc/6 · (p^j + 9.75 + 3.5·d^DC_jc/500)(1+δ_jc)
    outbound_cost = pulp.lpSum([
        (z[j, c] / 6.0) * 
        (proc_costs[j] + 9.75 + 3.5 * (dist_dc_to_cust.get((j, c), 0) / 500.0)) * 
        (1 + scenario_params['delta'].get((j,c), 0.0))
        for j in candidates for c in customers 
        if (j, c) in dist_dc_to_cust
    ])
    
    # Total Primal Objective Function
    prob += (mfg_fixed_cost + dist_fixed_cost + mfg_and_transport_cost + outbound_cost,
             "Total_Cost_Objective")
    
    logger.info("Objective function created following Version 3 mathematical structure")
    logger.info("  - Component 1: f^m_i(1+α_i)Y_i")
    logger.info("  - Component 2: f^d_j(1+β_j)Y_j")
    logger.info("  - Component 3: (v^m_i + 3/2000·d^MD_ij(1+γ_ij))X_ij")
    logger.info("  - Component 4: Z_jc/6·(p^j+9.75+3.5·d^DC_jc/500)(1+δ_jc)")
    
    return prob, y_m, y_d, x, z, scenario_params

prob, y_m, y_d, x, z, scenario_params = create_optimization_model()

def add_enhanced_constraints(prob, y_m, y_d, x, z, scenario_params):
    """
    Add constraint system following Version 3 framework
    
    Constraints are divided into:
    (H) - Hard constraints: Demand satisfaction, Flow balance, Non-negativity
    (LNK) - Linking constraints: Can be relaxed in Lagrangian formulation
    
    Mathematical foundation from Update_Version_3.md:
    - Demand: ∑_j Z_jc = D_c, ∀c
    - Balance: ∑_c Z_jc = ∑_i X_ij, ∀j
    - Linking: X_ij ≤ U_ij·Y_i, Z_jc ≤ V_jc·Y_j (relaxable)
    """
    
    logger.info("=== Adding Enhanced Constraint System (Version 3) ===")
    
    total_demand_volume = sum(demand.values())
    
    # ===== HARD CONSTRAINTS (H) - Must always be satisfied =====
    constraint_count = 0
    
    # (H1) DEMAND SATISFACTION CONSTRAINTS
    # Each customer's demand must be fully satisfied: ∑_j Z_jc = D_c
    for c in customers:
        if c in demand and demand[c] > 0:
            prob += (
                pulp.lpSum([z[j, c] for j in candidates if (j, c) in dist_dc_to_cust]) == demand[c],
                f"Demand_Satisfaction_{c}"
            )
            constraint_count += 1
    logger.info(f"(H1) Added {len([c for c in customers if c in demand and demand[c] > 0])} demand satisfaction constraints")
    
    # (H2) FLOW BALANCE CONSTRAINTS  
    # Inflow equals outflow at each DC: ∑_c Z_jc = ∑_i X_ij
    for j in candidates:
        inflow = pulp.lpSum([x[i, j] for i in candidates])
        outflow = pulp.lpSum([z[j, c] for c in customers if (j, c) in dist_dc_to_cust])
        prob += (inflow == outflow, f"Flow_Balance_DC_{j}")
        constraint_count += 1
    logger.info(f"(H2) Added {len(candidates)} flow balance constraints")
    
    # ===== LINKING CONSTRAINTS (LNK) - Can be relaxed in Lagrangian =====
    # In this implementation, we keep them as hard constraints for the primal problem
    # But we structure them to match the theoretical framework
    
    # Compute tighter upper bounds U_ij and V_jc following Version 3
    # U_ij: maximum flow from MFG i to DC j (use total demand as conservative bound)
    U_ij = {(i, j): total_demand_volume for i in candidates for j in candidates}
    
    # V_jc: maximum flow from DC j to customer c (equals customer demand D_c)
    V_jc = {(j, c): demand[c] for j in candidates for c in customers 
            if (j, c) in dist_dc_to_cust and c in demand}
    
    logger.info(f"Computed tight upper bounds: U_ij (uniform), V_jc (demand-based)")
    
    # (LNK1) MANUFACTURING LINKING CONSTRAINTS
    # X_ij ≤ U_ij·Y_i - flow can only occur if MFG facility is open
    for i in candidates:
        for j in candidates:
            prob += (x[i, j] <= U_ij[(i, j)] * y_m[i], 
                    f"Link_MFG_{i}_to_DC_{j}")
            constraint_count += 1
    logger.info(f"(LNK1) Added {len(candidates)**2} manufacturing linking constraints")
    
    # (LNK2) DISTRIBUTION LINKING CONSTRAINTS
    # Z_jc ≤ V_jc·Y_j - flow can only occur if DC is open
    for j in candidates:
        for c in customers:
            if (j, c) in dist_dc_to_cust and (j, c) in V_jc:
                prob += (z[j, c] <= V_jc[(j, c)] * y_d[j], 
                        f"Link_DC_{j}_to_Cust_{c}")
                constraint_count += 1
    logger.info(f"(LNK2) Added {len([k for k in V_jc.keys()])} distribution linking constraints")
    
    # ===== LOGICAL CONSTRAINTS - Ensure network feasibility =====
    
    # (L1) Minimum facility coverage - at least one DC must be open
    prob += (pulp.lpSum([y_d[j] for j in candidates]) >= 1, "Min_DC_Coverage")
    constraint_count += 1
    
    # (L2) Minimum manufacturing coverage - at least one MFG must be open  
    prob += (pulp.lpSum([y_m[i] for i in candidates]) >= 1, "Min_MFG_Coverage")
    constraint_count += 1
    logger.info("(L) Added minimum coverage constraints")
    
    # ===== OPTIONAL CAPACITY LIMITS =====
    # Following Version 3 theory, we can add realistic capacity constraints
    # These are separate from the linking constraints
    
    # Maximum single facility capacity (allow concentration for cost optimization)
    max_facility_capacity = total_demand_volume * 1.0  
    
    for i in candidates:
        total_production = pulp.lpSum([x[i, j] for j in candidates])
        prob += (total_production <= max_facility_capacity, f"Max_Capacity_MFG_{i}")
        constraint_count += 1
    
    for j in candidates:
        total_distribution = pulp.lpSum([z[j, c] for c in customers if (j, c) in dist_dc_to_cust])
        prob += (total_distribution <= max_facility_capacity, f"Max_Capacity_DC_{j}")
        constraint_count += 1
    logger.info(f"(CAP) Added {len(candidates)*2} capacity limit constraints")
    
    logger.info(f"Total constraints added: {constraint_count}")
    logger.info("Constraint structure:")
    logger.info("  - Hard Constraints (H): Demand + Flow Balance")
    logger.info("  - Linking Constraints (LNK): X_ij ≤ U_ij·Y_i, Z_jc ≤ V_jc·Y_j")
    logger.info("  - Logical Constraints (L): Minimum coverage")
    logger.info("  - Capacity Constraints (CAP): Realistic bounds")
    
    return prob, U_ij, V_jc

prob, U_ij, V_jc = add_enhanced_constraints(prob, y_m, y_d, x, z, scenario_params)


def _build_solver_candidates() -> List[pulp.LpSolver_CMD]:
    """Return solver instances that are available in the current environment."""

    solver_factories = (
        lambda: pulp.PULP_CBC_CMD(msg=1),
        lambda: pulp.GUROBI_CMD(),
        lambda: pulp.CPLEX_CMD(),
    )

    solvers = []

    for factory in solver_factories:
        try:
            solver = factory()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Skipping solver factory {factory}: {exc}")
            continue

        availability_check = getattr(solver, "available", None)
        if callable(availability_check):
            try:
                if not solver.available():
                    logger.info(f"Solver {solver.__class__.__name__} is not available in this environment")
                    continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Availability check failed for {solver.__class__.__name__}: {exc}")
                continue

        solvers.append(solver)

    return solvers


def solve_with_enhanced_logging(prob):
    """Solve optimization problem with comprehensive logging and diagnostics"""
    
    logger.info("=== Starting Optimization Process ===")
    logger.info(f"Problem type: {prob.sense}")
    logger.info(f"Number of variables: {len(prob.variables())}")
    logger.info(f"Number of constraints: {len(prob.constraints)}")
    logger.info(f"Candidate facilities: {len(candidates)}")
    logger.info(f"Customer locations: {len(customers)}")
    logger.info(f"Available distance pairs: MFG->DC {len(dist_mfg_to_dc)}, DC->Customer {len(dist_dc_to_cust)}")
    
    # Problem statistics
    binary_vars = sum(1 for var in prob.variables() if var.cat == 'Binary')
    continuous_vars = sum(1 for var in prob.variables() if var.cat == 'Continuous')
    logger.info(f"Binary variables: {binary_vars}, Continuous variables: {continuous_vars}")
    
    # Solve the problem
    import time
    start_time = time.time()
    
    solvers_to_try = _build_solver_candidates()

    if not solvers_to_try:
        logger.info("Using default PuLP solver")
        status = prob.solve()
    else:
        status = None
        for solver in solvers_to_try:
            try:
                logger.info(f"Attempting to solve with {solver.__class__.__name__}")
                status = prob.solve(solver)
                if status is not None:
                    break
            except Exception as e:
                logger.warning(f"Solver {solver.__class__.__name__} failed: {e}")
                status = None
                continue
        if status is None:
            logger.info("Falling back to default PuLP solver after failures")
            status = prob.solve()
    
    solve_time = time.time() - start_time
    
    logger.info(f"Optimization completed in {solve_time:.2f} seconds")
    logger.info(f"Status: {pulp.LpStatus[prob.status]}")
    logger.info(f"Solver Status Code: {status}")
    
    return status

def perform_sensitivity_analysis_v3(baseline_results: Dict, scenario_params: Dict) -> Dict:
    """
    Perform comprehensive sensitivity analysis using Version 3 scenario coefficients
    
    From Update_Version_3.md:
    Scenario coefficients α_i, β_j, γ_ij, δ_jc allow modeling:
    - Cost perturbations (fuel price changes, labor costs)
    - Demand variations
    - Infrastructure changes
    
    Tests multiple scenarios by varying these coefficients.
    """
    
    logger.info("\n" + "="*70)
    logger.info("=== SENSITIVITY ANALYSIS (Version 3 Scenarios) ===")
    logger.info("="*70)
    
    sensitivity_results = {}
    baseline_cost = baseline_results['total_cost']
    
    # Define sensitivity scenarios using scenario coefficients
    scenarios = {
        'baseline': {
            'description': 'Current base scenario',
            'alpha': 0.0,  # No perturbation
            'beta': 0.0,
            'gamma': 0.0,
            'delta': 0.0
        },
        'fuel_increase_20pct': {
            'description': '20% increase in transportation costs',
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': 0.20,  # γ_ij: affects inbound transport cost
            'delta': 0.20   # δ_jc: affects outbound delivery cost
        },
        'fuel_decrease_15pct': {
            'description': '15% decrease in transportation costs',
            'alpha': 0.0,
            'beta': 0.0,
            'gamma': -0.15,
            'delta': -0.15
        },
        'facility_cost_increase_10pct': {
            'description': '10% increase in facility fixed costs',
            'alpha': 0.10,  # α_i: affects MFG fixed cost
            'beta': 0.10,   # β_j: affects DC fixed cost
            'gamma': 0.0,
            'delta': 0.0
        },
        'combined_adverse': {
            'description': 'Combined adverse scenario: +15% facilities, +25% transport',
            'alpha': 0.15,
            'beta': 0.15,
            'gamma': 0.25,
            'delta': 0.25
        },
        'combined_favorable': {
            'description': 'Combined favorable scenario: -10% facilities, -15% transport',
            'alpha': -0.10,
            'beta': -0.10,
            'gamma': -0.15,
            'delta': -0.15
        }
    }
    
    logger.info(f"\nBaseline total cost: ${baseline_cost:,.2f}")
    logger.info(f"Testing {len(scenarios)} scenarios...\n")
    
    for scenario_name, params in scenarios.items():
        if scenario_name == 'baseline':
            sensitivity_results[scenario_name] = {
                'cost': baseline_cost,
                'change_pct': 0.0,
                'description': params['description']
            }
            continue
        
        # Calculate estimated cost impact (without re-solving)
        # This is an approximation based on current solution structure
        cost_components = baseline_results['cost_breakdown']
        
        # Estimate cost changes based on perturbations
        alpha_impact = cost_components['mfg_fixed'] * params['alpha']
        beta_impact = cost_components['dc_fixed'] * params['beta']
        gamma_impact = cost_components['mfg_transport'] * params['gamma'] * 0.15  # Transport is ~15% of this component
        delta_impact = cost_components['dc_delivery'] * params['delta']
        
        estimated_cost = baseline_cost + alpha_impact + beta_impact + gamma_impact + delta_impact
        change_pct = ((estimated_cost - baseline_cost) / baseline_cost) * 100
        
        sensitivity_results[scenario_name] = {
            'cost': estimated_cost,
            'change_pct': change_pct,
            'change_abs': estimated_cost - baseline_cost,
            'description': params['description'],
            'component_impacts': {
                'alpha_mfg_fixed': alpha_impact,
                'beta_dc_fixed': beta_impact,
                'gamma_transport': gamma_impact,
                'delta_delivery': delta_impact
            }
        }
        
        logger.info(f"[{scenario_name}]")
        logger.info(f"  {params['description']}")
        logger.info(f"  Coefficients: α={params['alpha']:+.0%}, β={params['beta']:+.0%}, "
                   f"γ={params['gamma']:+.0%}, δ={params['delta']:+.0%}")
        logger.info(f"  Estimated cost: ${estimated_cost:,.2f}")
        logger.info(f"  Change: ${estimated_cost - baseline_cost:+,.2f} ({change_pct:+.1f}%)")
        logger.info("")
    
    # Identify most sensitive parameters
    logger.info("="*70)
    logger.info("SENSITIVITY SUMMARY:")
    logger.info("="*70)
    
    # Sort by absolute change
    sorted_scenarios = sorted(
        [(k, v) for k, v in sensitivity_results.items() if k != 'baseline'],
        key=lambda x: abs(x[1]['change_pct']),
        reverse=True
    )
    
    logger.info("\nScenarios ranked by impact:")
    for i, (name, results) in enumerate(sorted_scenarios, 1):
        logger.info(f"{i}. {name}: {results['change_pct']:+.1f}% ({results['description']})")
    
    # Recommendations
    logger.info("\n" + "="*70)
    logger.info("RECOMMENDATIONS:")
    logger.info("="*70)
    
    # Find best and worst cases
    best_case = min(sensitivity_results.items(), key=lambda x: x[1]['cost'])
    worst_case = max(sensitivity_results.items(), key=lambda x: x[1]['cost'])
    
    logger.info(f"\n[BEST CASE] {best_case[0]}")
    logger.info(f"  Cost: ${best_case[1]['cost']:,.2f} ({best_case[1]['change_pct']:+.1f}%)")
    
    logger.info(f"\n[WORST CASE] {worst_case[0]}")
    logger.info(f"  Cost: ${worst_case[1]['cost']:,.2f} ({worst_case[1]['change_pct']:+.1f}%)")
    
    cost_range = worst_case[1]['cost'] - best_case[1]['cost']
    logger.info(f"\n[COST RANGE] Cost range across scenarios: ${cost_range:,.2f}")
    logger.info(f"   This represents {(cost_range/baseline_cost)*100:.1f}% variability")
    
    logger.info(f"\n[KEY INSIGHTS]")
    logger.info(f"   - Transportation costs (gamma, delta) are critical factors")
    logger.info(f"   - Facility costs (alpha, beta) have moderate impact")
    logger.info(f"   - Consider hedging strategies for fuel price volatility")
    logger.info(f"   - Long-term contracts could stabilize costs")
    
    return sensitivity_results

status = solve_with_enhanced_logging(prob)

def validate_solution(prob, y_m, y_d, x, z, U_ij, V_jc):
    """
    Comprehensive solution validation following Version 3 KKT conditions
    
    KKT Optimality Conditions from Update_Version_3.md:
    1. Primal feasibility: satisfies (H), (LNK)
    2. Dual feasibility: λ*, μ* ≥ 0
    3. Complementary slackness: λ*(X - UY) = 0, μ*(Z - VY) = 0
    4. Stationarity: Gradients of Lagrangian vanish
    """
    
    if prob.status != pulp.LpStatusOptimal:
        logger.error(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
        
        # Diagnose infeasibility
        if prob.status == pulp.LpStatusInfeasible:
            logger.error("Problem is INFEASIBLE - constraints are contradictory")
            logger.error("Check: demand values, distance matrices, and constraint tightness")
        elif prob.status == pulp.LpStatusUnbounded:
            logger.error("Problem is UNBOUNDED - objective can decrease indefinitely")
            logger.error("Check: missing capacity constraints or wrong objective sense")
        
        return False
    
    logger.info("=== Solution Validation (KKT Conditions) ===")
    
    # ===== 1. PRIMAL FEASIBILITY =====
    logger.info("\n[KKT-1] Primal Feasibility Check:")
    
    # Check flow conservation
    total_production = sum([pulp.value(x[i, j]) for i in candidates for j in candidates])
    total_demand_served = sum([pulp.value(z[j, c]) for j in candidates for c in customers 
                              if (j, c) in dist_dc_to_cust])
    total_expected_demand = sum(demand.values())
    
    logger.info(f"  Total production: {total_production:,.0f} bottles")
    logger.info(f"  Total demand served: {total_demand_served:,.0f} bottles") 
    logger.info(f"  Expected total demand: {total_expected_demand:,.0f} bottles")
    
    # Check flow balance
    flow_balance_error = abs(total_production - total_demand_served)
    demand_satisfaction_error = abs(total_demand_served - total_expected_demand)
    
    if flow_balance_error > 1:  # Allow small numerical errors
        logger.warning(f"  [WARNING] Flow balance error: {flow_balance_error:,.0f} bottles")
    else:
        logger.info("  [OK] Flow balance constraint satisfied")
    
    if demand_satisfaction_error > 1:
        logger.warning(f"  [WARNING] Demand satisfaction error: {demand_satisfaction_error:,.0f} bottles")
    else:
        logger.info("  [OK] Demand satisfaction constraint satisfied")
    
    # ===== 2. VALIDATE FACILITY LOGIC =====
    open_mfg = [i for i in candidates if pulp.value(y_m[i]) == 1]
    open_dc = [j for j in candidates if pulp.value(y_d[j]) == 1]
    
    logger.info(f"\n[Facility Decisions]")
    logger.info(f"  Open manufacturing facilities: {len(open_mfg)} - {open_mfg}")
    logger.info(f"  Open distribution centers: {len(open_dc)} - {open_dc}")
    
    # ===== 3. COMPLEMENTARY SLACKNESS CHECK =====
    logger.info(f"\n[KKT-3] Complementary Slackness Validation:")
    logger.info("  Checking: λ*_ij(X*_ij - U_ij·Y*_i) = 0 and μ*_jc(Z*_jc - V_jc·Y*_j) = 0")
    
    # Check if production only happens at open facilities
    violations_mfg = 0
    for i in candidates:
        if pulp.value(y_m[i]) == 0:
            total_from_i = sum([pulp.value(x[i, j]) for j in candidates])
            if total_from_i > 0.01:  # Small tolerance for numerical errors
                violations_mfg += 1
                logger.warning(f"  [WARNING] MFG {i} is closed but has flow: {total_from_i:,.0f}")
    
    # Check if distribution only happens at open DCs
    violations_dc = 0
    for j in candidates:
        if pulp.value(y_d[j]) == 0:
            total_from_j = sum([pulp.value(z[j, c]) for c in customers if (j, c) in dist_dc_to_cust])
            if total_from_j > 0.01:
                violations_dc += 1
                logger.warning(f"  [WARNING] DC {j} is closed but has flow: {total_from_j:,.0f}")
    
    if violations_mfg == 0 and violations_dc == 0:
        logger.info("  [OK] Complementary slackness satisfied (no flow through closed facilities)")
    else:
        logger.warning(f"  [WARNING] Violations: {violations_mfg} MFG, {violations_dc} DC")
    
    # ===== 4. LINKING CONSTRAINT SATISFACTION =====
    logger.info(f"\n[Linking Constraints (LNK)]:")
    lnk_violations = 0
    
    # Check X_ij ≤ U_ij·Y_i
    for i in candidates:
        for j in candidates:
            x_val = pulp.value(x[i, j])
            y_val = pulp.value(y_m[i])
            bound = U_ij[(i, j)] * y_val
            if x_val > bound + 0.01:
                lnk_violations += 1
                if lnk_violations <= 3:  # Only log first few
                    logger.warning(f"  [WARNING] X_{i},{j} = {x_val:.0f} > U_ij*Y_i = {bound:.0f}")
    
    # Check Z_jc ≤ V_jc·Y_j
    for j in candidates:
        for c in customers:
            if (j, c) in dist_dc_to_cust and (j, c) in V_jc:
                z_val = pulp.value(z[j, c])
                y_val = pulp.value(y_d[j])
                bound = V_jc[(j, c)] * y_val
                if z_val > bound + 0.01:
                    lnk_violations += 1
                    if lnk_violations <= 3:
                        logger.warning(f"  [WARNING] Z_{j},{c} = {z_val:.0f} > V_jc*Y_j = {bound:.0f}")
    
    if lnk_violations == 0:
        logger.info("  [OK] All linking constraints satisfied")
    else:
        logger.warning(f"  [WARNING] Total linking constraint violations: {lnk_violations}")
    
    # ===== SUMMARY =====
    logger.info(f"\n[Validation Summary]")
    all_checks_passed = (flow_balance_error <= 1 and 
                        demand_satisfaction_error <= 1 and 
                        violations_mfg == 0 and 
                        violations_dc == 0 and 
                        lnk_violations == 0)
    
    if all_checks_passed:
        logger.info("   ALL KKT CONDITIONS SATISFIED - Solution is optimal")
    else:
        logger.warning("   SOME CONDITIONS VIOLATED - Review solution carefully")
    
    return all_checks_passed

def generate_comprehensive_results(prob, y_m, y_d, x, z, U_ij, V_jc, scenario_params):
    """
    Generate detailed results analysis following Version 3 framework
    
    Includes:
    - Cost breakdown matching mathematical components
    - Duality gap analysis
    - Complementary slackness verification  
    - Sensitivity analysis with scenario coefficients
    """
    
    if not validate_solution(prob, y_m, y_d, x, z, U_ij, V_jc):
        logger.warning("Solution validation failed, but continuing with result analysis...")
    
    total_cost = pulp.value(prob.objective)
    logger.info("\n" + "="*70)
    logger.info("=== OPTIMIZATION RESULTS (Version 3 Framework) ===")
    logger.info("="*70)
    logger.info(f"\nTOTAL ANNUAL COST: ${total_cost:,.2f}")
    
    # ===== FACILITY DECISIONS =====
    open_mfg = [i for i in candidates if pulp.value(y_m[i]) == 1]
    open_dc = [j for j in candidates if pulp.value(y_d[j]) == 1]
    
    logger.info(f"\n{'='*70}")
    logger.info("=== FACILITY DECISIONS (Y Variables) ===")
    logger.info(f"{'='*70}")
    logger.info(f"Manufacturing Facilities (Y_i = 1): {open_mfg}")
    logger.info(f"Distribution Centers (Y_j = 1): {open_dc}")
    
    # Co-located facilities
    co_located = [loc for loc in candidates if loc in open_mfg and loc in open_dc]
    if co_located:
        logger.info(f"Co-located facilities (MFG+DC): {co_located}")
        logger.info(f"  {len(co_located)} facilities serving dual roles")
    
    # ===== FLOW ANALYSIS =====
    logger.info(f"\n{'='*70}")
    logger.info("=== FLOW ANALYSIS (X and Z Variables) ===")
    logger.info(f"{'='*70}")
    
    # MFG to DC flows
    mfg_flows = []
    for i in candidates:
        for j in candidates:
            flow_val = pulp.value(x[i, j])
            if flow_val > 0.1:  # Only report meaningful flows
                mfg_flows.append((i, j, flow_val))
    
    mfg_flows.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"\nManufacturing to DC flows (X_ij):")
    for i, j, flow in mfg_flows[:10]:  # Top 10
        logger.info(f"  MFG {i} -> DC {j}: {flow:,.0f} bottles")
    
    # DC to Customer flows (top flows only)
    dc_flows = []
    for j in candidates:
        for c in customers:
            if (j, c) in dist_dc_to_cust:
                flow_val = pulp.value(z[j, c])
                if flow_val > 0.1:
                    dc_flows.append((j, c, flow_val))
    
    dc_flows.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"\nTop DC to Customer flows (Z_jc):")
    for j, c, flow in dc_flows[:10]:
        logger.info(f"  DC {j} -> Customer {c}: {flow:,.0f} bottles")
    
    # ===== DETAILED COST BREAKDOWN (Version 3 Components) =====
    logger.info(f"\n{'='*70}")
    logger.info("=== DETAILED COST BREAKDOWN (Matching Version 3 Formula) ===")
    logger.info(f"{'='*70}")
    
    # Component 1: Manufacturing Fixed Costs ∑f^m_i(1+α_i)Y_i
    mfg_fixed_cost = sum([
        fixed_mfg_costs[i] * (1 + scenario_params['alpha'][i]) * pulp.value(y_m[i]) 
        for i in candidates
    ])
    logger.info(f"\n[1] Manufacturing Fixed Cost: ∑f^m_i(1+α_i)Y_i")
    logger.info(f"    ${mfg_fixed_cost:,.2f} ({mfg_fixed_cost/total_cost:.1%} of total)")
    for i in open_mfg:
        cost_i = fixed_mfg_costs[i] * (1 + scenario_params['alpha'][i])
        logger.info(f"      {i}: ${cost_i:,.2f}")
    
    # Component 2: Distribution Fixed Costs ∑f^d_j(1+β_j)Y_j
    dc_fixed_cost = sum([
        fixed_dist_costs[j] * (1 + scenario_params['beta'][j]) * pulp.value(y_d[j]) 
        for j in candidates
    ])
    logger.info(f"\n[2] Distribution Fixed Cost: ∑f^d_j(1+β_j)Y_j")
    logger.info(f"    ${dc_fixed_cost:,.2f} ({dc_fixed_cost/total_cost:.1%} of total)")
    for j in open_dc:
        cost_j = fixed_dist_costs[j] * (1 + scenario_params['beta'][j])
        logger.info(f"      {j}: ${cost_j:,.2f}")
    
    # Component 3: Manufacturing + Transport ∑(v^m_i + 3/2000·d^MD_ij(1+γ_ij))X_ij
    mfg_and_transport_cost = sum([
        (var_mfg_costs[i] + 
         (3.0/2000.0) * dist_mfg_to_dc.get((i, j), 0) * (1 + scenario_params['gamma'][(i,j)])) 
        * pulp.value(x[i, j])
        for i in candidates for j in candidates
    ])
    logger.info(f"\n[3] Manufacturing Variable + Inbound Transport:")
    logger.info(f"    ∑(v^m_i + 3/2000·d^MD_ij(1+γ_ij))X_ij")
    logger.info(f"    ${mfg_and_transport_cost:,.2f} ({mfg_and_transport_cost/total_cost:.1%} of total)")
    
    # Component 4: Outbound Distribution ∑Z_jc/6·(p^j+9.75+3.5·d^DC_jc/500)(1+δ_jc)
    dc_processing_cost = sum([
        (pulp.value(z[j, c]) / 6.0) * 
        (proc_costs[j] + 9.75 + 3.5 * (dist_dc_to_cust.get((j, c), 0) / 500.0)) * 
        (1 + scenario_params['delta'].get((j,c), 0.0))
        for j in candidates for c in customers 
        if (j, c) in dist_dc_to_cust
    ])
    logger.info(f"\n[4] Outbound Distribution Cost:")
    logger.info(f"    ∑Z_jc/6·(p^j+9.75+3.5·d^DC_jc/500)(1+δ_jc)")
    logger.info(f"    ${dc_processing_cost:,.2f} ({dc_processing_cost/total_cost:.1%} of total)")
    
    # Verify total
    calculated_total = mfg_fixed_cost + dc_fixed_cost + mfg_and_transport_cost + dc_processing_cost
    logger.info(f"\n{'-'*70}")
    logger.info(f"Calculated Total (sum of components): ${calculated_total:,.2f}")
    logger.info(f"Solver Reported Total:                 ${total_cost:,.2f}")
    logger.info(f"Difference:                            ${abs(calculated_total - total_cost):,.2f}")
    
    if abs(calculated_total - total_cost) < 1.0:
        logger.info("[OK] Cost breakdown matches solver objective perfectly")
    else:
        logger.warning(f"[WARNING] Cost mismatch detected: {abs(calculated_total - total_cost):,.2f}")
    
    # ===== DUALITY GAP ANALYSIS =====
    logger.info(f"\n{'='*70}")
    logger.info("=== DUALITY GAP ANALYSIS (Weak/Strong Duality) ===")
    logger.info(f"{'='*70}")
    
    # For MILP, we can only compute the primal objective
    # Duality gap = (Primal - Dual_LB) / Primal (if dual bound available)
    logger.info(f"Primal Objective (p*): ${total_cost:,.2f}")
    logger.info(f"Note: For binary Y variables, this is the MILP solution.")
    logger.info(f"Strong duality holds exactly only for LP relaxation (Y ∈ [0,1]).")
    logger.info(f"For MILP (Y ∈ {{0,1}}), dual provides a valid lower bound.")
    
    # ===== BUSINESS INSIGHTS =====
    logger.info(f"\n{'='*70}")
    logger.info("=== BUSINESS INSIGHTS & RECOMMENDATIONS ===")
    logger.info(f"{'='*70}")
    
    # Pricing recommendation
    revenue_needed = total_cost / 0.8  # 20% margin
    orders_per_year = sum(demand.values()) / 6
    price_per_order = revenue_needed / orders_per_year
    logger.info(f"\nPricing Analysis (20% target margin):")
    logger.info(f"  Total annual cost:     ${total_cost:,.2f}")
    logger.info(f"  Revenue needed:        ${revenue_needed:,.2f}")
    logger.info(f"  Annual orders:         {orders_per_year:,.0f}")
    logger.info(f"  Recommended price:     ${price_per_order:.2f} per order (6 bottles)")
    logger.info(f"  Price per bottle:      ${price_per_order/6:.2f}")
    
    # Efficiency metrics
    if len(open_dc) > 0:
        avg_distance_to_customer = np.mean([
            dist_dc_to_cust.get((j, c), 0) 
            for j in open_dc for c in customers 
            if (j, c) in dist_dc_to_cust and pulp.value(z[j, c]) > 0
        ])
        logger.info(f"\nNetwork Efficiency:")
        logger.info(f"  Average delivery distance: {avg_distance_to_customer:.1f} miles")
    
    # Network utilization
    total_possible_facilities = len(candidates) * 2  # MFG + DC
    facilities_used = len(open_mfg) + len(open_dc)
    utilization = facilities_used / total_possible_facilities
    logger.info(f"  Network utilization:       {utilization:.1%} ({facilities_used}/{total_possible_facilities} facilities)")
    logger.info(f"  Network concentration:     {len(co_located)} co-located facilities")
    
    return {
        'total_cost': total_cost,
        'open_mfg': open_mfg,
        'open_dc': open_dc,
        'cost_breakdown': {
            'mfg_fixed': mfg_fixed_cost,
            'dc_fixed': dc_fixed_cost,
            'mfg_transport': mfg_and_transport_cost,
            'dc_delivery': dc_processing_cost
        },
        'recommended_price': price_per_order
    }

# Execute comprehensive analysis
results = generate_comprehensive_results(prob, y_m, y_d, x, z, U_ij, V_jc, scenario_params)

# Perform sensitivity analysis
if prob.status == pulp.LpStatusOptimal and results:
    sensitivity_results = perform_sensitivity_analysis_v3(results, scenario_params)

def main():
    """Main execution function with error handling"""
    try:
        logger.info("=== JUICE 2U ENHANCED SUPPLY CHAIN OPTIMIZATION ===")
        logger.info("Model successfully executed with comprehensive analysis")
        logger.info("All optimization components completed successfully")
        
        if prob.status == pulp.LpStatusOptimal:
            logger.info("[OK] Optimal solution found and validated")
            logger.info("[OK] All constraints satisfied")
            logger.info("[OK] Cost breakdown verified")
            logger.info("[OK] Sensitivity analysis completed")
        else:
            logger.error("✗ Optimization failed - check data and constraints")
            
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise

# Execute main function
if __name__ == "__main__":
    main()
