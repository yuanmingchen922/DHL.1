import pulp
import pandas as pd

xl = pd.ExcelFile('DHLData.xlsx')

df_customers = pd.read_excel(xl, 'Customers')
df_customers = df_customers.dropna(subset=['City; State', '% of Volume'])

df_costs = pd.read_excel(xl, 'fixed & variable costs')

df_dc_to_cust = pd.read_excel(xl, 'Distances DC to Cust')
df_mfg_to_dc = pd.read_excel(xl, 'Distances MFG to DC')

candidates_full = df_costs['Candidate Facility'].tolist()
customers_full = df_customers['City; State'].tolist()

def simplify_name(name):
    if ';' in name:
        base_name = name.split(';')[0].strip()
    elif ',' in name:
        base_name = name.split(',')[0].strip()
    else:
        base_name = name.strip()
    
    name_mapping = {
        'New York City': 'New York',
        'Nashville-Davidson': 'Nashville',
        'Louisville/Jefferson': 'Louisville', 
        'Oklahoma ': 'Oklahoma City',
        'Oklahoma City ': 'Oklahoma City',  
        'Oklahoma': 'Oklahoma City' 
    }
    
    return name_mapping.get(base_name, base_name)

candidates = [simplify_name(name) for name in candidates_full]
customers = [simplify_name(name) for name in customers_full]

print(f"Candidate facilities: {candidates}")
print(f"Customer cities: {customers[:5]}... (total {len(customers)})")

def clean_cost_value(value):
    if isinstance(value, str):
        value = value.replace('$', '').replace(' ', '')
        if 'K' in value:
            return float(value.replace('K', ''))
        return float(value)
    return float(value)

fixed_mfg_costs = {}
fixed_dist_costs = {}
for i, row in df_costs.iterrows():
    candidate = simplify_name(row['Candidate Facility'])
    fixed_mfg_costs[candidate] = clean_cost_value(row['Manufacturing Fixed Annual Cost'])
    fixed_dist_costs[candidate] = clean_cost_value(row['Distribution Fixed Annual Cost'])

var_mfg_costs = dict(zip(candidates, df_costs['Vairable Manufacturing Cost Per Bottle']))
proc_costs = dict(zip(candidates, df_costs['Distribution Processing Cost Per Order (6 Bottles)']))

print(f"Manufacturing fixed costs: {fixed_mfg_costs}")
print(f"Distribution fixed costs: {fixed_dist_costs}")
print(f"Manufacturing variable costs: {var_mfg_costs}")
print(f"Processing costs: {proc_costs}")

total_volume = 2000000
demand = {}
for i, row in df_customers.iterrows():
    customer_name = simplify_name(row['City; State'])
    demand[customer_name] = row['% of Volume'] * total_volume

print(f"Demand data sample: {dict(list(demand.items())[:5])}")

def build_distance_matrix(df_distance, candidates, locations):
    distance_dict = {}
    
    distance_columns = df_distance.columns[1:].tolist()
    candidate_cols = [simplify_name(col) for col in distance_columns]
    
    location_rows = [simplify_name(name) for name in df_distance.iloc[:, 0].tolist()]
    
    for i, from_candidate in enumerate(candidate_cols):
        if from_candidate in candidates:
            for j, to_location in enumerate(location_rows):
                if to_location in locations:
                    try:
                        distance = df_distance.iloc[j, i+1]
                        distance_dict[(from_candidate, to_location)] = distance
                    except:
                        distance_dict[(from_candidate, to_location)] = 0
                        
    return distance_dict

dist_mfg_to_dc = build_distance_matrix(df_mfg_to_dc, candidates, candidates)
dist_dc_to_cust = build_distance_matrix(df_dc_to_cust, candidates, customers)

print(f"MFG to DC distance data sample: {dict(list(dist_mfg_to_dc.items())[:5])}")
print(f"DC to customer distance data sample: {dict(list(dist_dc_to_cust.items())[:5])}")

prob = pulp.LpProblem("Juice_2U_Network_Design", pulp.LpMinimize)

y_m = pulp.LpVariable.dicts("MFG_Open", candidates, cat='Binary')
y_d = pulp.LpVariable.dicts("DC_Open", candidates, cat='Binary')
x = pulp.LpVariable.dicts("MFG_to_DC", [(i, j) for i in candidates for j in candidates], lowBound=0)
z = pulp.LpVariable.dicts("DC_to_Cust", [(j, c) for j in candidates for c in customers], lowBound=0)

prob += (
    pulp.lpSum([fixed_mfg_costs[i] * 1000 * y_m[i] for i in candidates]) +
    pulp.lpSum([fixed_dist_costs[j] * 1000 * y_d[j] for j in candidates]) +
    pulp.lpSum([var_mfg_costs[i] * x[i, j] for i in candidates for j in candidates]) +
    pulp.lpSum([(x[i, j] / 2000) * 3 * dist_mfg_to_dc.get((i, j), 0) 
                for i in candidates for j in candidates]) +
    pulp.lpSum([(z[j, c] / 6) * (proc_costs[j] + 9.75 + 3.5 * (dist_dc_to_cust.get((j, c), 0) / 500))
                for j in candidates for c in customers if (j, c) in dist_dc_to_cust])
)

M = 2000000

for c in customers:
    prob += pulp.lpSum([z[j, c] for j in candidates if (j, c) in dist_dc_to_cust]) == demand[c]

for j in candidates:
    prob += (pulp.lpSum([z[j, c] for c in customers if (j, c) in dist_dc_to_cust]) == 
             pulp.lpSum([x[i, j] for i in candidates]))

for i in candidates:
    for j in candidates:
        prob += x[i, j] <= M * y_m[i]

for j in candidates:
    for c in customers:
        if (j, c) in dist_dc_to_cust:
            prob += z[j, c] <= M * y_d[j]

print("Starting optimization...")
print(f"Number of candidate facilities: {len(candidates)}")
print(f"Number of customers: {len(customers)}")
print(f"Available distance data: MFG to DC {len(dist_mfg_to_dc)}, DC to customer {len(dist_dc_to_cust)}")

status = prob.solve()

print("Status:", pulp.LpStatus[prob.status])
print("Solver Status Code:", status)

if prob.status == pulp.LpStatusOptimal:
    total_cost = pulp.value(prob.objective)
    print("Total Annual Cost: ${:,.2f}".format(total_cost))

    print("\nOpen Facilities:")
    open_mfg = []
    open_dc = []
    for i in candidates:
        if pulp.value(y_m[i]) == 1:
            open_mfg.append(i)
            print(f"MFG at {i}")
        if pulp.value(y_d[i]) == 1:
            open_dc.append(i)
            print(f"DC at {i}")

    print("\nFlows (bottles):")
    print("=== MFG to DC ===")
    for i in candidates:
        for j in candidates:
            flow = pulp.value(x[i, j])
            if flow and flow > 0:
                print(f"From MFG {i} to DC {j}: {flow:,.0f}")
    
    print("=== DC to Customer ===")
    for j in candidates:
        for c in customers:
            if (j, c) in dist_dc_to_cust:
                flow = pulp.value(z[j, c])
                if flow and flow > 0:
                    print(f"From DC {j} to Customer {c}: {flow:,.0f}")

    revenue_needed = total_cost / 0.8
    orders_per_year = 2000000 / 6
    price_per_order = revenue_needed / orders_per_year
    print(f"\nRecommended Price per Order (20% margin): ${price_per_order:,.2f}")
    
    print("\n=== Cost Breakdown ===")
    mfg_fixed_cost = sum([fixed_mfg_costs[i] * 1000 * pulp.value(y_m[i]) for i in candidates])
    dc_fixed_cost = sum([fixed_dist_costs[j] * 1000 * pulp.value(y_d[j]) for j in candidates])
    mfg_var_cost = sum([var_mfg_costs[i] * pulp.value(x[i, j]) for i in candidates for j in candidates])
    transport_cost = sum([(pulp.value(x[i, j]) / 2000) * 3 * dist_mfg_to_dc.get((i, j), 0) 
                         for i in candidates for j in candidates])
    dc_processing_cost = sum([(pulp.value(z[j, c]) / 6) * (proc_costs[j] + 9.75 + 3.5 * (dist_dc_to_cust.get((j, c), 0) / 500))
                             for j in candidates for c in customers if (j, c) in dist_dc_to_cust])
    
    print(f"Manufacturing fixed cost: ${mfg_fixed_cost:,.2f}")
    print(f"Distribution fixed cost: ${dc_fixed_cost:,.2f}")
    print(f"Manufacturing variable cost: ${mfg_var_cost:,.2f}")
    print(f"MFG to DC transport cost: ${transport_cost:,.2f}")
    print(f"DC processing and delivery cost: ${dc_processing_cost:,.2f}")
    
else:
    print("Optimization failed or no optimal solution found")
