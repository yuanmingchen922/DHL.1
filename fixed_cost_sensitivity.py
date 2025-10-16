"""
固定配送成本敏感性分析
分析固定配送成本增加多少会改变最优位置选择
"""

import pulp
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入原始代码的数据处理函数
exec(open('1_clean.py').read())

def run_fixed_cost_analysis(fixed_cost_multiplier: float = 1.0,
                            force_newark: bool = False,
                            outbound_cost_multiplier: float = 1.0,
                            newark_demand_share: float = 0.0):
    """
    运行固定成本敏感性分析
    
    参数:
        fixed_cost_multiplier: 固定配送成本倍数
        force_newark: 是否强制Newark承担指定比例的需求
        outbound_cost_multiplier: 出站运输成本倍数
        newark_demand_share: Newark必须承担的需求比例（0-1）
    """
    
    # 创建优化问题
    prob = pulp.LpProblem("Fixed_Cost_Sensitivity", pulp.LpMinimize)
    
    # 决策变量
    y_m = pulp.LpVariable.dicts("MFG_Open", candidates, cat='Binary')
    y_d = pulp.LpVariable.dicts("DC_Open", candidates, cat='Binary')
    x = pulp.LpVariable.dicts("MFG_to_DC", 
                             [(i, j) for i in candidates for j in candidates], 
                             lowBound=0, cat='Continuous')
    z = pulp.LpVariable.dicts("DC_to_Cust", 
                             [(j, c) for j in candidates for c in customers 
                              if (j, c) in dist_dc_to_cust], 
                             lowBound=0, cat='Continuous')
    
    # 目标函数组件（应用固定成本倍数）
    mfg_fixed_cost = pulp.lpSum([fixed_mfg_costs[i] * y_m[i] for i in candidates])
    dist_fixed_cost = pulp.lpSum([fixed_cost_multiplier * fixed_dist_costs[j] * y_d[j] 
                                  for j in candidates])
    mfg_var_cost = pulp.lpSum([var_mfg_costs[i] * x[i, j] 
                              for i in candidates for j in candidates])
    inbound_transport_cost = pulp.lpSum([(x[i, j] / 2000) * 3 * dist_mfg_to_dc.get((i, j), 0)
                                        for i in candidates for j in candidates])
    outbound_cost = pulp.lpSum([outbound_cost_multiplier * (z[j, c] / 6) * 
                                (proc_costs[j] + 9.75 + 3.5 * (dist_dc_to_cust.get((j, c), 0) / 500))
                               for j in candidates for c in customers 
                               if (j, c) in dist_dc_to_cust])
    
    # 总目标函数
    prob += (mfg_fixed_cost + dist_fixed_cost + mfg_var_cost + 
             inbound_transport_cost + outbound_cost)
    
    # 添加约束
    total_demand_volume = sum(demand.values())
    M = total_demand_volume * 1.2
    
    # 1. 需求满足约束
    for c in customers:
        if c in demand and demand[c] > 0:
            prob += (pulp.lpSum([z[j, c] for j in candidates if (j, c) in dist_dc_to_cust]) 
                    == demand[c], f"Demand_{c}")
    
    # 2. 流平衡约束
    for j in candidates:
        inflow = pulp.lpSum([x[i, j] for i in candidates])
        outflow = pulp.lpSum([z[j, c] for c in customers if (j, c) in dist_dc_to_cust])
        prob += (inflow == outflow, f"FlowBalance_{j}")
    
    # 3. 制造能力约束
    for i in candidates:
        total_production = pulp.lpSum([x[i, j] for j in candidates])
        prob += (total_production <= M * y_m[i], f"MFG_Cap_{i}")
    
    # 4. 配送能力约束
    for j in candidates:
        total_distribution = pulp.lpSum([z[j, c] for c in customers if (j, c) in dist_dc_to_cust])
        prob += (total_distribution <= M * y_d[j], f"DC_Cap_{j}")
    
    # 5. Newark需求分担约束（如果指定）
    if force_newark and newark_demand_share > 0:
        newark_facility = 'Newark'
        if newark_facility in candidates:
            required_newark_demand = total_demand_volume * newark_demand_share
            total_newark_distribution = pulp.lpSum([z[newark_facility, c] for c in customers 
                                                   if (newark_facility, c) in dist_dc_to_cust])
            prob += (total_newark_distribution >= required_newark_demand, 
                    f"Newark_MinDemand_{newark_demand_share}")
            prob += (total_newark_distribution <= required_newark_demand * 1.01,
                    f"Newark_MaxDemand_{newark_demand_share}")
    
    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 返回结果
    return {
        'status': pulp.LpStatus[prob.status],
        'total_cost': pulp.value(prob.objective),
        'mfg_fixed': pulp.value(mfg_fixed_cost),
        'dist_fixed': pulp.value(dist_fixed_cost),
        'mfg_var': pulp.value(mfg_var_cost),
        'inbound': pulp.value(inbound_transport_cost),
        'outbound': pulp.value(outbound_cost),
        'facilities_mfg': [i for i in candidates if pulp.value(y_m[i]) > 0.5],
        'facilities_dc': [j for j in candidates if pulp.value(y_d[j]) > 0.5],
        'flow_distribution': {j: sum([pulp.value(z[j, c]) for c in customers 
                                     if (j, c) in dist_dc_to_cust])
                             for j in candidates if pulp.value(y_d[j]) > 0.5}
    }

def find_breakeven_fixed_cost():
    """
    找到固定配送成本的临界点，使得最优位置发生变化
    """
    print("\n" + "="*90)
    print("固定配送成本临界点分析")
    print("="*90)
    
    # 基准场景
    baseline = run_fixed_cost_analysis(1.0)
    baseline_facilities = set(baseline['facilities_dc'])
    
    print(f"\n基准场景 (固定成本倍数 = 1.0):")
    print(f"  总成本: ${baseline['total_cost']:,.2f}")
    print(f"  配送设施: {baseline_facilities}")
    print(f"  固定配送成本: ${baseline['dist_fixed']:,.2f}")
    
    # 二分搜索找临界点
    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    
    results = []
    for mult in multipliers:
        result = run_fixed_cost_analysis(mult)
        current_facilities = set(result['facilities_dc'])
        changed = current_facilities != baseline_facilities
        
        results.append({
            'multiplier': mult,
            'total_cost': result['total_cost'],
            'dist_fixed': result['dist_fixed'],
            'facilities': current_facilities,
            'changed': changed,
            'flow': result['flow_distribution']
        })
        
        print(f"\n倍数 {mult:.1f}x:")
        print(f"  总成本: ${result['total_cost']:,.2f} ({(result['total_cost']/baseline['total_cost']-1)*100:+.2f}%)")
        print(f"  固定配送成本: ${result['dist_fixed']:,.2f}")
        print(f"  配送设施: {current_facilities}")
        if changed:
            print(f"  *** 位置发生变化！从 {baseline_facilities} 变为 {current_facilities} ***")
        
        # 显示流量分配
        for facility, flow in result['flow_distribution'].items():
            pct = flow / sum(demand.values()) * 100
            print(f"    {facility}: {flow:,.0f} 瓶 ({pct:.1f}%)")
    
    return results

def analyze_newark_50pct_with_outbound():
    """
    分析Newark分担50%需求 + 出站成本增加50%的组合场景
    """
    print("\n" + "="*90)
    print("组合场景分析：Newark分担50%需求 + 出站成本增加50%")
    print("="*90)
    
    # 基准场景
    baseline = run_fixed_cost_analysis(1.0, False, 1.0, 0.0)
    
    # 场景1: 仅Newark 50%
    scenario1 = run_fixed_cost_analysis(1.0, True, 1.0, 0.5)
    
    # 场景2: 仅出站成本+50%
    scenario2 = run_fixed_cost_analysis(1.0, False, 1.5, 0.0)
    
    # 场景3: 组合场景
    scenario3 = run_fixed_cost_analysis(1.0, True, 1.5, 0.5)
    
    scenarios = {
        '基准': baseline,
        'Newark 50%': scenario1,
        '出站成本+50%': scenario2,
        'Newark 50% + 出站成本+50%': scenario3
    }
    
    print("\n场景对比：")
    print("-" * 90)
    print(f"{'场景':<30} {'总成本 ($)':<20} {'变化 ($)':<20} {'变化率 (%)':<15}")
    print("-" * 90)
    
    for name, result in scenarios.items():
        cost_change = result['total_cost'] - baseline['total_cost']
        pct_change = (result['total_cost'] / baseline['total_cost'] - 1) * 100
        print(f"{name:<30} {result['total_cost']:>18,.2f} {cost_change:>18,.2f} {pct_change:>13.2f}%")
    
    print("\n详细成本分解：")
    for name, result in scenarios.items():
        print(f"\n{name}:")
        print(f"  制造固定成本:   ${result['mfg_fixed']:>12,.2f}")
        print(f"  配送固定成本:   ${result['dist_fixed']:>12,.2f}")
        print(f"  制造变动成本:   ${result['mfg_var']:>12,.2f}")
        print(f"  入站运输成本:   ${result['inbound']:>12,.2f}")
        print(f"  出站配送成本:   ${result['outbound']:>12,.2f}")
        print(f"  ───────────────────────────")
        print(f"  总成本:         ${result['total_cost']:>12,.2f}")
        print(f"\n  配送设施: {result['facilities_dc']}")
        print(f"  流量分配:")
        for facility, flow in result['flow_distribution'].items():
            pct = flow / sum(demand.values()) * 100
            print(f"    {facility}: {flow:>10,.0f} 瓶 ({pct:>5.1f}%)")
    
    # 计算交互效应
    individual_impact = (scenario1['total_cost'] - baseline['total_cost']) + \
                       (scenario2['total_cost'] - baseline['total_cost'])
    combined_impact = scenario3['total_cost'] - baseline['total_cost']
    interaction_effect = combined_impact - individual_impact
    
    print("\n" + "="*90)
    print("交互效应分析：")
    print("-" * 90)
    print(f"Newark 50%单独影响:        ${scenario1['total_cost'] - baseline['total_cost']:>12,.2f}")
    print(f"出站成本+50%单独影响:      ${scenario2['total_cost'] - baseline['total_cost']:>12,.2f}")
    print(f"两者独立影响之和:          ${individual_impact:>12,.2f}")
    print(f"组合场景实际影响:          ${combined_impact:>12,.2f}")
    print(f"交互效应:                  ${interaction_effect:>12,.2f}")
    
    if abs(interaction_effect) < 1000:
        print("\n结论: 两个因素基本独立，交互效应可忽略不计")
    elif interaction_effect > 0:
        print(f"\n结论: 存在负向协同效应，组合影响比预期多 ${interaction_effect:,.2f} ({(interaction_effect/individual_impact)*100:.1f}%)")
    else:
        print(f"\n结论: 存在正向协同效应，组合影响比预期少 ${-interaction_effect:,.2f} ({(-interaction_effect/individual_impact)*100:.1f}%)")
    
    return scenarios

if __name__ == "__main__":
    print("\n" + "="*90)
    print("DHL供应链固定成本敏感性分析")
    print("="*90)
    
    # 1. 找固定成本临界点
    print("\n\n第一部分：固定配送成本临界点分析")
    print("="*90)
    breakeven_results = find_breakeven_fixed_cost()
    
    # 2. Newark 50% + 出站成本+50%组合分析
    print("\n\n第二部分：组合场景分析")
    print("="*90)
    combined_scenarios = analyze_newark_50pct_with_outbound()
    
    print("\n" + "="*90)
    print("分析完成！")
    print("="*90)
