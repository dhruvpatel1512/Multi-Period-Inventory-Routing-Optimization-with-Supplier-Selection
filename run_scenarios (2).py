from sensitivity_analysis_v2 import run_model
# Define base demand
base_demand = [
    [10, 20, 40],
    [20, 10, 30],
    [30, 30, 10],
    [20, 40, 10]
]
# Scenario configurations
scenarios = [
    {
        "name": "Baseline",
        "demand": base_demand,
        "vehicle_capacity": 250,
        "transport_cost": 10,
        "holding_cost": [10, 20, 5, 10],
        "ordering_cost" : [10, 20, 15, 25]
    },
    {
    "name": "Peak Demand",
    "demand": [
        [20, 30, 50],
        [30, 20, 40],
        [40, 40, 20],
        [30, 50, 20]
    ],
    "vehicle_capacity": 250,
    "transport_cost": 10,
    "holding_cost": [10, 20, 5, 10],
    "ordering_cost" : [10, 20, 15, 25]
    },
    {
        "name": "Higher Transport Cost",
        "demand": base_demand,
        "vehicle_capacity": 250,
        "transport_cost": 15,
        "holding_cost": [10, 20, 5, 10],
        "ordering_cost" : [10, 20, 15, 25]
    },
    {
        "name": "Lower Vehicle Capacity",
        "demand": base_demand,
        "vehicle_capacity": 200,
        "transport_cost": 10,
        "holding_cost": [10, 20, 5, 10],
        "ordering_cost" : [10, 20, 15, 25]
    },
    {
        "name": "Higher Holding Cost",
        "demand": base_demand,
        "vehicle_capacity": 250,
        "transport_cost": 10,
        "holding_cost": [20, 25, 10, 20],
        "ordering_cost" : [10, 20, 15, 25]
    },
    {
        "name": "Higher Ordering Cost",
        "demand": base_demand,
        "vehicle_capacity": 250,
        "transport_cost": 10,
        "holding_cost": [20, 25, 10, 20],
        "ordering_cost" : [20, 40, 30, 40]
    }
]
# Run all scenarios
for scenario in scenarios:
    print(f"\n🧪 Running Scenario: {scenario['name']}")
    print("--------------------------------------------------")
    result = run_model(
        demand_data=scenario["demand"],
        vehicle_capacity_val=scenario["vehicle_capacity"],
        transport_cost_val=scenario["transport_cost"],
        holding_costs=scenario["holding_cost"],
        ordering_cost=scenario["ordering_cost"]
    )
    if result is not None and 'total_cost' in result:
        print(f"{scenario['name']} ➜ Total Cost: {result['total_cost']}")
    # else:
    #     print(f"{scenario['name']} ➜ No solution or error occurred")