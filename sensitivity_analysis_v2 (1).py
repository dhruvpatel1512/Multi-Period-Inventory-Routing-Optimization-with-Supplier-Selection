def run_model(demand_data, vehicle_capacity_val, transport_cost_val, holding_costs, ordering_cost):
    from ortools.linear_solver import pywraplp

    # Create solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return {"total_cost": None}
    
    demand = demand_data
    vehicle_capacity = vehicle_capacity_val
    transport_cost_per_km = transport_cost_val
    holding_cost = holding_costs

    # Sets
    I = range(4)  # suppliers 0–3
    J = range(4)  # components 0–3
    T = range(3)  # periods 0–2
    F = range(2)  # vehicles 0–1
    K = range(4)  # discount intervals 0–3
    N = 5  

    # Data
    supply = [
        [  # Supplier 1
            [10, 10, 10],  # Component 1 capacity per period
            [20, 20, 20],  # Component 2
            [5, 5, 5],     # Component 3
            [10, 10, 10],  # Component 4
        ],
        [  # Supplier 2
            [5, 5, 5],
            [15, 15, 15],
            [10, 10, 10],
            [20, 20, 20],
        ],
        [  # Supplier 3
            [4, 4, 4],
            [30, 30, 30],
            [12, 12, 12],
            [15, 15, 15],
        ],
        [  # Supplier 4
            [20, 20, 20],
            [10, 10, 10],
            [22, 22, 22],
            [30, 30, 30],
        ],
    ]
    # UNDISCOUNTED PRICE - price[i][j]
    price = [
        [10, 20, 30, 20],  # Supplier 1
        [20, 10, 30, 40],  # Supplier 2
        [40, 30, 10, 10],  # Supplier 3
        [10, 20, 15, 30]   # Supplier 4
    ]

    # DISTANCE MATRIX - depot is index 4 (5th row/column)
    distance = [
        [0, 20, 30, 50, 30],  # Supplier 1
        [20, 0, 30, 40, 20],  # Supplier 2
        [30, 30, 0, 10, 40],  # Supplier 3
        [50, 40, 10, 0, 30],  # Supplier 4
        [30, 20, 40, 30, 0]   # Depot (index 4)
    ]

    # DISCOUNT RATIOS - discount_ratio[i][k]
    discount_ratio = [
        [1, 0.8, 0.65, 0.55],
        [1, 0.75, 0.65, 0.5],
        [1, 0.9, 0.8, 0.7],
        [1, 0.85, 0.7, 0.55]
    ]

    # DISCOUNT LIMITS - discount_limit[i][k]
    discount_limit = [
        [0, 500, 1000, 1500],
        [0, 1000, 3000, 5000],
        [0, 1000, 1500, 2000],
        [0, 2000, 2800, 3500]
    ]

    # UNIT WEIGHT (uj)
    unit_weight = [3, 3, 3, 3]

    # TRANSPORTATION COSTS
    fixed_vehicle_cost = 20

    # Big M for constraints
    M = 20000

    # Decision Variables
    # Q[i][j][t] - quantity ordered from supplier i of component j in period t
    Q = [[[solver.IntVar(0, solver.infinity(), f'Q_{i}_{j}_{t}') for t in T] for j in J] for i in I]

    # Ivar[j][t] - inventory level of component j at period t, note t in [0..T], so T+1 length
    Ivar = [[solver.IntVar(0, solver.infinity(), f'I_{j}_{t}') for t in range(len(T)+1)] for j in J]

    # X[i][t] - binary, 1 if ordering from supplier i in period t
    X = [[solver.BoolVar(f'X_{i}_{t}') for t in T] for i in I]

    # Tit[i][t] - total amount before discount from supplier i in period t
    Tit = [[solver.NumVar(0, solver.infinity(), f'Tit_{i}_{t}') for t in T] for i in I]

    # Vit[i][t] - total amount after discount from supplier i in period t
    Vit = [[solver.NumVar(0, solver.infinity(), f'Vit_{i}_{t}') for t in T] for i in I]

    # U[i][t][k] - binary, 1 if discount interval k applied for supplier i in period t
    U = [[[solver.BoolVar(f'U_{i}_{t}_{k}') for k in K] for t in T] for i in I]

    N = len(distance)  # total number of locations including depot
    Rf = [[[[solver.BoolVar(f'Rf_{f}_{t}_{m}_{n}') for n in range(N)] for m in range(N)] for t in T] for f in F]

    # Y[f][t] - binary, 1 if vehicle f is used in period t
    Y = [[solver.BoolVar(f'Y_{f}_{t}') for t in T] for f in F]

    # Objective Function
    total_cost = solver.Objective()

    # Purchasing cost (sum of discounted amount Vit[i][t])
    for i in I:
        for t in T:
            total_cost.SetCoefficient(Vit[i][t], 1)

    # Ordering cost (ordering cost * X[i][t])
    for i in I:
        for t in T:
            total_cost.SetCoefficient(X[i][t], ordering_cost[i])

    for j in J:
        for t in T:
            total_cost.SetCoefficient(Ivar[j][t + 1], holding_cost[j])

    # Fixed vehicle cost (fixed_vehicle_cost * Y[f][t])
    for f in F:
        for t in T:
            total_cost.SetCoefficient(Y[f][t], fixed_vehicle_cost)

    # Transportation cost (transport_cost_per_km * distance[m][n] * Rf[f][t][m][n])
    for f in F:
        for t in T:
            for m in range(len(distance)):
                for n in range(len(distance[0])):
                    total_cost.SetCoefficient(Rf[f][t][m][n], transport_cost_per_km * distance[m][n])
    total_cost.SetMinimization()

    # Constraints
    depot_index = 4
    N = len(distance)

    # Inventory balance constraints:
    for j in J:
        for t in T[:-1]:  # t from 0 to T-2 (because we use t+1)
            solver.Add(Ivar[j][t + 1] == Ivar[j][t] - demand[j][t] + 
                    sum(Q[i][j][t] for i in I))

    # Initial inventory = 0
    for j in J:
        solver.Add(Ivar[j][0] == 0)

    # Final inventory = 0
    for j in J:
        solver.Add(Ivar[j][len(T)] == 0)

    # Supplier capacity constraints:
    for i in I:
        for j in J:
            for t in T:
                solver.Add(Q[i][j][t] <= supply[i][j][t])

    # Linking quantity and order decision:
    for i in I:
        for t in T:
            solver.Add(sum(Q[i][j][t] for j in J) <= M * X[i][t])

    # Calculate total amount before discount Tit[i][t]:
    for i in I:
        for t in T:
            solver.Add(Tit[i][t] == sum(Q[i][j][t] * price[i][j] for j in J))

    # Auxiliary variable to linearize product Tit[i][t] * U[i][t][k]
    Z = [[[solver.NumVar(0, solver.infinity(), f'Z_{i}_{t}_{k}') for k in K] for t in T] for i in I]
    epsilon = 0.001
    for i in I:
        for t in T:
            # Link Z and Tit, U
            for k in K:
                solver.Add(Z[i][t][k] <= Tit[i][t])
                solver.Add(Z[i][t][k] <= M * U[i][t][k])
                solver.Add(Z[i][t][k] >= Tit[i][t] - M * (1 - U[i][t][k]))
                solver.Add(Z[i][t][k] >= 0)
                
            # Vit calculation as weighted sum of Z by discount_ratio
            solver.Add(Vit[i][t] == sum(discount_ratio[i][k] * Z[i][t][k] for k in K))

            # Discount intervals constraints (Big-M):
            for k in range(len(K) - 1):
                solver.Add(discount_limit[i][k] + M * (U[i][t][k] - 1) <= Tit[i][t])
                solver.Add(Tit[i][t] <= discount_limit[i][k + 1] + M * (1 - U[i][t][k]) - epsilon)

            # Only one discount interval can be chosen:
            solver.Add(sum(U[i][t][k] for k in K) == 1)

    # Vehicle capacity constraints (without full routing)
    for t in T:
        total_weight = solver.Sum(
            Q[i][j][t] * unit_weight[j]
            for i in I for j in J
        )
        total_vehicle_capacity = solver.Sum(
            Y[f][t] * vehicle_capacity
            for f in F
        )
        solver.Add(total_weight <= total_vehicle_capacity)

    aux_weight = [[[[solver.NumVar(0, solver.infinity(), f'aux_weight_{f}_{t}_{i}_{j}') 
                    for j in J] for i in I] for t in T] for f in F]

    # Vehicle must leave and return to depot *at least* once if used
    for f in F:
        for t in T:
            solver.Add(sum(Rf[f][t][depot_index][n] for n in range(N) if n != depot_index) >= Y[f][t])
            solver.Add(sum(Rf[f][t][m][depot_index] for m in range(N) if m != depot_index) >= Y[f][t])

    for f in F:
        for t in T:
            solver.Add(
                sum(Rf[f][t][depot_index][n] for n in range(N) if n != depot_index) >= Y[f][t]
            )
            solver.Add(
                sum(Rf[f][t][m][depot_index] for m in range(N) if m != depot_index) >= Y[f][t]
            )
                
    # For each vehicle f, period t, and supplier i
    for f in F:
        for t in T:
            for i in I:
                # Sum of arcs entering i = sum of arcs leaving i = 1 if supplier i visited
                solver.Add(
                    sum(Rf[f][t][m][i] for m in range(N) if m != i) ==
                    sum(Rf[f][t][i][n] for n in range(N) if n != i)
                )

    for f in F:
        for t in T:
            for n in range(N):
                solver.Add(
                    sum(Rf[f][t][m][n] for m in range(N) if m != n) ==
                    sum(Rf[f][t][n][m] for m in range(N) if m != n)
                )
                
    for f in F:
        for t in T:
            solver.Add(
                sum(Rf[f][t][N-1][n] for n in range(N-1)) >= Y[f][t]
            )
            solver.Add(
                sum(Rf[f][t][n][N-1] for n in range(N-1)) >= Y[f][t]
            )
            
    # Link routing with ordering: if supplier i is visited in period t, X[i][t] must be 1
    for f in F:
        for t in T:
            for i in I:
                incoming = sum(Rf[f][t][m][i] for m in range(depot_index+1) if m != i)
                solver.Add(incoming <= X[i][t])
                
    for f in F:
        for t in T:
            for n in range(N):
                solver.Add(
                    sum(Rf[f][t][n][m] for m in range(N) if m != n) <= Y[f][t] * len(I)
                )
                
    for i in I:
        for t in T:
            visit_sum = sum(Rf[f][t][m][i] for f in F for m in range(N) if m != i)
            solver.Add(visit_sum <= len(F) * X[i][t])

    # Uaux[f][t][i] - order of visit for supplier i by vehicle f in period t
    Uaux = [[[solver.NumVar(0, len(I), f'Uaux_{f}_{t}_{i}') for i in I] for t in T] for f in F]

    for f in F:
        for t in T:
            for i in I:
                for j in I:
                    if i != j:
                        solver.Add(Uaux[f][t][i] + 1 <= Uaux[f][t][j] + len(I) * (1 - Rf[f][t][i][j]))

    for f in F:
        for t in T:
            for i in I:
                solver.Add(Uaux[f][t][i] >= 1 - (1 - sum(Rf[f][t][m][i] for m in range(N) if m != i)))

    # ---- Per-Vehicle Capacity Constraints ----
    for f in F:
        for t in T:
            for i in I:
                for j in J:
                    # aux_weight[f][t][i][j] == Q[i][j][t] if vehicle f visits supplier i (from depot), else 0
                    solver.Add(aux_weight[f][t][i][j] <= Q[i][j][t])
                    solver.Add(aux_weight[f][t][i][j] <= M * Rf[f][t][depot_index][i])
                    solver.Add(aux_weight[f][t][i][j] >= Q[i][j][t] - M * (1 - Rf[f][t][depot_index][i]))
            
            # Total weight carried by vehicle f at period t
            total_weight = solver.Sum(
                unit_weight[j] * aux_weight[f][t][i][j]
                for i in I for j in J
            )
            solver.Add(total_weight <= vehicle_capacity * Y[f][t])

    for f in F:
        for t in T:
            for m in I:
                for n in I:
                    if m != n:
                        solver.Add(Rf[f][t][m][n] + Rf[f][t][n][m] <= 1)

    # Encourage each vehicle to visit at least 2 suppliers if used
    for f in F:
        for t in T:
            supplier_visits = solver.Sum(Rf[f][t][m][n] for m in I for n in I if m != n)
            solver.Add(supplier_visits >= 1.5 * Y[f][t])  # at least 2 visits if vehicle used

    solver.Objective().SetMinimization()
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("\n✅ OPTIMAL SOLUTION FOUND")
        print("--------------------------------------------------")

        # Total cost
        print(f"🔹 Total Cost: {solver.Objective().Value():.2f}")
        print("--------------------------------------------------\n")

        # Demand vs Supplied Quantity by Component
        print("📊 Demand vs Supplied Quantity by Component:")
        for j in J:
            total_demand = sum(demand[j])
            total_supplied = sum(Q[i][j][t].solution_value() for i in I for t in T)
            print(f"  Component {j+1}: Demand = {total_demand}, Supplied = {total_supplied:.2f}")
        print("--------------------------------------------------\n")

        # Supplier-wise order amounts (before and after discount)
        print("📦 Supplier Order Summary (Before and After Discount):")
        for i in I:
            for t in T:
                before = Tit[i][t].solution_value()
                after = Vit[i][t].solution_value()
                if before > 0:
                    print(f"  Supplier {i+1} | Period {t+1}: Before Discount = {before:.2f}, After Discount = {after:.2f}")
        print("--------------------------------------------------\n")

        # Component quantities from each supplier per period
        print("🧮 Component Quantity Supplied by Each Supplier (Qijt):")
        for t in T:
            print(f"Period {t+1}:")
            for i in I:
                for j in J:
                    qty = Q[i][j][t].solution_value()
                    if qty > 0:
                        print(f"  Supplier {i+1} -> Component {j+1} | Quantity: {qty}")
            print()
        print("--------------------------------------------------\n")

        # Inventory levels by component and period
        print("📦 Inventory Levels by Component and Period:")
        # Ivar index goes from 0 to len(T), so include last period inventory
        for j in J:
            for t in range(len(T)+1):
                inv = Ivar[j][t].solution_value()
                print(f"  Component {j+1} | Period {t}: Inventory = {inv}")
        print("--------------------------------------------------\n")

        # Vehicle usage and routes
        print("🚚 Vehicle Routes:")
        depot_index = len(distance) - 1  # depot assumed last index in distance matrix
        N = len(distance)
        for f in F:
            for t in T:
                used = Y[f][t].solution_value()
                if used > 0.5:
                    print(f"  Vehicle {f+1} | Period {t+1}:")
                    # Build route
                    route = [depot_index]
                    current = depot_index
                    while True:
                        next_node = None
                        for n in range(N):
                            if n != current and Rf[f][t][current][n].solution_value() > 0.5:
                                next_node = n
                                break
                        if next_node is None or next_node == depot_index:
                            route.append(depot_index)
                            break
                        else:
                            route.append(next_node)
                            current = next_node
                    # Print route with human-readable names
                    route_str = " -> ".join(
                        f"Depot" if x == depot_index else f"Supplier {x+1}" for x in route
                    )
                    print(f"    Route: {route_str}")
                else:
                    print(f"  Vehicle {f+1} | Period {t+1}: Not used")
        print("--------------------------------------------------\n")

        # Summary: Which suppliers and vehicles were used
        print("📋 Summary of Used Suppliers and Vehicles:")
        for i in I:
            for t in T:
                if X[i][t].solution_value() > 0.5:
                    print(f"  Supplier {i+1} used in Period {t+1}")
        for f in F:
            for t in T:
                if Y[f][t].solution_value() > 0.5:
                    print(f"  Vehicle {f+1} used in Period {t+1}")
        print("--------------------------------------------------")

    else:
        print("\n❌ No optimal solution found.")
