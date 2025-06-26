from pulp import (
    LpProblem, LpVariable, LpMinimize, LpBinary, LpInteger, LpContinuous,
    lpSum, LpStatusOptimal, LpStatus, value, PULP_CBC_CMD
)
from datetime import date
from typing import List, Dict, Optional
import string
import time
import random
import csv
import io
import streamlit as st

class Property:
    """Represents a single rental property with its details."""
    def __init__(self, property_id: str, total_units: int, rate_per_day: float, occupancy_rating: int, group: str):
        self.property_id = property_id
        self.total_units = total_units
        self.rate_per_day = rate_per_day
        self.occupancy_rating = occupancy_rating
        self.group = group

class LeasePlan:
    """Represents a leased portion of a property in a solution."""
    def __init__(self, property_id: str, units: int, rate_per_day: float, occupancy_rating_at_selection: int, full_property: bool, pre_selected: bool = False):
        self.property_id = property_id
        self.units = units
        self.rate_per_day = rate_per_day
        self.occupancy_rating_at_selection = occupancy_rating_at_selection
        self.full_property = full_property
        self.pre_selected = pre_selected # Indicates if this property was pre-selected

class LeaseOptimizerMILP:
    """
    Optimizes a lease plan for rental properties given client requirements
    and property constraints using Mixed-Integer Linear Programming (MILP).
    """
    def __init__(
        self,
        properties: List[Property],
        units_required: int,
        budget: float, # This is now interpreted as minimum acceptable total cost
        start_date: date,
        end_date: date,
        budget_tolerance_percent: float = 5.0, # This now only applies to the upper bound
        unit_tolerance_percent: float = 10.0,
        full_property_penalty_per_unit: float = 1.0,
        min_properties_per_group: int = 2,
        occupancy_penalty_per_unit_per_day: float = 100.0,
        occupancy_rating_threshold: int = 6,
        solver_time_limit_seconds: Optional[int] = None,
        solver_mip_gap: Optional[float] = None,
        explicit_pre_selected_property_ids: Optional[List[str]] = None
    ):
        self.properties = properties
        self.units_required = units_required
        self.budget = budget
        self.start_date = start_date
        self.end_date = end_date
        self.days = (end_date - start_date).days + 1
        self.budget_tolerance = budget_tolerance_percent / 100
        self.unit_tolerance = unit_tolerance_percent / 100
        self.full_property_penalty_per_unit = full_property_penalty_per_unit
        self.min_properties_per_group = min_properties_per_group
        self.occupancy_penalty_per_unit_per_day = occupancy_penalty_per_unit_per_day
        self.occupancy_rating_threshold = occupancy_rating_threshold
        self.solver_time_limit_seconds = solver_time_limit_seconds
        self.solver_mip_gap = solver_mip_gap

        self.properties_by_group: Dict[str, List[int]] = {group_char: [] for group_char in string.ascii_uppercase}
        for i, p in enumerate(self.properties):
            if p.group in self.properties_by_group:
                self.properties_by_group[p.group].append(i)

        self.active_groups = {
            group for group, prop_indices in self.properties_by_group.items()
            if len(prop_indices) >= self.min_properties_per_group
        }
        
        self.pre_selected_property_indices: List[int] = []
        self.pre_selected_property_ids: List[str] = []
        
        if explicit_pre_selected_property_ids:
            found_ids_set = set()
            for prop_id_to_find in explicit_pre_selected_property_ids:
                found = False
                for i, p in enumerate(self.properties):
                    if p.property_id == prop_id_to_find:
                        self.pre_selected_property_indices.append(i)
                        self.pre_selected_property_ids.append(p.property_id)
                        found_ids_set.add(prop_id_to_find)
                        found = True
                        break
                if not found:
                    pass
            
            if len(self.pre_selected_property_ids) != len(explicit_pre_selected_property_ids):
                not_found_ids = set(explicit_pre_selected_property_ids) - found_ids_set
                if not_found_ids:
                    pass


    def solve(self, num_solutions: int = 1) -> List[List[LeasePlan]]:
        solutions = []
        excluded_property_patterns: List[List[int]] = []

        M = sum(p.total_units * p.rate_per_day * self.days for p in self.properties) + 1

        if not self.active_groups and self.properties and self.min_properties_per_group > 0:
            st.warning(f"Warning: No groups have at least {self.min_properties_per_group} properties. The 'min properties per group' constraint will effectively be ignored for all groups as it cannot be met.")


        for sol_idx in range(num_solutions):
            start_solve_time = time.time()
            prob = LpProblem(f"Lease_Optimization_Solution_{sol_idx+1}", LpMinimize)

            x = LpVariable.dicts("units_leased", range(len(self.properties)), lowBound=0, cat=LpInteger)
            z = LpVariable.dicts("property_selected", range(len(self.properties)), cat=LpBinary)
            y = LpVariable.dicts("full_property", range(len(self.properties)), cat=LpBinary)

            for i, p in enumerate(self.properties):
                prob += x[i] <= p.total_units * z[i], f"Units_max_limit_for_selected_prop_{i}"
                prob += x[i] >= z[i], f"Units_min_limit_for_selected_prop_{i}"
                prob += x[i] >= p.total_units * y[i], f"Full_Prop_If_Units_Full_1_{i}"
                prob += x[i] <= p.total_units * y[i] + (1 - y[i]) * (p.total_units - 1 + 0.5), f"Full_Prop_If_Units_Full_2_{i}"
                prob += (p.total_units - x[i]) <= M * (1 - y[i]), f"Full_Prop_If_Units_Not_Full_Y_is_0_{i}"
                prob += (x[i] - p.total_units) <= M * y[i], f"Full_Prop_If_Units_Full_Y_is_1_{i}"
                prob += y[i] <= z[i], f"Full_Prop_implies_Selected_{i}"

            for i in self.pre_selected_property_indices:
                p_pre_selected = self.properties[i]
                prob += z[i] == 1, f"PreSelected_Prop_{p_pre_selected.property_id}_Selected"
                prob += x[i] == p_pre_selected.total_units, f"PreSelected_Prop_{p_pre_selected.property_id}_FullUnits"
                prob += y[i] == 1, f"PreSelected_Prop_{p_pre_selected.property_id}_FullLease"

            if self.min_properties_per_group > 0:
                for group_char in self.active_groups:
                    prop_indices_in_group = self.properties_by_group[group_char]
                    if prop_indices_in_group:
                        prob += lpSum(z[i] for i in prop_indices_in_group) >= self.min_properties_per_group, f"Min_{self.min_properties_per_group}_Properties_From_Group_{group_char}"

            total_cost_expr = lpSum(
                x[i] * self.properties[i].rate_per_day * self.days
                for i in range(len(self.properties))
            )
            total_units_leased_expr = lpSum(x[i] for i in range(len(self.properties)))

            prob += total_cost_expr >= self.budget, "Min_Budget_Constraint"
            prob += total_cost_expr <= self.budget * (1 + self.budget_tolerance), "Max_Budget_Constraint"

            prob += total_units_leased_expr >= self.units_required * (1 - self.unit_tolerance), "Min_Units_Constraint"
            prob += total_units_leased_expr <= self.units_required * (1 + self.unit_tolerance), "Max_Units_Constraint"

            penalty_for_partial_leases_term = lpSum(
                (z[i] - y[i]) * self.properties[i].total_units * self.full_property_penalty_per_unit
                for i in range(len(self.properties))
            )

            occupancy_penalty_term = lpSum(
                x[i] * self.occupancy_penalty_per_unit_per_day * self.days
                for i, p in enumerate(self.properties)
                if p.occupancy_rating > self.occupancy_rating_threshold
            )
            
            prob += total_cost_expr + penalty_for_partial_leases_term + occupancy_penalty_term, "Minimize_Total_Cost_and_Penalties"

            for prev_pattern_idx, prev_pattern in enumerate(excluded_property_patterns):
                non_pre_selected_indices = [i for i in range(len(self.properties)) if i not in self.pre_selected_property_indices]
                
                if non_pre_selected_indices:
                    prob += lpSum(
                        z[i] * (1 - prev_pattern[i]) + (1 - z[i]) * prev_pattern[i]
                        for i in non_pre_selected_indices
                    ) >= 1, f"Exclude_Prev_Solution_Pattern_Sol{sol_idx}_Prev{prev_pattern_idx}"
                else:
                    if sol_idx > 0:
                        st.warning(f"Warning: All properties are pre-selected. Only one unique selection pattern exists. Stopping solution search.")
                        return solutions
                    
            solver_params = {}
            if self.solver_time_limit_seconds is not None:
                solver_params['timeLimit'] = self.solver_time_limit_seconds
            if self.solver_mip_gap is not None:
                solver_params['gapRel'] = self.solver_mip_gap
            
            solver = PULP_CBC_CMD(**solver_params)
            result = prob.solve(solver)
            
            elapsed_time = time.time() - start_solve_time

            if prob.status == LpStatusOptimal:
                st.success(f"Plan {sol_idx+1} found an optimal solution in {elapsed_time:.2f} seconds.")
            elif prob.status == LpStatusNotSolved and value(prob.objective) is not None:
                st.info(f"Plan {sol_idx+1} found a feasible solution (not proven optimal) in {elapsed_time:.2f} seconds. Status: {LpStatus[prob.status]}")
            elif prob.status == LpStatusUserStopped and value(prob.objective) is not None:
                st.info(f"Plan {sol_idx+1} stopped by user (e.g., time limit) with a feasible solution in {elapsed_time:.2f} seconds. Status: {LpStatus[prob.status]}")
            elif prob.status == LpStatusInfeasible:
                st.error(f"Plan {sol_idx+1}: Problem is Infeasible. No solution found. (Elapsed: {elapsed_time:.2f}s)")
                break
            else:
                st.error(f"Plan {sol_idx+1}: Solver did not find a solution. Status: {LpStatus[prob.status]} (Elapsed: {elapsed_time:.2f}s)")
                break

            if value(prob.objective) is None:
                st.error(f"No feasible solution found for Plan {sol_idx+1}. Stopping further solution search.")
                break

            current_lease_plan: List[LeasePlan] = []
            current_property_pattern: List[int] = [0] * len(self.properties)

            for i, p in enumerate(self.properties):
                units_val = round(value(x[i]) or 0)
                selected_val = round(value(z[i]) or 0)
                full_prop_val = round(value(y[i]) or 0)
                
                is_pre_selected = (i in self.pre_selected_property_indices)

                if selected_val == 1 and units_val > 0:
                    current_lease_plan.append(
                        LeasePlan(
                            property_id=p.property_id,
                            units=units_val,
                            rate_per_day=p.rate_per_day,
                            occupancy_rating_at_selection=p.occupancy_rating,
                            full_property=bool(full_prop_val),
                            pre_selected=is_pre_selected
                        )
                    )
                current_property_pattern[i] = selected_val

            if current_lease_plan:
                solutions.append(current_lease_plan)
                excluded_property_patterns.append(current_property_pattern)
            else:
                st.warning(f"No properties selected in Plan {sol_idx+1}. Stopping further solution search.")
                break

        return solutions

# --- Data Loading/Generation Functions ---
def load_properties_from_csv(file_obj) -> List[Property]:
    """
    Loads property details from a CSV file-like object.

    Expected CSV columns: property_id,total_units,rate_per_day,occupancy_rating,group
    """
    properties = []
    
    reader = csv.DictReader(file_obj)
    line_num = 1
    for row in reader:
        line_num += 1
        try:
            property_id = row['property_id']
            total_units = int(row['total_units'])
            rate_per_day = float(row['rate_per_day'])
            occupancy_rating = int(row['occupancy_rating'])
            group = row['group']

            if not (1 <= occupancy_rating <= 10):
                raise ValueError("occupancy_rating must be between 1 and 10.")
            if not (isinstance(group, str) and len(group) == 1 and group.isalpha() and group.isupper()):
                raise ValueError("group must be a single uppercase letter (A-Z).")
            if total_units <= 0 or rate_per_day <= 0:
                raise ValueError("total_units and rate_per_day must be positive.")

            properties.append(Property(property_id, total_units, rate_per_day, occupancy_rating, group))
        except KeyError as ke:
            st.error(f"Error: Missing column '{ke}' in CSV row {line_num}. Row skipped.")
        except ValueError as ve:
            st.error(f"Error: Invalid data type or value in CSV row {line_num} for property '{row.get('property_id', 'N/A')}': {ve}. Row skipped.")
        except Exception as e:
            st.error(f"An unexpected error occurred parsing CSV row {line_num}: {e}. Row skipped.")
    
    return properties

def generate_random_properties(num_properties: int) -> List[Property]:
    """
    Generates a list of Property objects with random values, including group and occupancy rating.
    """
    generated_properties = []
    groups = list(string.ascii_uppercase)
    for i in range(num_properties):
        property_id = f"R_P_{i+1}"
        total_units = random.randint(3, 20)
        rate_per_day = random.randint(500, 2000)
        occupancy_rating = random.randint(1, 10)
        assigned_group = random.choice(groups)
        generated_properties.append(Property(property_id, total_units, rate_per_day, occupancy_rating, assigned_group))
    return generated_properties

def create_sample_properties_csv_content() -> str:
    """
    Generates a sample CSV content string for property details.
    """
    sample_data = [
        {"property_id": "Sample_P1", "total_units": 8, "rate_per_day": 1200.0, "occupancy_rating": 7, "group": "A"},
        {"property_id": "Sample_P2", "total_units": 15, "rate_per_day": 950.0, "occupancy_rating": 4, "group": "B"},
        {"property_id": "Sample_P3", "total_units": 5, "rate_per_day": 1500.0, "occupancy_rating": 9, "group": "C"},
        {"property_id": "Sample_P4", "total_units": 10, "rate_per_day": 700.0, "occupancy_rating": 3, "group": "D"},
        {"property_id": "Sample_P5", "total_units": 7, "rate_per_day": 1100.0, "occupancy_rating": 8, "group": "A"},
    ]
    
    output = io.StringIO()
    fieldnames = ["property_id", "total_units", "rate_per_day", "occupancy_rating", "group"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(sample_data)
    
    return output.getvalue()

# --- Refactored Display Functions for Streamlit ---

def display_optimizer_configuration_content_st(optimizer: LeaseOptimizerMILP):
    """Prints the configuration settings of the optimizer using Streamlit, formatted as a card."""
    st.markdown("### ⚙️ Optimizer Configuration")

    # Client Request
    st.markdown("#### Client Request")
    st.info(f"**Target Units:** {optimizer.units_required}  \n"
            f"**Minimum Cost Target:** ₹{optimizer.budget:,.0f}  \n"
            f"**Lease Duration:** {optimizer.days} days ({optimizer.start_date.strftime('%b %d, %Y')} to {optimizer.end_date.strftime('%b %d, %Y')})")
    
    st.markdown("#### Constraints & Penalties")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.write(f"**Budget Upper Tolerance:** {optimizer.budget_tolerance * 100:.0f}%")
        st.write(f"**Units Tolerance:** {optimizer.unit_tolerance * 100:.0f}%")
        st.write(f"**Min Properties per Group:** {optimizer.min_properties_per_group}")
    with col_c2:
        st.write(f"**Partial Lease Penalty:** ₹{optimizer.full_property_penalty_per_unit:.2f} / unit")
        st.write(f"**High Occupancy Penalty:** ₹{optimizer.occupancy_penalty_per_unit_per_day:.2f} / unit / day")
        st.write(f"**Occupancy Threshold (for penalty):** > {optimizer.occupancy_rating_threshold}")

    st.markdown("#### Solver Settings")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write(f"**Time Limit (per plan):** {optimizer.solver_time_limit_seconds} seconds")
    with col_s2:
        st.write(f"**MIP Gap Tolerance (per plan):** {optimizer.solver_mip_gap*100:.1f}%")

    if optimizer.pre_selected_property_ids:
        st.markdown("#### Pre-selected Properties")
        st.caption(f"**Explicitly Included:** {', '.join(optimizer.pre_selected_property_ids)}")
    else:
        st.caption("No properties were explicitly pre-selected.")


def calculate_plan_metrics(lease_plan: List[LeasePlan], all_properties_data: List[Property], optimizer: LeaseOptimizerMILP):
    """Calculates key metrics for a given lease plan."""
    total_plan_actual_cost = 0
    total_plan_units = 0
    total_penalty_for_partial_leases = 0
    total_penalty_for_high_occupancy = 0
    selected_groups: Dict[str, int] = {}
    total_properties_selected_in_plan = 0

    for lease in lease_plan:
        prop_obj_orig = next((p for p in all_properties_data if p.property_id == lease.property_id), None)
        
        if prop_obj_orig:
            current_rental_for_units = lease.units * lease.rate_per_day * optimizer.days
            total_plan_actual_cost += current_rental_for_units
            total_plan_units += lease.units
            total_properties_selected_in_plan += 1

            if not lease.full_property and not lease.pre_selected:
                total_penalty_for_partial_leases += prop_obj_orig.total_units * optimizer.full_property_penalty_per_unit
            
            if lease.occupancy_rating_at_selection > optimizer.occupancy_rating_threshold:
                total_penalty_for_high_occupancy += lease.units * optimizer.occupancy_penalty_per_unit_per_day * optimizer.days

            selected_groups[prop_obj_orig.group] = selected_groups.get(prop_obj_orig.group, 0) + 1
        
    total_objective_value = total_plan_actual_cost + total_penalty_for_partial_leases + total_penalty_for_high_occupancy

    min_budget_allowed = optimizer.budget
    max_budget_allowed = optimizer.budget * (1 + optimizer.budget_tolerance)
    budget_status = (min_budget_allowed <= total_plan_actual_cost <= max_budget_allowed)

    min_units_allowed = optimizer.units_required * (1 - optimizer.unit_tolerance)
    max_units_allowed = optimizer.units_required * (1 + optimizer.unit_tolerance)
    units_status = (min_units_allowed <= total_plan_units <= max_units_allowed)

    group_constraint_met_overall = True
    if optimizer.min_properties_per_group > 0:
        for group_char in optimizer.active_groups:
            count = selected_groups.get(group_char, 0)
            if count < optimizer.min_properties_per_group:
                group_constraint_met_overall = False
                break

    return {
        "total_plan_actual_cost": total_plan_actual_cost,
        "total_plan_units": total_plan_units,
        "total_penalty_for_partial_leases": total_penalty_for_partial_leases,
        "total_penalty_for_high_occupancy": total_penalty_for_high_occupancy,
        "total_objective_value": total_objective_value,
        "total_properties_selected": total_properties_selected_in_plan,
        "budget_status": budget_status,
        "units_status": units_status,
        "group_constraint_met_overall": group_constraint_met_overall,
        "selected_groups": selected_groups,
        "min_budget_allowed": min_budget_allowed,
        "max_budget_allowed": max_budget_allowed,
        "min_units_allowed": min_units_allowed,
        "max_units_allowed": max_units_allowed,
    }


def print_plan_details_st(plan_idx: int, lease_plan: List[LeasePlan], all_properties_data: List[Property], optimizer: LeaseOptimizerMILP):
    """Prints detailed information for a single optimized lease plan within an expander."""
    
    metrics = calculate_plan_metrics(lease_plan, all_properties_data, optimizer)

    st.markdown(f"#### Properties in Plan {plan_idx}")
    
    property_details_for_display = []
    for lease in lease_plan:
        prop_obj_orig = next((p for p in all_properties_data if p.property_id == lease.property_id), None)
        if prop_obj_orig:
            current_rental_for_units = lease.units * lease.rate_per_day * optimizer.days
            pre_selected_indicator = "(PRE-SELECTED)" if lease.pre_selected else ""
            property_details_for_display.append({
                "Property ID": lease.property_id,
                "Group": prop_obj_orig.group,
                "Occupancy Rating": lease.occupancy_rating_at_selection,
                "Units Leased": lease.units,
                "Rate/Unit/Day": f"₹{lease.rate_per_day:.2f}",
                "Lease Type": "FULL PROPERTY" if lease.full_property else "PARTIAL LEASE",
                "Cost for Property": f"₹{int(current_rental_for_units):,}",
                "Status": pre_selected_indicator
            })
    
    if property_details_for_display:
        st.dataframe(property_details_for_display, use_container_width=True)
    else:
        st.info("No properties selected in this plan.")


    st.markdown(f"#### Summary Metrics for Plan {plan_idx}")
    st.write(f"  Total Plan Actual Rental Cost: ₹{int(metrics['total_plan_actual_cost']):,}")
    st.write(f"  Total Plan Units Leased: {metrics['total_plan_units']}")
    st.write(f"  Total Penalty for Partial Leases: ₹{int(metrics['total_penalty_for_partial_leases']):,}")
    st.write(f"  Total Penalty for High Occupancy Ratings: ₹{int(metrics['total_penalty_for_high_occupancy']):,}")
    st.markdown(f"** TOTAL OBJECTIVE VALUE: ₹{int(metrics['total_objective_value']):,}**")
    st.write(f"  Total Properties Selected: {metrics['total_properties_selected']}")

    st.write(f"  Minimum Cost Target: ₹{optimizer.budget:,.0f} (Allowed Range: ₹{metrics['min_budget_allowed']:,.0f} - ₹{metrics['max_budget_allowed']:,.0f}) {'✅' if metrics['budget_status'] else '❌'}")
    st.write(f"  Units Target: {optimizer.units_required} (Range: {metrics['min_units_allowed']:.1f} - {metrics['max_units_allowed']:.1f}) {'✅' if metrics['units_status'] else '❌'}")

    st.markdown("#### Selected Properties by Group:")
    group_counts_all_data = {p.group: 0 for p in all_properties_data} # Initialize with all groups from data
    for p in all_properties_data:
        group_counts_all_data[p.group] = group_counts_all_data.get(p.group, 0) + 1

    all_groups_present = sorted(list(set(group_counts_all_data.keys()).union(metrics['selected_groups'].keys())))
    
    group_summary_lines = []
    for group_char in all_groups_present:
        count = metrics['selected_groups'].get(group_char, 0)
        
        if group_char in optimizer.active_groups and optimizer.min_properties_per_group > 0: 
            if count >= optimizer.min_properties_per_group:
                status_icon = "✅"
                group_summary_lines.append(f"    Group '{group_char}': {count} properties selected (Required >= {optimizer.min_properties_per_group}) {status_icon}")
            else:
                status_icon = "❌"
                group_summary_lines.append(f"    Group '{group_char}': {count} properties selected (Required >= {optimizer.min_properties_per_group}) {status_icon}")
        elif count > 0:
            status_icon = "➖"
            group_summary_lines.append(f"    Group '{group_char}': {count} properties selected {status_icon} (Constraint not applicable)")
        # else: Don't print groups with 0 selected properties if no constraint applies

    for line in group_summary_lines:
        st.write(line)

    if metrics['group_constraint_met_overall']:
        st.success("✅ All required group constraints met for this plan.")
    else:
        st.error("❌ Some required group constraints NOT met for this plan.")


# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use a wide layout
st.title("Rental Lease Optimization App")
st.write("Configure parameters and find optimal lease plans based on your criteria.")

# --- Input Widgets ---
st.sidebar.header("Data Source & Lease Period")
data_source_option = st.sidebar.radio("Select Property Data Source:", ("Generate Random Properties", "Upload CSV File"))

properties_data: List[Property] = []
if data_source_option == "Generate Random Properties":
    num_properties = st.sidebar.slider("Number of Random Properties", 50, 1000, 300)
    properties_data = generate_random_properties(num_properties)
    st.sidebar.info(f"Generated {len(properties_data)} random properties.")
else: # Upload CSV
    csv_sample_data = create_sample_properties_csv_content()
    st.sidebar.download_button(
        label="Download Sample CSV",
        data=csv_sample_data,
        file_name="sample_properties.csv",
        mime="text/csv",
        help="Download a sample CSV file to see the expected format (property_id,total_units,rate_per_day,occupancy_rating,group)."
    )

    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv", help="Expected columns: property_id,total_units,rate_per_day,occupancy_rating,group")
    if uploaded_file is not None:
        string_data = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        try:
            properties_data = load_properties_from_csv(string_data)
            if properties_data:
                st.sidebar.success(f"Loaded {len(properties_data)} properties from CSV.")
            else:
                st.sidebar.error("CSV loaded, but no valid properties parsed. Check error messages above.")
        except Exception as e:
            st.sidebar.error(f"Error processing CSV: {e}. Please check file format and try again.")
            properties_data = [] # Clear data if error

if not properties_data:
    st.error("No properties loaded or generated. Please adjust data source settings to proceed.")
    st.stop() # Stop execution flow if no properties are available

st.sidebar.header("Client Requirements")
col1, col2 = st.sidebar.columns(2)
start_date_val = col1.date_input("Start Date", date(2025, 7, 1))
end_date_val = col2.date_input("End Date", date(2025, 7, 10))

lease_days = (end_date_val - start_date_val).days + 1
if lease_days <= 0:
    st.sidebar.error("End Date must be after Start Date.")
    st.stop()

# Estimate initial units/budget based on loaded/generated data
total_potential_units = sum(p.total_units for p in properties_data)
avg_rate_per_day = sum(p.rate_per_day for p in properties_data) / len(properties_data) if properties_data else 1000
estimated_units = max(1, int(total_potential_units * 0.4))
estimated_budget_val = int(estimated_units * avg_rate_per_day * lease_days * 1.05)

units_required = st.sidebar.number_input("Target Units Required", value=estimated_units, min_value=1)
budget = st.sidebar.number_input("Minimum Cost Target (₹)", value=float(estimated_budget_val), min_value=1.0)

st.sidebar.header("Optimizer Settings")
budget_tolerance_percent = st.sidebar.slider("Budget Upper Tolerance (%)", 0.0, 50.0, 10.0, help="Allows final cost to be up to this percentage above the Minimum Cost Target.")
unit_tolerance_percent = st.sidebar.slider("Units Tolerance (%)", 0.0, 50.0, 15.0, help="Allows final units to be within +/- this percentage of Target Units Required.")
full_property_penalty_per_unit = st.sidebar.number_input("Partial Lease Penalty (₹/unit)", value=50.0, min_value=0.0, help="Penalty for each unit short of a full property lease, if property is selected.")
min_properties_per_group = st.sidebar.slider("Min Properties per Group", 0, 5, 2, help="Minimum number of properties to select from each group. Set to 0 to disable this constraint.")
occupancy_penalty_per_unit_per_day = st.sidebar.number_input("High Occupancy Penalty (₹/unit/day)", value=150.0, min_value=0.0, help="Penalty applied per unit per day for properties whose Occupancy Rating is above the threshold.")
occupancy_rating_threshold = st.sidebar.slider("Occupancy Rating Threshold (for penalty)", 1, 10, 6, help="Properties with an Occupancy Rating *greater than* this value will incur a penalty.")

st.sidebar.header("Solver Settings")
solver_time_limit_seconds = st.sidebar.number_input("Solver Time Limit (seconds/plan)", value=10, min_value=1, help="Maximum time the solver will run for each individual plan before returning the best solution found so far.")
solver_mip_gap = st.sidebar.slider("Solver MIP Gap (%)", 0.0, 10.0, 5.0, help="Solver stops when the solution is guaranteed to be within this percentage of the true optimal value.") / 100.0 # Convert to fraction

st.sidebar.header("Pre-selected Properties")
available_property_ids = [p.property_id for p in properties_data]
pre_selected_ids = st.sidebar.multiselect("Select Properties to ALWAYS Include:", available_property_ids, help="These properties will be forced into the solution, taking all their units.")

num_solutions = st.sidebar.slider("Number of Solutions to Find", 1, 5, 3, help="The optimizer will attempt to find this many distinct optimal/near-optimal solutions.")


# --- Run Optimization Button ---
if st.button("Run Optimization"):
    st.header("Optimization Results")
    
    valid_pre_selected_ids = []
    all_property_ids_set = {p.property_id for p in properties_data}
    for pid in pre_selected_ids:
        if pid in all_property_ids_set:
            valid_pre_selected_ids.append(pid)
        else:
            st.warning(f"Pre-selected property ID '{pid}' not found in the available properties. It will be ignored.")
    
    if min_properties_per_group > 0:
        group_counts_all_data = {}
        for p in properties_data:
            group_counts_all_data[p.group] = group_counts_all_data.get(p.group, 0) + 1
        
        insufficient_groups_for_constraint = [group for group, count in group_counts_all_data.items() if count < min_properties_per_group]
        if insufficient_groups_for_constraint:
            st.warning(f"Warning: The following groups have fewer than {min_properties_per_group} properties: {', '.join(insufficient_groups_for_constraint)}. This might make the 'min properties per group' constraint infeasible.")


    optimizer = LeaseOptimizerMILP(
        properties=properties_data,
        units_required=units_required,
        budget=budget,
        start_date=start_date_val,
        end_date=end_date_val,
        budget_tolerance_percent=budget_tolerance_percent,
        unit_tolerance_percent=unit_tolerance_percent,
        full_property_penalty_per_unit=full_property_penalty_per_unit,
        min_properties_per_group=min_properties_per_group,
        occupancy_penalty_per_unit_per_day=occupancy_penalty_per_unit_per_day,
        occupancy_rating_threshold=occupancy_rating_threshold,
        solver_time_limit_seconds=solver_time_limit_seconds,
        solver_mip_gap=solver_mip_gap,
        explicit_pre_selected_property_ids=valid_pre_selected_ids
    )

    # Display Optimizer Configuration and Solver Time side-by-side
    st.markdown("### Optimization Run Summary")
    col_config, col_time = st.columns([3, 1]) # Adjust column ratios for better balance
    
    with col_config:
        with st.expander("View Optimizer Configuration", expanded=True): # Renamed expander title
            display_optimizer_configuration_content_st(optimizer)

    with col_time:
        st.markdown("##### Solver Run Time")
        total_solve_start_time = time.time()
        plans = optimizer.solve(num_solutions=num_solutions)
        total_solve_end_time = time.time()
        total_elapsed_time = total_solve_end_time - total_solve_start_time
        st.metric(label="Total time to find all plans", value=f"{total_elapsed_time:.2f} seconds")


    if not plans:
        st.error("❌ No feasible plans found based on the given constraints.")
    else:
        st.markdown("---")
        st.subheader("Summary of Generated Plans")
        
        cols_per_row = 3 # Adjust as needed
        
        for i in range(0, len(plans), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if (i + j) < len(plans):
                    plan_idx = i + j + 1
                    plan = plans[i+j]
                    
                    with cols[j]:
                        st.markdown(f"#### Plan {plan_idx}")
                        metrics = calculate_plan_metrics(plan, properties_data, optimizer)

                        st.write(f"**Cost:** ₹{int(metrics['total_plan_actual_cost']):,}")
                        st.write(f"**Units:** {metrics['total_plan_units']}")
                        st.write(f"**Total Penalty:** ₹{int(metrics['total_penalty_for_partial_leases'] + metrics['total_penalty_for_high_occupancy']):,}")
                        st.markdown(f"**Objective Value:** ₹{int(metrics['total_objective_value']):,}")
                        
                        st.write(f"Budget Status: {'✅' if metrics['budget_status'] else '❌'}")
                        st.write(f"Units Status: {'✅' if metrics['units_status'] else '❌'}")
                        st.write(f"Group Status: {'✅' if metrics['group_constraint_met_overall'] else '❌'}")
                        
                        # Add a button/expander to view details
                        with st.expander(f"View Details for Plan {plan_idx}"):
                            print_plan_details_st(plan_idx, plan, properties_data, optimizer)
            st.markdown("---") # Separator between rows of summaries