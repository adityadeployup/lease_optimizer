from pulp import (
    LpProblem, LpVariable, LpMinimize, LpBinary, LpInteger, LpContinuous,
    lpSum, LpStatusOptimal, LpStatus, value, PULP_CBC_CMD
)
from datetime import date
from typing import List, Dict, Optional, Tuple
import string
import time
import random
import csv
import io
import streamlit as st
import json

class ProductFullDetail:
    """Represents full product-code-specific details including base rate, default discount, and max applicable discount."""
    def __init__(self, product_code: str, base_rate_per_day: float, default_card_rate_discount: float, max_discount_applicable: float):
        self.product_code = product_code
        self.base_rate_per_day = base_rate_per_day
        self.default_card_rate_discount = default_card_rate_discount
        self.max_discount_applicable = max_discount_applicable

class Property:
    """Represents a single rental property with its details, linking to ProductFullDetail."""
    def __init__(self, property_id: str, total_units: int, occupancy_rating: int, group: str, city: str, product_code: str):
        self.property_id = property_id
        self.total_units = total_units
        self.occupancy_rating = occupancy_rating
        self.group = group
        self.city = city
        self.product_code = product_code # Link to ProductFullDetail

class LeasePlan:
    """Represents a leased portion of a property in a solution."""
    def __init__(self, property_id: str, units: int, effective_rate_per_day: float, occupancy_rating_at_selection: int, full_property: bool, pre_selected: bool = False, original_rate_per_day: float = 0.0, card_rate_discount_at_selection: float = 0.0):
        self.property_id = property_id
        self.units = units
        self.effective_rate_per_day = effective_rate_per_day # Rate after applying discount
        self.occupancy_rating_at_selection = occupancy_rating_at_selection
        self.full_property = full_property
        self.pre_selected = pre_selected # Indicates if this property was pre-selected
        self.original_rate_per_day = original_rate_per_day # This is now the base_rate_per_day from ProductFullDetail
        self.card_rate_discount_at_selection = card_rate_discount_at_selection # This is the default_card_rate_discount from ProductFullDetail initially, then adjusted

class LeaseOptimizerMILP:
    """
    Optimizes a lease plan for rental properties given client requirements
    and property constraints using Mixed-Integer Linear Programming (MILP).
    """
    def __init__(
        self,
        properties: List[Property],
        product_full_details: List[ProductFullDetail], # New: Consolidated product data
        units_required: int,
        budget: float,
        start_date: date,
        end_date: date,
        budget_tolerance_percent: float = 5.0,
        unit_tolerance_percent: float = 10.0,
        full_property_penalty_per_unit: float = 1.0,
        min_properties_per_group: int = 2,
        occupancy_penalty_per_unit_per_day: float = 100.0,
        occupancy_rating_threshold: int = 6,
        solver_time_limit_seconds: Optional[int] = None,
        solver_mip_gap: Optional[float] = None,
        explicit_pre_selected_property_ids: Optional[List[str]] = None,
        min_units_per_city: Optional[Dict[str, int]] = None,
        min_units_per_product_code: Optional[Dict[str, int]] = None,
        default_max_discount_if_not_specified: float = 40.0 # This default applies if a product_code is NOT in product_full_details
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

        self.min_units_per_city = min_units_per_city if min_units_per_city else {}
        self.min_units_per_product_code = min_units_per_product_code if min_units_per_product_code else {}
        
        # New: Map product_full_details for quick lookup of base rates, default discounts, and max caps
        self.product_full_details_map: Dict[str, ProductFullDetail] = {pfd.product_code: pfd for pfd in product_full_details}
        if not self.product_full_details_map:
            st.error("Error: No Product Details data provided. Please generate or upload product details.")
            st.stop()

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
            for prop_id_to_find in explicit_pre_selected_property_ids:
                for i, p in enumerate(self.properties):
                    if p.property_id == prop_id_to_find:
                        self.pre_selected_property_indices.append(i)
                        self.pre_selected_property_ids.append(p.property_id)
                        break

        self.default_max_discount_if_not_specified = default_max_discount_if_not_specified


    def solve(self, num_solutions: int = 1) -> List[List[LeasePlan]]:
        solutions = []
        excluded_property_patterns: List[List[int]] = []

        # A large enough number for Big M method. Sum of all possible units * max rate * days + 1
        max_rate = 0.0
        if self.product_full_details_map:
            max_rate = max(pfd.base_rate_per_day for pfd in self.product_full_details_map.values())
        if max_rate == 0: max_rate = 1
            
        M = sum(p.total_units for p in self.properties) * max_rate * self.days + 1


        if not self.active_groups and self.properties and self.min_properties_per_group > 0:
            st.warning(f"Warning: No groups have at least {self.min_properties_per_group} properties. The 'min properties per group' constraint will effectively be ignored for all groups as it cannot be met.")


        for sol_idx in range(num_solutions):
            start_solve_time = time.time()
            prob = LpProblem(f"Lease_Optimization_Solution_{sol_idx+1}", LpMinimize)

            # Decision Variables
            x = LpVariable.dicts("units_leased", range(len(self.properties)), lowBound=0, cat=LpInteger)
            z = LpVariable.dicts("property_selected", range(len(self.properties)), cat=LpBinary)
            y = LpVariable.dicts("full_property", range(len(self.properties)), cat=LpBinary)

            # Constraints for each property
            for i, p in enumerate(self.properties):
                # Ensure product_code exists in product_full_details_map
                if p.product_code not in self.product_full_details_map:
                    st.error(f"Error: Property {p.property_id} has unknown product_code '{p.product_code}'. Please ensure all product codes in properties data exist in Product Details data.")
                    return [] # Abort solution process

                product_detail = self.product_full_details_map[p.product_code]

                prob += x[i] <= p.total_units * z[i], f"Units_max_limit_for_selected_prop_{i}"
                prob += x[i] >= z[i], f"Units_min_limit_for_selected_prop_{i}"

                prob += p.total_units - x[i] <= M * (1 - y[i]), f"Full_Prop_If_Units_Full_Part1_{i}"
                prob += x[i] - p.total_units <= M * (1 - y[i]), f"Full_Prop_If_Units_Full_Part2_{i}"
                
                prob += x[i] >= p.total_units * y[i], f"Units_Full_If_Full_Prop_1_{i}"
                prob += x[i] <= p.total_units * y[i] + (p.total_units -1) * (1-y[i]), f"Units_Full_If_Full_Prop_2_{i}"

                prob += y[i] <= z[i], f"Full_Prop_implies_Selected_{i}"

                # Constraint: Default discount for this product code must be <= max allowed for that product code
                # Max discount for this product code is now directly from ProductFullDetail
                prob += product_detail.default_card_rate_discount * z[i] <= product_detail.max_discount_applicable * z[i], f"Max_Discount_ProductCode_{p.property_id}"


            # Constraints for pre-selected properties
            for i in self.pre_selected_property_indices:
                p_pre_selected = self.properties[i]
                prob += z[i] == 1, f"PreSelected_Prop_{p_pre_selected.property_id}_Selected"
                prob += x[i] == p_pre_selected.total_units, f"PreSelected_Prop_{p_pre_selected.property_id}_FullUnits"
                prob += y[i] == 1, f"PreSelected_Prop_{p_pre_selected.property_id}_FullLease"


            # Minimum properties per group constraint
            if self.min_properties_per_group > 0:
                for group_char in self.active_groups:
                    prop_indices_in_group = self.properties_by_group[group_char]
                    if prop_indices_in_group:
                        prob += lpSum(z[i] for i in prop_indices_in_group) >= self.min_properties_per_group, f"Min_{self.min_properties_per_group}_Properties_From_Group_{group_char}"

            # Min units per City
            if self.min_units_per_city:
                properties_by_city: Dict[str, List[int]] = {}
                for i, p in enumerate(self.properties):
                    properties_by_city.setdefault(p.city, []).append(i)
                
                for city, min_units in self.min_units_per_city.items():
                    if city in properties_by_city:
                        prob += lpSum(x[i] for i in properties_by_city[city]) >= min_units, f"Min_Units_City_{city}"
                    else:
                        st.warning(f"Warning: No properties found for city '{city}' specified in 'Min Units per City' constraint. This constraint might cause infeasibility if not addressed.")

            # Min units per Product Code
            if self.min_units_per_product_code:
                properties_by_product_code: Dict[str, List[int]] = {}
                for i, p in enumerate(self.properties):
                    properties_by_product_code.setdefault(p.product_code, []).append(i)

                for product_code, min_units in self.min_units_per_product_code.items():
                    if product_code in properties_by_product_code:
                        prob += lpSum(x[i] for i in properties_by_product_code[product_code]) >= min_units, f"Min_Units_ProductCode_{product_code}"
                    else:
                        st.warning(f"Warning: No properties found for product code '{product_code}' specified in 'Min Units per Product Code' constraint. This constraint might cause infeasibility if not addressed.")


            # Calculate total cost and total units leased
            total_cost_expr = lpSum(
                x[i] * self.product_full_details_map[self.properties[i].product_code].base_rate_per_day * (1 - self.product_full_details_map[self.properties[i].product_code].default_card_rate_discount / 100.0) * self.days
                for i in range(len(self.properties))
            )
            total_units_leased_expr = lpSum(x[i] for i in range(len(self.properties)))

            # Budget constraints
            prob += total_cost_expr >= self.budget, "Min_Budget_Constraint"
            prob += total_cost_expr <= self.budget * (1 + self.budget_tolerance), "Max_Budget_Constraint"

            # Units constraints
            prob += total_units_leased_expr >= self.units_required * (1 - self.unit_tolerance), "Min_Units_Constraint"
            prob += total_units_leased_expr <= self.units_required * (1 + self.unit_tolerance), "Max_Units_Constraint"

            # Penalties
            penalty_for_partial_leases_term = lpSum(
                (z[i] - y[i]) * self.properties[i].total_units * self.full_property_penalty_per_unit
                for i in range(len(self.properties))
            )

            occupancy_penalty_term = lpSum(
                x[i] * self.occupancy_penalty_per_unit_per_day * self.days
                for i, p in enumerate(self.properties)
                if p.occupancy_rating > self.occupancy_rating_threshold
            )
            
            # Objective Function
            prob += total_cost_expr + penalty_for_partial_leases_term + occupancy_penalty_term, "Minimize_Total_Cost_and_Penalties"

            # Exclude previously found solutions
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
                    
            # Solver parameters
            solver_params = {}
            if self.solver_time_limit_seconds is not None:
                solver_params['timeLimit'] = self.solver_time_limit_seconds
            if self.solver_mip_gap is not None:
                solver_params['gapRel'] = self.solver_mip_gap
            
            solver = PULP_CBC_CMD(**solver_params)
            result = prob.solve(solver)
            
            elapsed_time = time.time() - start_solve_time

            # Log solver status
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
                    product_detail = self.product_full_details_map[p.product_code]
                    effective_rate = product_detail.base_rate_per_day * (1 - product_detail.default_card_rate_discount / 100.0)
                    current_lease_plan.append(
                        LeasePlan(
                            property_id=p.property_id,
                            units=units_val,
                            effective_rate_per_day=effective_rate,
                            occupancy_rating_at_selection=p.occupancy_rating,
                            full_property=bool(full_prop_val),
                            pre_selected=is_pre_selected,
                            original_rate_per_day=product_detail.base_rate_per_day, # Base rate from ProductFullDetail
                            card_rate_discount_at_selection=product_detail.default_card_rate_discount # Default discount from ProductFullDetail
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
def load_product_full_details_from_csv(file_obj) -> List[ProductFullDetail]:
    """
    Loads full product details from a CSV file-like object.
    Expected CSV columns: product_code,base_rate_per_day,default_card_rate_discount,max_discount_applicable
    """
    product_full_details = []
    reader = csv.DictReader(file_obj)
    line_num = 1
    for row in reader:
        line_num += 1
        try:
            product_code = row['product_code'].strip().upper()
            base_rate_per_day = float(row['base_rate_per_day'])
            default_card_rate_discount = float(row['default_card_rate_discount'])
            max_discount_applicable = float(row['max_discount_applicable'])

            if base_rate_per_day <= 0: raise ValueError("base_rate_per_day must be positive.")
            if not (0.0 <= default_card_rate_discount <= 100.0): raise ValueError("default_card_rate_discount must be between 0.0 and 100.0.")
            if not (0.0 <= max_discount_applicable <= 100.0): raise ValueError("max_discount_applicable must be between 0.0 and 100.0.")
            if default_card_rate_discount > max_discount_applicable: raise ValueError(f"default_card_rate_discount ({default_card_rate_discount}%) cannot be greater than max_discount_applicable ({max_discount_applicable}%) for product_code '{product_code}'.")


            product_full_details.append(ProductFullDetail(product_code, base_rate_per_day, default_card_rate_discount, max_discount_applicable))
        except KeyError as ke:
            st.error(f"Error: Missing column '{ke}' in Product Details CSV row {line_num}. Row skipped.")
        except ValueError as ve:
            st.error(f"Error: Invalid data type or value in Product Details CSV row {line_num} for product_code '{row.get('product_code', 'N/A')}': {ve}. Row skipped.")
        except Exception as e:
            st.error(f"An unexpected error occurred parsing Product Details CSV row {line_num}: {e}. Row skipped.")
    return product_full_details

def generate_random_product_full_details(num_product_codes: int) -> List[ProductFullDetail]:
    """Generates random ProductFullDetail objects."""
    generated_details = []
    available_codes = ["COMMERCIAL", "RETAIL", "RESIDENTIAL", "WAREHOUSE", "OFFICE", "INDUSTRIAL", "FARM"]
    codes_to_use = random.sample(available_codes, min(num_product_codes, len(available_codes)))

    for code in codes_to_use:
        base_rate = random.randint(700, 2500)
        default_discount = random.uniform(0.0, 15.0)
        max_discount = random.uniform(default_discount, min(default_discount + 20, 100.0)) # Max is >= default, up to 100
        generated_details.append(ProductFullDetail(code, float(base_rate), round(default_discount, 1), round(max_discount, 1)))
    return generated_details

def create_sample_product_full_details_csv_content() -> str:
    """Generates sample CSV content for ProductFullDetail."""
    sample_data = [
        {"product_code": "COMMERCIAL", "base_rate_per_day": 1500.0, "default_card_rate_discount": 5.0, "max_discount_applicable": 20.0},
        {"product_code": "RETAIL", "base_rate_per_day": 1000.0, "default_card_rate_discount": 2.0, "max_discount_applicable": 15.0},
        {"product_code": "RESIDENTIAL", "base_rate_per_day": 1200.0, "default_card_rate_discount": 7.5, "max_discount_applicable": 10.0},
    ]
    output = io.StringIO()
    fieldnames = ["product_code", "base_rate_per_day", "default_card_rate_discount", "max_discount_applicable"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sample_data)
    return output.getvalue()


def load_properties_from_csv(file_obj) -> List[Property]:
    """
    Loads property details from a CSV file-like object.
    Expected CSV columns: property_id,total_units,occupancy_rating,group,city,product_code
    """
    properties = []
    reader = csv.DictReader(file_obj)
    line_num = 1
    for row in reader:
        line_num += 1
        try:
            property_id = row['property_id']
            total_units = int(row['total_units'])
            occupancy_rating = int(row['occupancy_rating'])
            group = row['group']
            city = row['city']
            product_code = row['product_code'].strip().upper()

            if not (1 <= occupancy_rating <= 10): raise ValueError("occupancy_rating must be between 1 and 10.")
            if not (isinstance(group, str) and len(group) == 1 and group.isalpha() and group.isupper()): raise ValueError("group must be a single uppercase letter (A-Z).")
            if total_units <= 0: raise ValueError("total_units must be positive.")

            properties.append(Property(property_id, total_units, occupancy_rating, group, city, product_code))
        except KeyError as ke:
            st.error(f"Error: Missing column '{ke}' in Properties CSV row {line_num}. Row skipped.")
        except ValueError as ve:
            st.error(f"Error: Invalid data type or value in Properties CSV row {line_num} for property '{row.get('property_id', 'N/A')}': {ve}. Row skipped.")
        except Exception as e:
            st.error(f"An unexpected error occurred parsing Properties CSV row {line_num}: {e}. Row skipped.")
    return properties

def generate_random_properties(num_properties: int, available_product_codes: List[str]) -> List[Property]:
    """
    Generates a list of Property objects with random values, using provided product codes.
    """
    generated_properties = []
    groups = list(string.ascii_uppercase)
    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jamshedpur"]

    if not available_product_codes:
        st.error("Cannot generate random properties without available product codes. Please generate/upload Product Details first.")
        return []

    for i in range(num_properties):
        property_id = f"R_P_{i+1}"
        total_units = random.randint(3, 20)
        occupancy_rating = random.randint(1, 10)
        assigned_group = random.choice(groups)
        assigned_city = random.choice(cities)
        assigned_product_code = random.choice(available_product_codes)
        generated_properties.append(Property(property_id, total_units, occupancy_rating, assigned_group, assigned_city, assigned_product_code))
    return generated_properties

def create_sample_properties_csv_content() -> str:
    """
    Generates a sample CSV content string for property details.
    Note: product_code in this sample must match codes in sample product_details.
    """
    sample_data = [
        {"property_id": "Sample_P1", "total_units": 8, "occupancy_rating": 7, "group": "A", "city": "Mumbai", "product_code": "COMMERCIAL"},
        {"property_id": "Sample_P2", "total_units": 15, "occupancy_rating": 4, "group": "B", "city": "Delhi", "product_code": "RETAIL"},
        {"property_id": "Sample_P3", "total_units": 5, "occupancy_rating": 9, "group": "C", "city": "Bengaluru", "product_code": "RESIDENTIAL"},
        {"property_id": "Sample_P4", "total_units": 10, "occupancy_rating": 3, "group": "D", "city": "Chennai", "product_code": "COMMERCIAL"},
        {"property_id": "Sample_P5", "total_units": 7, "occupancy_rating": 8, "group": "A", "city": "Mumbai", "product_code": "RESIDENTIAL"},
    ]
    
    output = io.StringIO()
    fieldnames = ["property_id", "total_units", "occupancy_rating", "group", "city", "product_code"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    
    writer.writeheader()
    writer.writerows(sample_data)
    
    return output.getvalue()


# --- Post-Optimization Discount Readjustment Logic ---
def post_process_discounts(
    initial_lease_plan: List[LeasePlan], 
    all_properties_data: List[Property], # For initial property details lookup
    optimizer: LeaseOptimizerMILP,
    initial_metrics: Dict # Initial metrics from the solver's raw output
) -> Tuple[List[LeasePlan], float, Dict[str, float]]:
    """
    Redistributes 'extra discount' if the actual cost is above the minimum budget target,
    up to the max_discount_applicable per product code.
    
    Returns:
        - adjusted_lease_plan (List[LeasePlan]): The lease plan with adjusted discounts.
        - infeasible_discount_value (float): Any remaining value that couldn't be distributed.
        - final_adjusted_discounts_per_property (Dict[str, float]): Final applied discount % for each property.
    """
    adjusted_lease_plan = [lp for lp in initial_lease_plan] # Create a mutable copy
    
    original_properties_map = {p.property_id: p for p in all_properties_data}

    # Calculate how much 'extra discount' needs to be applied to meet the minimum budget target
    remaining_discount_to_distribute = initial_metrics['total_plan_actual_cost'] - optimizer.budget
    
    if remaining_discount_to_distribute <= 0:
        final_adjusted_discounts_per_property = {
            lp.property_id: lp.card_rate_discount_at_selection for lp in adjusted_lease_plan
        }
        return adjusted_lease_plan, 0.0, final_adjusted_discounts_per_property

    properties_for_adjustment_info = []
    for lp in adjusted_lease_plan:
        prop_orig = original_properties_map.get(lp.property_id)
        if not prop_orig: continue

        # Get max discount for this product code from ProductFullDetail
        product_code_detail = optimizer.product_full_details_map.get(prop_orig.product_code)
        if not product_code_detail: continue # Should have been caught earlier, but safety check

        max_discount_for_pc = product_code_detail.max_discount_applicable
        
        current_discount_pct = lp.card_rate_discount_at_selection
        discount_room_pct = max_discount_for_pc - current_discount_pct

        if discount_room_pct > 1e-6:
            monetary_value_per_pct_discount = (lp.original_rate_per_day / 100.0) * lp.units * optimizer.days
            max_additional_monetary_discount_for_property = discount_room_pct * monetary_value_per_pct_discount
            
            properties_for_adjustment_info.append({
                'lease_plan_obj': lp,
                'property_id': lp.property_id,
                'product_code': prop_orig.product_code,
                'current_discount_pct': current_discount_pct,
                'max_allowed_discount_pct': max_discount_for_pc,
                'remaining_discount_room_monetary': max_additional_monetary_discount_for_property,
                'monetary_value_per_pct_discount': monetary_value_per_pct_discount
            })

    total_distributed_value = 0.0
    
    while remaining_discount_to_distribute > 1e-6 and any(p['remaining_discount_room_monetary'] > 1e-6 for p in properties_for_adjustment_info):
        
        eligible_product_codes = {}
        for p_info in properties_for_adjustment_info:
            if p_info['remaining_discount_room_monetary'] > 1e-6:
                eligible_product_codes[p_info['product_code']] = eligible_product_codes.get(p_info['product_code'], 0.0) + p_info['remaining_discount_room_monetary']
        
        if not eligible_product_codes: break

        total_eligible_monetary_room_overall = sum(eligible_product_codes.values())
        if total_eligible_monetary_room_overall < 1e-6: break
            
        share_factor_this_round = min(1.0, remaining_discount_to_distribute / total_eligible_monetary_room_overall)

        applied_this_round_iteration = 0.0

        for pc in eligible_product_codes:
            monetary_room_for_pc = eligible_product_codes[pc]
            amount_to_apply_to_this_pc = monetary_room_for_pc * share_factor_this_round
            
            properties_in_pc = [p for p in properties_for_adjustment_info if p['product_code'] == pc and p['remaining_discount_room_monetary'] > 1e-6]
            
            total_current_monetary_value_in_pc = sum(p['lease_plan_obj'].units * p['lease_plan_obj'].effective_rate_per_day * optimizer.days for p in properties_in_pc)
            
            if total_current_monetary_value_in_pc < 1e-6: continue
            
            proportional_distribution_factor = amount_to_apply_to_this_pc / total_current_monetary_value_in_pc

            for p_adj_info in properties_in_pc:
                potential_monetary_reduction_from_property = p_adj_info['lease_plan_obj'].units * p_adj_info['lease_plan_obj'].effective_rate_per_day * optimizer.days * proportional_distribution_factor
                actual_monetary_reduction_from_property = min(potential_monetary_reduction_from_property, p_adj_info['remaining_discount_room_monetary'])

                if p_adj_info['monetary_value_per_pct_discount'] > 1e-6:
                    additional_discount_pct_increase = actual_monetary_reduction_from_property / p_adj_info['monetary_value_per_pct_discount']
                else:
                    additional_discount_pct_increase = 0.0
                
                p_adj_info['lease_plan_obj'].card_rate_discount_at_selection += additional_discount_pct_increase
                p_adj_info['lease_plan_obj'].card_rate_discount_at_selection = min(p_adj_info['lease_plan_obj'].card_rate_discount_at_selection, p_adj_info['max_allowed_discount_pct'])
                
                p_adj_info['lease_plan_obj'].effective_rate_per_day = p_adj_info['lease_plan_obj'].original_rate_per_day * (1 - p_adj_info['lease_plan_obj'].card_rate_discount_at_selection / 100.0)
                
                p_adj_info['remaining_discount_room_monetary'] -= actual_monetary_reduction_from_property
                applied_this_round_iteration += actual_monetary_reduction_from_property

        remaining_discount_to_distribute -= applied_this_round_iteration
        total_distributed_value += applied_this_round_iteration
        
        if applied_this_round_iteration < 1e-6 and remaining_discount_to_distribute > 1e-6: break

    infeasible_discount_value = remaining_discount_to_distribute if remaining_discount_to_distribute > 1e-6 else 0.0

    final_adjusted_discounts_per_property = {
        p_info['property_id']: p_info['lease_plan_obj'].card_rate_discount_at_selection
        for p_info in properties_for_adjustment_info
    }
    for lp in adjusted_lease_plan:
        if lp.property_id not in final_adjusted_discounts_per_property:
            final_adjusted_discounts_per_property[lp.property_id] = lp.card_rate_discount_at_selection

    return adjusted_lease_plan, infeasible_discount_value, final_adjusted_discounts_per_property

def handle_over_max_budget_adjustment(
    current_lease_plan: List[LeasePlan], 
    all_properties_data: List[Property], 
    optimizer: LeaseOptimizerMILP,
    current_metrics: Dict # Current metrics for the plan
) -> Tuple[List[LeasePlan], float, Dict[str, float]]:
    """
    Attempts to apply further discounts to bring the total cost down if it exceeds the Max Budget allowed.
    
    Returns:
        - adjusted_lease_plan (List[LeasePlan]): The lease plan with adjusted discounts.
        - remaining_over_budget_amount (float): Any amount that could not be reduced.
        - final_adjusted_discounts_per_property (Dict[str, float]): Final applied discount % for each property.
    """
    adjusted_lease_plan = [lp for lp in current_lease_plan] # Create a mutable copy
    original_properties_map = {p.property_id: p for p in all_properties_data}

    max_allowed_cost = optimizer.budget * (1 + optimizer.budget_tolerance)
    amount_to_reduce_by_discount = current_metrics['total_plan_actual_cost'] - max_allowed_cost

    if amount_to_reduce_by_discount <= 1e-2:
        final_adjusted_discounts_per_property = {
            lp.property_id: lp.card_rate_discount_at_selection for lp in adjusted_lease_plan
        }
        return adjusted_lease_plan, 0.0, final_adjusted_discounts_per_property

    properties_for_adjustment_info = []
    for lp in adjusted_lease_plan:
        prop_orig = original_properties_map.get(lp.property_id)
        if not prop_orig: continue

        product_code_detail = optimizer.product_full_details_map.get(prop_orig.product_code)
        if not product_code_detail: continue 

        max_discount_for_pc = product_code_detail.max_discount_applicable
        current_discount_pct = lp.card_rate_discount_at_selection
        
        discount_room_pct = max_discount_for_pc - current_discount_pct

        if discount_room_pct > 1e-6:
            monetary_value_per_pct_discount = (lp.original_rate_per_day / 100.0) * lp.units * optimizer.days
            max_additional_monetary_discount_for_property = discount_room_pct * monetary_value_per_pct_discount
            
            properties_for_adjustment_info.append({
                'lease_plan_obj': lp,
                'property_id': lp.property_id,
                'product_code': prop_orig.product_code,
                'current_discount_pct': current_discount_pct,
                'max_allowed_discount_pct': max_discount_for_pc,
                'remaining_discount_room_monetary': max_additional_monetary_discount_for_property,
                'monetary_value_per_pct_discount': monetary_value_per_pct_discount
            })

    total_reduced_value = 0.0
    
    while amount_to_reduce_by_discount > 1e-6 and any(p['remaining_discount_room_monetary'] > 1e-6 for p in properties_for_adjustment_info):
        
        eligible_properties_with_room = [p_info for p_info in properties_for_adjustment_info if p_info['remaining_discount_room_monetary'] > 1e-6]
        if not eligible_properties_with_room: break

        share_per_property_this_round = amount_to_reduce_by_discount / len(eligible_properties_with_room)
        applied_this_round_iteration = 0.0

        for p_adj_info in eligible_properties_with_room:
            actual_monetary_reduction_from_property = min(share_per_property_this_round, p_adj_info['remaining_discount_room_monetary'])

            if p_adj_info['monetary_value_per_pct_discount'] > 1e-6:
                additional_discount_pct_increase = actual_monetary_reduction_from_property / p_adj_info['monetary_value_per_pct_discount']
            else:
                additional_discount_pct_increase = 0.0
            
            p_adj_info['lease_plan_obj'].card_rate_discount_at_selection += additional_discount_pct_increase
            p_adj_info['lease_plan_obj'].card_rate_discount_at_selection = min(p_adj_info['lease_plan_obj'].card_rate_discount_at_selection, p_adj_info['max_allowed_discount_pct'])
            
            p_adj_info['lease_plan_obj'].effective_rate_per_day = p_adj_info['lease_plan_obj'].original_rate_per_day * (1 - p_adj_info['lease_plan_obj'].card_rate_discount_at_selection / 100.0)
            
            p_adj_info['remaining_discount_room_monetary'] -= actual_monetary_reduction_from_property
            applied_this_round_iteration += actual_monetary_reduction_from_property

        amount_to_reduce_by_discount -= applied_this_round_iteration
        total_reduced_value += applied_this_round_iteration

        if applied_this_round_iteration < 1e-6 and amount_to_reduce_by_discount > 1e-6: break

    remaining_over_budget_amount = amount_to_reduce_by_discount if amount_to_reduce_by_discount > 1e-2 else 0.0

    final_adjusted_discounts_per_property = {
        p_info['property_id']: p_info['lease_plan_obj'].card_rate_discount_at_selection
        for p_info in properties_for_adjustment_info
    }
    for lp in adjusted_lease_plan:
        if lp.property_id not in final_adjusted_discounts_per_property:
            final_adjusted_discounts_per_property[lp.property_id] = lp.card_rate_discount_at_selection

    return adjusted_lease_plan, remaining_over_budget_amount, final_adjusted_discounts_per_property


# --- Refactored Display Functions for Streamlit ---

def display_optimizer_configuration_content_st(optimizer: LeaseOptimizerMILP):
    """Prints the configuration settings of the optimizer using Streamlit, formatted as a card."""
    st.markdown("### ⚙️ **Optimizer Configuration**")

    st.markdown("#### Client Request")
    st.info(f"**Target Units:** {optimizer.units_required} \n"
            f"**Minimum Cost Target:** ₹{optimizer.budget:,.0f} \n"
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

    st.markdown("#### Optional Unit Constraints")
    if optimizer.min_units_per_city:
        st.write("**Min Units per City:**")
        for city, units in optimizer.min_units_per_city.items():
            st.write(f"  - {city}: {units} units")
    else:
        st.write("Min Units per City: None specified.")

    if optimizer.min_units_per_product_code:
        st.write("**Min Units per Product Code:**")
        for code, units in optimizer.min_units_per_product_code.items():
            st.write(f"  - {code}: {units} units")
    else:
        st.write("Min Units per Product Code: None specified.")

    st.markdown("#### Product Details & Max Discounts:")
    if optimizer.product_full_details_map:
        for code, pfd in optimizer.product_full_details_map.items():
            st.write(f"  - **{code}**: Base Rate ₹{pfd.base_rate_per_day:.2f}/day, Default Discount {pfd.default_card_rate_discount:.1f}%, Max Discount {pfd.max_discount_applicable:.1f}%")
    st.write(f"Default Max Discount (for unknown Product Codes): {optimizer.default_max_discount_if_not_specified:.1f}%")


def calculate_plan_metrics(lease_plan: List[LeasePlan], all_properties_data: List[Property], optimizer: LeaseOptimizerMILP):
    """Calculates key metrics for a given lease plan."""
    # Wrap calculations in try-except to catch unexpected errors and return safe defaults
    try:
        total_plan_actual_cost = 0
        total_plan_units = 0
        total_penalty_for_partial_leases = 0
        total_penalty_for_high_occupancy = 0
        selected_groups: Dict[str, int] = {}
        selected_cities_units: Dict[str, int] = {}
        selected_product_codes_units: Dict[str, int] = {}
        total_properties_selected_in_plan = 0

        for lease in lease_plan:
            prop_obj_orig = next((p for p in all_properties_data if p.property_id == lease.property_id), None)
            
            if prop_obj_orig:
                current_rental_for_units = lease.units * lease.effective_rate_per_day * optimizer.days
                total_plan_actual_cost += current_rental_for_units
                total_plan_units += lease.units
                total_properties_selected_in_plan += 1

                if not lease.full_property and not lease.pre_selected:
                    total_penalty_for_partial_leases += optimizer.full_property_penalty_per_unit # Corrected to use optimizer's penalty
                
                if lease.occupancy_rating_at_selection > optimizer.occupancy_rating_threshold:
                    total_penalty_for_high_occupancy += lease.units * optimizer.occupancy_penalty_per_unit_per_day * optimizer.days

                selected_groups[prop_obj_orig.group] = selected_groups.get(prop_obj_orig.group, 0) + 1
                selected_cities_units[prop_obj_orig.city] = selected_cities_units.get(prop_obj_orig.city, 0) + lease.units
                selected_product_codes_units[prop_obj_orig.product_code] = selected_product_codes_units.get(prop_obj_orig.product_code, 0) + lease.units
            
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
        
        city_constraint_met_overall = True
        if optimizer.min_units_per_city:
            for city, required_units in optimizer.min_units_per_city.items():
                if selected_cities_units.get(city, 0) < required_units:
                    city_constraint_met_overall = False
                    break

        product_code_constraint_overall = True
        if optimizer.min_units_per_product_code:
            for product_code, required_units in optimizer.min_units_per_product_code.items():
                if selected_product_codes_units.get(product_code, 0) < required_units:
                    product_code_constraint_overall = False
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
            "city_constraint_met_overall": city_constraint_met_overall,
            "product_code_constraint_met_overall": product_code_constraint_overall,
            "selected_groups": selected_groups,
            "selected_cities_units": selected_cities_units,
            "selected_product_codes_units": selected_product_codes_units,
            "min_budget_allowed": min_budget_allowed,
            "max_budget_allowed": max_budget_allowed,
            "min_units_allowed": min_units_allowed,
            "max_units_allowed": max_units_allowed,
        }
    except Exception as e:
        st.error(f"Error calculating metrics for a plan: {e}. Returning default values.")
        return {
            "total_plan_actual_cost": 0.0,
            "total_plan_units": 0,
            "total_penalty_for_partial_leases": 0.0,
            "total_penalty_for_high_occupancy": 0.0,
            "total_objective_value": None, # Indicate objective calculation failed
            "total_properties_selected": 0,
            "budget_status": False,
            "units_status": False,
            "group_constraint_met_overall": False,
            "city_constraint_met_overall": False,
            "product_code_constraint_met_overall": False,
            "selected_groups": {},
            "selected_cities_units": {},
            "selected_product_codes_units": {},
            "min_budget_allowed": 0.0,
            "max_budget_allowed": 0.0,
            "min_units_allowed": 0.0,
            "max_units_allowed": 0.0
        }


def print_plan_details_st(
    plan_idx: int, 
    lease_plan: List[LeasePlan], 
    all_properties_data: List[Property], 
    optimizer: LeaseOptimizerMILP,
    infeasible_discount_value: float = 0.0,
    remaining_over_budget_amount: float = 0.0,
    adjusted_discounts_per_property: Optional[Dict[str, float]] = None
):
    """Prints detailed information for a single optimized lease plan within an expander."""
    
    metrics = calculate_plan_metrics(lease_plan, all_properties_data, optimizer)

    st.markdown(f"#### Properties in Plan {plan_idx}")
    
    property_details_for_display = []
    for lease in lease_plan:
        prop_obj_orig = next((p for p in all_properties_data if p.property_id == lease.property_id), None)
        
        if prop_obj_orig:
            current_rental_for_units = lease.units * lease.effective_rate_per_day * optimizer.days
            pre_selected_indicator = "(PRE-SELECTED)" if lease.pre_selected else ""
            
            final_discount_pct = adjusted_discounts_per_property.get(lease.property_id, lease.card_rate_discount_at_selection) if adjusted_discounts_per_property else lease.card_rate_discount_at_selection
            
            property_details_for_display.append({
                "Property ID": lease.property_id,
                "City": prop_obj_orig.city,
                "Product Code": prop_obj_orig.product_code,
                "Group": prop_obj_orig.group,
                "Units Leased": lease.units,
                "Base Rate/Day": f"₹{lease.original_rate_per_day:.2f}",
                "Initial Discount (%)": f"{lease.card_rate_discount_at_selection:.1f}%",
                "Final Discount (%)": f"{final_discount_pct:.1f}%",
                "Effective Rate/Day": f"₹{lease.effective_rate_per_day:.2f}",
                "Lease Type": "FULL PROPERTY" if lease.full_property else "PARTIAL LEASE",
                "Cost for Property": f"₹{int(current_rental_for_units):,}",
                "Status": pre_selected_indicator
            })
    
    if property_details_for_display:
        st.dataframe(property_details_for_display, use_container_width=True)
    else:
        st.info("No properties selected in this plan.")


    st.markdown(f"#### Summary Metrics for Plan {plan_idx}")
    # Use .get() for robustness, and provide default format string if value is not numeric
    total_plan_actual_cost_display = f"₹{int(metrics.get('total_plan_actual_cost',0.0)):,}" # Default float 0.0
    total_plan_units_display = metrics.get('total_plan_units', 0)
    total_penalty_partial_display = f"₹{int(metrics.get('total_penalty_for_partial_leases',0.0)):,}" # Default float 0.0
    total_penalty_occupancy_display = f"₹{int(metrics.get('total_penalty_for_high_occupancy',0.0)):,}" # Default float 0.0
    total_objective_display = f"₹{int(metrics.get('total_objective_value',0.0)):,}" # Default float 0.0
    total_properties_selected_display = metrics.get('total_properties_selected', 0)

    st.write(f"  **Total Plan Actual Rental Cost:** {total_plan_actual_cost_display}")
    st.write(f"  **Total Plan Units Leased:** {total_plan_units_display}")
    st.write(f"  **Total Penalty for Partial Leases:** {total_penalty_partial_display}")
    st.write(f"  **Total Penalty for High Occupancy Ratings:** {total_penalty_occupancy_display}")
    st.markdown(f"### **TOTAL OBJECTIVE VALUE:** {total_objective_display}")
    st.write(f"  **Total Properties Selected:** {total_properties_selected_display}")

    # Robust formatting for budget and units target ranges
    min_budget_allowed_display = f"₹{int(metrics.get('min_budget_allowed', 0.0)):,.0f}"
    max_budget_allowed_display = f"₹{int(metrics.get('max_budget_allowed', 0.0)):,.0f}"
    budget_status_icon = '✅' if metrics.get('budget_status', False) else '❌'

    min_units_allowed_display = f"{metrics.get('min_units_allowed', 0.0):.1f}"
    max_units_allowed_display = f"{metrics.get('max_units_allowed', 0.0):.1f}"
    units_status_icon = '✅' if metrics.get('units_status', False) else '❌'


    st.write(f"  **Minimum Cost Target:** ₹{optimizer.budget:,.0f} (Range: {min_budget_allowed_display} - {max_budget_allowed_display}) {budget_status_icon}")
    st.write(f"  **Units Target:** {optimizer.units_required} (Range: {min_units_allowed_display} - {max_units_allowed_display}) {units_status_icon}")

    if infeasible_discount_value > 1e-2:
        st.warning(f"⚠️ **Min Budget Fill-up Infeasible Discount:** ₹{int(infeasible_discount_value):,} (Couldn't apply enough discount to reach minimum budget target due to caps.)")
    else:
        st.success("✅ Min Budget Fill-up Achieved.")

    if remaining_over_budget_amount > 1e-2:
        st.error(f"❌ **Remaining Over Max Budget:** ₹{int(remaining_over_budget_amount):,} (Plan still exceeds max allowed budget after all possible discounts.)")
    else:
        st.success("✅ Plan is within or below max allowed budget after adjustments.")


    st.markdown("#### Selected Properties by Group:")
    group_counts_all_data = {p.group: 0 for p in all_properties_data}
    for p in all_properties_data:
        group_counts_all_data[p.group] = group_counts_all_data.get(p.group, 0) + 1

    all_groups_present = sorted(list(set(group_counts_all_data.keys()).union(metrics.get('selected_groups', {}).keys())))
    
    group_summary_lines = []
    for group_char in all_groups_present:
        count = metrics.get('selected_groups', {}).get(group_char, 0)
        
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
    
    for line in group_summary_lines:
        st.write(line)

    if metrics.get('group_constraint_met_overall', False):
        st.success("✅ **All required group constraints met for this plan.**")
    else:
        st.error("❌ **Some required group constraints NOT met for this plan.**")

    st.markdown("#### Units by City:")
    if optimizer.min_units_per_city:
        for city, required_units in optimizer.min_units_per_city.items():
            actual_units = metrics.get('selected_cities_units', {}).get(city, 0)
            status = '✅' if actual_units >= required_units else '❌'
            st.write(f"  - City '{city}': {actual_units} units selected (Required >= {required_units}) {status}")
        if metrics.get('city_constraint_met_overall', False):
            st.success("✅ **All minimum units per city constraints met.**")
        else:
            st.error("❌ **Some minimum units per city constraints NOT met.**")
    else:
        st.info("No minimum units per city constraints specified.")

    st.markdown("#### Units by Product Code:")
    if optimizer.min_units_per_product_code:
        for product_code, required_units in optimizer.min_units_per_product_code.items():
            actual_units = metrics.get('selected_product_codes_units', {}).get(product_code, 0)
            status = '✅' if actual_units >= required_units else '❌'
            st.write(f"  - Product Code '{product_code}': {actual_units} units selected (Required >= {required_units}) {status}")
        if metrics.get('product_code_constraint_met_overall', False):
            st.success("✅ **All minimum units per product code constraints met.**")
        else:
            st.error("❌ **Some minimum units per product code constraints NOT met.**")
    else:
        st.info("No minimum units per product code constraints specified.")


# --- Helper for parsing dict string ---
def parse_dict_string(dict_string: str) -> Dict[str, int]:
    """Parses a string representation of a dictionary (e.g., '{"CityA": 10, "CityB": 5}') into a dictionary."""
    if not dict_string.strip(): return {}
    try:
        parsed_dict = json.loads(dict_string)
        if not isinstance(parsed_dict, dict): raise ValueError("Input is not a valid dictionary format.")
        return {str(k): int(v) for k, v in parsed_dict.items()}
    except json.JSONDecodeError:
        result = {}
        cleaned_string = dict_string.replace('{', '').replace('}', '').strip()
        parts = [p.strip() for p in cleaned_string.split(',')]
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip().strip("'\"")
                try: result[key] = int(value.strip())
                except ValueError: raise ValueError(f"Invalid integer value for key '{key}': '{value.strip()}'")
            elif part: raise ValueError(f"Invalid format: '{part}'. Expected 'key:value' pairs or a valid JSON string.")
        return result
    except ValueError as e: raise ValueError(f"Error parsing dictionary string: {e}")
    except Exception as e: raise ValueError(f"Unexpected error parsing dictionary string: {e}")


def generate_random_tag(length: int = 4) -> str:
    """Generates a random alphanumeric tag."""
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for i in range(length))

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("Rental Lease Optimization App")
st.write("Configure parameters and find optimal lease plans based on your criteria.")

# --- Product Details Data Source (Combined) ---
st.sidebar.header("Product Details & Discount Limits")
# Initial state for expander to be open by default
product_details_expander_default = False
if 'product_details_expander_state' not in st.session_state:
    st.session_state.product_details_expander_state = product_details_expander_default

with st.sidebar.expander("Product Details Data Source", expanded=st.session_state.product_details_expander_state):
    product_full_details_data_option = st.radio(
        "Select Product Details Data Source:",
        ("Generate Random Product Details", "Upload CSV File"),
        key="product_full_details_data_source"
    )

    product_full_details_data: List[ProductFullDetail] = []
    product_details_tag = ""

    if product_full_details_data_option == "Generate Random Product Details":
        num_product_codes_to_generate_details = st.slider("Number of Product Details to Generate", 1, 10, 3)
        product_full_details_data = generate_random_product_full_details(num_product_codes_to_generate_details)
        if product_full_details_data:
            product_details_tag = f" (Randomly Generated - {generate_random_tag()})"
            st.info(f"Generated {len(product_full_details_data)} random product details.")
        else:
            st.warning("No product details generated.")
    else:
        csv_sample_product_full_details = create_sample_product_full_details_csv_content()
        st.download_button(
            label="Download Sample Product Details CSV",
            data=csv_sample_product_full_details,
            file_name="sample_product_details_full.csv",
            mime="text/csv",
            help="Download a sample CSV for Product Details. Expected columns: product_code,base_rate_per_day,default_card_rate_discount,max_discount_applicable."
        )
        uploaded_product_full_details_file = st.file_uploader(
            "Choose a CSV file for Product Details",
            type="csv",
            help="Expected columns: product_code,base_rate_per_day,default_card_rate_discount,max_discount_applicable."
        )
        if uploaded_product_full_details_file is not None:
            string_data_product_full_details = io.StringIO(uploaded_product_full_details_file.getvalue().decode("utf-8"))
            try:
                product_full_details_data = load_product_full_details_from_csv(string_data_product_full_details)
                if product_full_details_data:
                    st.success(f"Loaded **{len(product_full_details_data)}** product details from CSV.")
                else:
                    st.warning("Product Details CSV loaded, but no valid details parsed.")
            except Exception as e:
                st.error(f"Error processing Product Details CSV: {e}.")
                product_full_details_data = []

# Update expander title dynamically
st.sidebar.markdown(f'<h3 style="margin-top: 0rem;">Product Details & Discount Limits{product_details_tag}</h3>', unsafe_allow_html=True)


if not product_full_details_data:
    st.error("No Product Details loaded or generated. Please provide this data to proceed.")
    st.stop()

# Get available product codes from the loaded/generated ProductFullDetail for consistency checks
available_product_codes_from_details = [pfd.product_code for pfd in product_full_details_data]


# --- Property Data Source ---
st.sidebar.markdown("---")
st.sidebar.header("Property Data Source") # This will now be inside the expander
# Initial state for expander to be open by default
property_expander_default = False
if 'property_expander_state' not in st.session_state:
    st.session_state.property_expander_state = property_expander_default

with st.sidebar.expander("Property Data Source", expanded=st.session_state.property_expander_state):
    data_source_option = st.radio("Select Property Data Source:", ("Generate Random Properties", "Upload CSV File"), key="property_data_source")

    properties_data: List[Property] = []
    properties_tag = ""

    if data_source_option == "Generate Random Properties":
        num_properties = st.slider("Number of Random Properties", 50, 1000, 300)
        properties_data = generate_random_properties(num_properties, available_product_codes_from_details)
        if properties_data:
            properties_tag = f" (Randomly Generated - {generate_random_tag()})"
            st.info(f"Generated {len(properties_data)} random properties.")
        else:
            st.warning("No properties generated.")
    else:
        csv_sample_data = create_sample_properties_csv_content()
        st.download_button(
            label="Download Sample Property CSV",
            data=csv_sample_data,
            file_name="sample_properties.csv",
            mime="text/csv",
            help="Download a sample CSV file. Expected columns: property_id,total_units,occupancy_rating,group,city,product_code."
        )
        uploaded_file = st.file_uploader("Choose a CSV file for Properties", type="csv", help="Expected columns: property_id,total_units,occupancy_rating,group,city,product_code.")
        if uploaded_file is not None:
            string_data = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            try:
                properties_data = load_properties_from_csv(string_data)
                if properties_data:
                    st.success(f"Loaded **{len(properties_data)}** properties from CSV.")
                    
                    # Validation: Check if property product codes exist in product_full_details_data
                    missing_prop_codes = set()
                    for p in properties_data:
                        if p.product_code not in available_product_codes_from_details:
                            missing_prop_codes.add(p.product_code)
                    if missing_prop_codes:
                        st.error(f"Error: Properties data contains unknown product codes: {', '.join(missing_prop_codes)}. Please ensure all product codes in properties data are defined in Product Details data.")
                        properties_data = [] # Clear data if consistency issue
                else: st.error("CSV loaded, but no valid properties parsed. Check error messages above.")
            except Exception as e:
                st.error(f"Error processing Properties CSV: {e}. Please check file format and try again.")
                properties_data = []

# Update expander title dynamically
st.sidebar.markdown(f'<h3 style="margin-top: 0rem;">Property Data Source{properties_tag}</h3>', unsafe_allow_html=True)


if not properties_data:
    st.error("No properties loaded or generated. Please adjust data source settings to proceed.")
    st.stop()


# --- Lease Period ---
st.sidebar.markdown("---")
st.sidebar.header("Lease Period")
col1, col2 = st.sidebar.columns(2)
start_date_val = col1.date_input("Start Date", date(2025, 7, 1))
end_date_val = col2.date_input("End Date", date(2025, 7, 10))

lease_days = (end_date_val - start_date_val).days + 1
if lease_days <= 0:
    st.sidebar.error("End Date must be after Start Date.")
    st.stop()

# Estimate initial units/budget using product details
estimated_units = max(1, int(sum(p.total_units for p in properties_data) * 0.4))
avg_effective_rate_overall = 0.0
total_properties_with_valid_product_detail = 0
for p in properties_data:
    # Ensure product code exists (should be guaranteed by earlier check if properties_data is not empty)
    if p.product_code in available_product_codes_from_details:
        pfd_obj = next(pfd for pfd in product_full_details_data if pfd.product_code == p.product_code)
        avg_effective_rate_overall += pfd_obj.base_rate_per_day * (1 - pfd_obj.default_card_rate_discount / 100.0)
        total_properties_with_valid_product_detail += 1
if total_properties_with_valid_product_detail > 0:
    avg_effective_rate_overall /= total_properties_with_valid_product_detail
else:
    avg_effective_rate_overall = 1000.0
estimated_budget_val = int(estimated_units * avg_effective_rate_overall * lease_days * 1.05)


# --- Grouped Controls ---

# Basic Controls
with st.sidebar.expander("🎯 Basic Client Requirements", expanded=True):
    units_required = st.number_input("Target Units Required", value=estimated_units, min_value=1)
    budget = st.number_input("Minimum Cost Target (₹)", value=float(estimated_budget_val), min_value=1.0)
    budget_tolerance_percent = st.slider("Budget Upper Tolerance (%)", 0.0, 50.0, 10.0, help="Allows final cost to be up to this percentage above the Minimum Cost Target.")
    unit_tolerance_percent = st.slider("Units Tolerance (%)", 0.0, 50.0, 15.0, help="Allows final units to be within +/- this percentage of Target Units Required.")

# Pro Controls
with st.sidebar.expander("⚖️ Pro Constraints & Specifics", expanded=False):
    min_properties_per_group = st.slider("Min Properties per Group", 0, 5, 0, help="Minimum number of properties to select from each group. Set to 0 to disable this constraint.")
    available_property_ids = [p.property_id for p in properties_data]
    pre_selected_ids = st.multiselect("Select Properties to ALWAYS Include:", available_property_ids, help="These properties will be forced into the solution, taking all their units.")

# Advanced Controls
with st.sidebar.expander("✨ Advanced Optional Constraints", expanded=False):
    min_units_city_input = st.text_area(
        "Min Units per City (JSON or 'City:Units')",
        value="{}",
        help="Specify minimum units required from certain cities. Example: `{'Mumbai': 10, 'Delhi': 5}` or `Mumbai:10,Delhi:5`. Leave empty for no constraint."
    )
    min_units_product_code_input = st.text_area(
        "Min Units per Product Code (JSON or 'Code:Units')",
        value="{}",
        help="Specify minimum units required for certain product codes. Example: `{'COMMERCIAL': 20, 'RETAIL': 8}` or `COMMERCIAL:20,RETAIL:8`. Leave empty for no constraint."
    )

# Admin Controls
with st.sidebar.expander("⚙️ Admin Settings & Penalties", expanded=False):
    full_property_penalty_per_unit = st.number_input("Partial Lease Penalty (₹/unit)", value=50.0, min_value=0.0, help="Penalty for each unit short of a full property lease, if property is selected.")
    occupancy_penalty_per_unit_per_day = st.number_input("High Occupancy Penalty (₹/unit/day)", value=150.0, min_value=0.0, help="Penalty applied per unit per day for properties whose Occupancy Rating is above the threshold.")
    occupancy_rating_threshold = st.slider("Occupancy Rating Threshold (for penalty)", 1, 10, 6, help="Properties with an Occupancy Rating *greater than* this value will incur a penalty.")
    solver_time_limit_seconds = st.number_input("Solver Time Limit (seconds/plan)", value=10, min_value=1, help="Maximum time the solver will run for each individual plan before returning the best solution found so far.")
    solver_mip_gap = st.slider("Solver MIP Gap (%)", 0.0, 10.0, 5.0, help="Solver stops when the solution is guaranteed to be within this percentage of the true optimal value.") / 100.0
    num_solutions = st.slider("Number of Solutions to Find", 1, 5, 3, help="The optimizer will attempt to find this many distinct optimal/near-optimal solutions.")
    default_max_discount_if_not_specified = st.number_input(
        "Default Max Discount (for unknown Product Codes) (%)",
        value=40.0,
        min_value=0.0,
        max_value=100.0,
        help="This discount limit will be used as the 'max_discount_applicable' for any product code that isn't defined in the 'Product Details Data'. It acts as a safety net."
    )


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

    parsed_min_units_per_city = {}
    if min_units_city_input:
        try: parsed_min_units_per_city = parse_dict_string(min_units_city_input)
        except ValueError as e: st.error(f"Error parsing 'Min Units per City' input: {e}."); st.stop()

    parsed_min_units_per_product_code = {}
    if min_units_product_code_input:
        try: parsed_min_units_per_product_code = parse_dict_string(min_units_product_code_input)
        except ValueError as e: st.error(f"Error parsing 'Min Units per Product Code' input: {e}."); st.stop()


    optimizer = LeaseOptimizerMILP(
        properties=properties_data,
        product_full_details=product_full_details_data,
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
        explicit_pre_selected_property_ids=valid_pre_selected_ids,
        min_units_per_city=parsed_min_units_per_city,
        min_units_per_product_code=parsed_min_units_per_product_code,
        default_max_discount_if_not_specified=default_max_discount_if_not_specified
    )

    st.markdown("### Optimization Run Summary")
    col_config, col_time = st.columns([3, 1])
    with col_config:
        with st.expander("View Optimizer Configuration", expanded=True):
            display_optimizer_configuration_content_st(optimizer)

    with col_time:
        st.markdown("##### Solver Run Time")
        total_solve_start_time = time.time()
        plans_raw = optimizer.solve(num_solutions=num_solutions)
        total_solve_end_time = time.time()
        total_elapsed_time = total_solve_end_time - total_solve_start_time
        st.metric(label="Total time to find all plans", value=f"{total_elapsed_time:.2f} seconds")

    plans_with_metrics_and_adjusted_info = []
    for plan_list in plans_raw:
        initial_metrics = calculate_plan_metrics(plan_list, properties_data, optimizer)
        
        # Check if initial_metrics calculation itself failed
        if initial_metrics['total_objective_value'] is None:
            st.error(f"Skipping a plan due to error in initial metric calculation. Plan Details: {plan_list[0].property_id if plan_list else 'N/A'}")
            continue

        adjusted_plan_step1, infeasible_discount_val_step1, final_adj_discounts_step1 = post_process_discounts(
            plan_list, properties_data, optimizer, initial_metrics
        )
        
        metrics_after_step1 = calculate_plan_metrics(adjusted_plan_step1, properties_data, optimizer)
        if metrics_after_step1['total_objective_value'] is None:
            st.error(f"Skipping a plan due to error in metrics after Step 1 adjustment. Plan Details: {plan_list[0].property_id if plan_list else 'N/A'}")
            continue

        adjusted_plan_step2, remaining_over_budget_amount, final_adj_discounts_step2 = handle_over_max_budget_adjustment(
            adjusted_plan_step1, properties_data, optimizer, metrics_after_step1
        )
        
        final_metrics_after_adjustment = calculate_plan_metrics(adjusted_plan_step2, properties_data, optimizer)
        if final_metrics_after_adjustment['total_objective_value'] is None:
            st.error(f"Skipping a plan due to error in final metric calculation. Plan Details: {plan_list[0].property_id if plan_list else 'N/A'}")
            continue

        plans_with_metrics_and_adjusted_info.append(
            (final_metrics_after_adjustment['total_objective_value'], adjusted_plan_step2, final_metrics_after_adjustment, infeasible_discount_val_step1, remaining_over_budget_amount, final_adj_discounts_step2)
        )

    sorted_plans_to_display = sorted(plans_with_metrics_and_adjusted_info, key=lambda x: x[0])
    
    if not sorted_plans_to_display:
        st.error("❌ **No feasible plans found based on the given constraints.** Please adjust your parameters and try again.")
    else:
        st.markdown("---")
        st.subheader("Summary of Generated Plans")
        
        cols_per_row = 3
        
        for i in range(0, len(sorted_plans_to_display), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if (i + j) < len(sorted_plans_to_display):
                    plan_idx = i + j + 1
                    objective_value, plan, metrics, infeasible_discount_val_step1, remaining_over_budget_amount, final_adj_discounts = sorted_plans_to_display[i+j]
                    
                    with cols[j]:
                        st.markdown(f"#### Plan {plan_idx}")
                        st.write(f"**Cost:** ₹{int(metrics.get('total_plan_actual_cost',0.0)):,}")
                        st.write(f"**Units:** {metrics.get('total_plan_units',0)}")
                        st.write(f"**Total Penalty:** ₹{int(metrics.get('total_penalty_for_partial_leases',0.0) + metrics.get('total_penalty_for_high_occupancy',0.0)):,}")
                        st.markdown(f"**Objective Value:** ₹{int(metrics.get('total_objective_value',0.0)):,}")
                        
                        st.write(f"Budget Status: {'✅' if metrics.get('budget_status', False) else '❌'}")
                        st.write(f"Units Status: {'✅' if metrics.get('units_status', False) else '❌'}")
                        st.write(f"Group Status: {'✅' if metrics.get('group_constraint_met_overall', False) else '❌'}")
                        st.write(f"City Units Status: {'✅' if metrics.get('city_constraint_met_overall', False) else '❌'}")
                        st.write(f"Product Code Units Status: {'✅' if metrics.get('product_code_constraint_met_overall', False) else '❌'}")

                        if infeasible_discount_val_step1 > 1e-2:
                            st.warning("⚠️ Min Budget Not Reached (see details)")
                        else:
                            st.success("✅ Min Budget Achieved")

                        if remaining_over_budget_amount > 1e-2:
                            st.error("❌ Over Max Budget (see details)")
                        else:
                            st.success("✅ Within Max Budget")
                        
                        with st.expander(f"View Details for Plan {plan_idx}"):
                            print_plan_details_st(plan_idx, plan, properties_data, optimizer, infeasible_discount_value=infeasible_discount_val_step1, remaining_over_budget_amount=remaining_over_budget_amount, adjusted_discounts_per_property=final_adj_discounts)
            st.markdown("---")