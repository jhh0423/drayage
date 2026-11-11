import random
import math
import copy
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Union, Optional
from pathlib import Path
import csv
import matplotlib.pyplot as plt


# %%
# 데이터 구조 정의
class Request:
    def __init__(self):
        self.id = ""
        self.terminal_index = ""
        self.x = 0.0
        self.y = 0.0
        self.e = 0
        self.l = 0
        self.e_t = 0
        self.l_t = 0
        self.demand = 0
        self.type = ""
        self.service_time = 0
        self.service_time_t = 0


class Fleet:
    def __init__(self):
        self.id = ""
        self.x = 0.0
        self.y = 0.0
        self.e = 0
        self.l = 0
        self.capacity = 0


class Depot:
    def __init__(self):
        self.id = ""
        self.x = 0.0
        self.y = 0.0


class Node:
    """
    Pickup 및 Delivery 노드를 나타내는 클래스.
    하나의 Request는 Pickup('P')과 Delivery('D') 두 개의 Node로 분리됩니다.
    """
    def __init__(self, request: Request, node_type: str):
        self.request = request
        self.node_type = node_type  # 'P' (Pickup) or 'D' (Delivery)
        self.req_id = request.id
        self.is_bypassed = False  # ce->se direct 연결 시 컨테이너 창고 방문을 생략하는 플래그

        # 요청 종류별 P, D 지점, 시간창, 서비스 시간, 용량 변화 정의
        if request.type == 'cf':  # Inbound Full: Terminal(P) -> Customer(D)
            if node_type == 'P':
                self.location_id = f"tm{request.terminal_index}"
                self.e = request.e_t
                self.l = request.l_t
                self.service_time = request.service_time_t
                self.demand_delta = -request.demand
            else:
                self.location_id = f"cf{request.id}"
                self.e = request.e
                self.l = request.l
                self.service_time = request.service_time
                self.demand_delta = request.demand

        elif request.type == 'sf':  # Outbound Full: Customer(P) -> Terminal(D)
            if node_type == 'P':
                self.location_id = f"sf{request.id}"
                self.e = request.e
                self.l = request.l
                self.service_time = request.service_time
                self.demand_delta = request.demand
            else:
                self.location_id = f"tm{request.terminal_index}"
                self.e = request.e_t
                self.l = request.l_t
                self.service_time = request.service_time_t
                self.demand_delta = -request.demand

        elif request.type == 'ce':  # Inbound Empty: Customer(P) -> Container Depot(D)
            if node_type == 'P':
                self.location_id = f"ce{request.id}"
                self.e = request.e
                self.l = request.l
                self.service_time = request.service_time
                self.demand_delta = request.demand
            else:
                self.location_id = f"cd{request.terminal_index}"
                self.e = -10000
                self.l = 10000
                self.service_time = request.service_time_t
                self.demand_delta = -request.demand

        elif request.type == 'se':  # Outbound Empty: Container Depot(P) -> Customer(D)
            if node_type == 'P':
                self.location_id = f"cd{request.terminal_index}"
                self.e = -10000
                self.l = 10000
                self.service_time = request.service_time_t
                self.demand_delta = -request.demand
            else:
                self.location_id = f"se{request.id}"
                self.e = request.e
                self.l = request.l
                self.service_time = request.service_time
                self.demand_delta = request.demand

        else:
            self.location_id = f"{request.type}{request.id}"
            self.e = request.e
            self.l = request.l
            self.service_time = request.service_time
            self.demand_delta = request.demand if node_type == 'P' else -request.demand

    def __repr__(self):
        bypass_str = "(Bypassed)" if self.is_bypassed else ""
        return f"{self.node_type}_{self.location_id}(req{self.req_id}){bypass_str}"


# %%
# ALNS 솔버 클래스
class DrayageALNSSolver:
    def __init__(self, list_fleet: List[Fleet], list_request: List[Request], 
                 list_terminal: List[Depot], list_container_depot: List[Depot], 
                 dict_arc: Dict, time_unit: int = 1, 
                 cost_transportation: float = 1.0, cost_waiting: float = 0.5):
        
        # 파라미터
        self.time_unit = time_unit
        self.cost_transportation = cost_transportation
        self.cost_waiting = cost_waiting
        
        # 문제 데이터
        self.list_fleet = list_fleet
        self.list_request = list_request
        self.list_terminal = list_terminal
        self.list_container_depot = list_container_depot
        self.dict_arc = dict_arc
        num_request = len(list_request)
        
        # ALNS 파라미터
        self.max_iter = 25000
        self.cooling_rate = 0.999
        self.start_temp = 100
        self.min_temp = 0.1
        self.noise_level = 0.05  # Noise Insertion 비율 (5%)
        
        self.destroy_ops = ['random', 'worst', 'time_window', 'related']
        self.repair_ops = ['greedy', 'regret']
        self.destroy_min = num_request // 10
        self.destroy_max = max(1, num_request // 5)
        self.decay = 0.8
        self.time_window_penalty = 1000
        self.capacity_penalty = 500
        self.location_related_factor = 0.7
        
        # 연산자 관리
        self.weights = {
            'destroy': {op: 1.0 for op in self.destroy_ops},
            'repair': {op: 1.0 for op in self.repair_ops}
        }
        self.scores = {
            'destroy': defaultdict(float),
            'repair': defaultdict(float)
        }
        self.usage = {
            'destroy': defaultdict(int),
            'repair': defaultdict(int)
        }
        
        # 미지정(-1) Empty 요청에 대해 가장 가까운 컨테이너 창고 매칭
        self._assign_initial_container_depots()
        
        # 위치 기반 관련성 계산
        self._compute_location_proximity()
        
        # 솔루션 정보
        self.best_solution = self.initial_solution()
        self.best_cost, _, _ = self.calculate_total_cost(self.best_solution)
        self.history = []
        self.history_best = []
        
        print(f"Initial solution cost: {self.best_cost:.2f}")
        
    def _assign_initial_container_depots(self):
        """terminal_index가 -1인 Empty 요청(ce, se)에 대해 가장 가까운 컨테이너 창고를 초기 매칭"""
        if not self.list_container_depot:
            return

        for req in self.list_request:
            if req.type in ['ce', 'se']:
                try:
                    t_idx = int(req.terminal_index)
                except ValueError:
                    t_idx = -1

                if t_idx < 0:
                    nearest_depot = min(
                        self.list_container_depot,
                        key=lambda cd: math.sqrt((req.x - cd.x)**2 + (req.y - cd.y)**2)
                    )
                    req.terminal_index = nearest_depot.id

    def _compute_location_proximity(self):
        """위치 기반 관련성 계산"""
        self.location_proximity = defaultdict(dict)
        all_locations = []
        
        for req in self.list_request:
            all_locations.append((f"{req.type}{req.id}", req.x, req.y))
        
        for term in self.list_terminal:
            all_locations.append((f"tm{term.id}", term.x, term.y))
            
        for cd in self.list_container_depot:
            all_locations.append((f"cd{cd.id}", cd.x, cd.y))
        
        for i, (loc_i, x_i, y_i) in enumerate(all_locations):
            for j, (loc_j, x_j, y_j) in enumerate(all_locations):
                if i != j:
                    dist = math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    self.location_proximity[loc_i][loc_j] = 1.0 / (dist + 1e-5)

    def _find_best_insertion_in_fleet(self, solution: Dict[str, List[Node]], req: Request, fleet_id: str, use_noise: bool = True) -> Tuple[float, int, int]:
        """
        특정 차량에 요청(P, D)을 최적 위치에 삽입하는 비용 계산
        개선: Hard Constraint 사전 스크리닝 및 Noise Insertion 반영
        """
        route = solution[fleet_id]
        if len(route) > 80:
            return float('inf'), -1, -1

        p_node = Node(req, 'P')
        d_node = Node(req, 'D')

        best_cost = float('inf')
        best_p, best_d = -1, -1
        fleet = next(f for f in self.list_fleet if f.id == fleet_id)

        L = len(route)
        for p in range(L + 1):
            for d in range(p + 1, L + 2):
                temp_route = list(route)
                temp_route.insert(p, p_node)
                temp_route.insert(d, d_node)

                # 1. 실시간 적재량 사전 검증
                load = 0
                feasible_load = True
                for n in temp_route:
                    if not n.is_bypassed:
                        load += n.demand_delta
                    if load > fleet.capacity or load < 0:
                        feasible_load = False
                        break
                if not feasible_load:
                    continue

                # 2. 실제 비용 평가
                solution[fleet_id] = temp_route
                cost, tv, cv = self.calculate_total_cost(solution)
                solution[fleet_id] = route  # 원복

                # 3. Hard Constraint Screening (제약 위반 또는 단절 아크 존재 시 탈락)
                if tv > 0 or cv > 0 or cost >= 10000:
                    continue

                # 4. Noise Insertion 적용 (지역 최적해 탈출용)
                if use_noise:
                    eval_cost = cost * (1.0 + random.uniform(-self.noise_level, self.noise_level))
                else:
                    eval_cost = cost

                if eval_cost < best_cost:
                    best_cost = eval_cost
                    best_p, best_d = p, d

        return best_cost, best_p, best_d

    def initial_solution(self) -> Dict[str, List[Node]]:
        """탐욕적 방식으로 P/D 노드가 교대 배치될 수 있는 초기 해 생성"""
        solution = {fleet.id: [] for fleet in self.list_fleet}
        unassigned = copy.deepcopy(self.list_request)
        
        while unassigned:
            best_req = None
            best_fleet = None
            best_p, best_d = -1, -1
            min_cost_increase = float('inf')

            for req in unassigned:
                for fleet in self.list_fleet:
                    cost, p_idx, d_idx = self._find_best_insertion_in_fleet(solution, req, fleet.id, use_noise=False)
                    if cost < min_cost_increase:
                        min_cost_increase = cost
                        best_req = req
                        best_fleet = fleet.id
                        best_p, best_d = p_idx, d_idx

            if best_req and best_fleet is not None:
                solution[best_fleet].insert(best_p, Node(best_req, 'P'))
                solution[best_fleet].insert(best_d, Node(best_req, 'D'))
                unassigned.remove(best_req)
            else:
                req = unassigned.pop(0)
                f_id = random.choice([f.id for f in self.list_fleet])
                solution[f_id].append(Node(req, 'P'))
                solution[f_id].append(Node(req, 'D'))

        return solution

    def calculate_total_cost(self, solution: Dict[str, List[Node]], debug: bool = False) -> Tuple[float, float, float]:
        """P/D 노드 순서에 따른 해의 총 비용 계산 (이동비용 + 대기비용 + 페널티)"""
        total_cost = 0.0
        time_violation = 0.0
        capacity_violation = 0.0
        
        for fleet_id, route in solution.items():
            fleet = next(f for f in self.list_fleet if f.id == fleet_id)
            current_time = fleet.e
            current_location = f'ts{fleet_id}'
            
            # 먼저 한 번 진행해서, 얼마나 차고지에서 늦게 출발할 수 있는지 파악
            max_possible_wait_time = float('inf')
            for node in route:
                arc_key = (current_location, current_time)
                if arc_key not in self.dict_arc or node.location_id not in self.dict_arc[arc_key]:
                    current_location = node.location_id
                    continue
                
                arc_info = self.dict_arc[arc_key][node.location_id]
                
                travel_time = arc_info[0]
                arrival_time = current_time + travel_time
                current_time = arrival_time + node.service_time

                possible_wait_time = node.l - current_time
                max_possible_wait_time = min(max_possible_wait_time, possible_wait_time)

                current_location = node.location_id
                
            # 진짜 cost 계산 시작
            current_time = fleet.e + max_possible_wait_time  # 차고지에서 늦게 출발할 수 있는 시간만큼 대기
            current_location = f'ts{fleet_id}'
            current_load = 0
            
            picked_up = set()
            
            for node in route:
                if node.node_type == 'P':
                    picked_up.add(node.request.id)
                    
                if node.is_bypassed:
                    continue
                
                if node.node_type == 'D':
                    if node.request.id not in picked_up:
                        total_cost += 100000
    
                arc_key = (current_location, current_time)
                
                if arc_key not in self.dict_arc or node.location_id not in self.dict_arc[arc_key]:
                    total_cost += 1000  # 아크 미존재 페널티
                    current_location = node.location_id
                    continue
    
                # 1. 이동 시간 및 이동 비용
                arc_info = self.dict_arc[arc_key][node.location_id]
                travel_time = arc_info[0]
                travel_cost = arc_info[1]
                total_cost += travel_cost
                
                arrival_time = current_time + travel_time
    
                # 2. 대기 시간(Wait Time) 및 대기 비용 계산 (조기 도착 시)
                wait_time = max(0, node.e - arrival_time)             
                waiting_cost = wait_time * self.cost_waiting
                total_cost += waiting_cost
    
                # 3. 서비스 시작 및 완료 시간
                start_service_time = arrival_time + max(0, node.e - arrival_time)  # 대기 시간 포함
                finish_time = start_service_time + node.service_time
                total_cost += node.service_time * self.cost_waiting
                
                # 4. 시간창 지연(Late Arrival) 페널티 계산
                if start_service_time > node.l and node.l > 0:
                    viol = start_service_time - node.l
                    time_violation += viol
                    total_cost += viol * self.time_window_penalty
    
                # 5. 용량 검증
                current_load += node.demand_delta
                if current_load > fleet.capacity:
                    cap_viol = current_load - fleet.capacity
                    capacity_violation += cap_viol
                    total_cost += cap_viol * self.capacity_penalty
                elif current_load < 0:
                    total_cost += 10000
    
                # 6. 다음 노드를 위한 시간/위치 갱신
                current_time = finish_time
                current_location = node.location_id
                
            # 7. 운행 종료 시 잔여 화물/공컨테이너 적재 시 페널티 (미하차 방지)
            if current_load != 0:
                total_cost += abs(current_load) * 100000
    
            # 차량 복귀 비용
            arc_key = (current_location, current_time)
            te_node = f'te{fleet_id}'
            if arc_key in self.dict_arc and te_node in self.dict_arc[arc_key]:
                total_cost += self.dict_arc[arc_key][te_node][1]
    
        return total_cost, time_violation, capacity_violation

    # ==================== Local Search (지역 탐색 모듈) ====================
    def local_search(self, solution: Dict[str, List[Node]]) -> Dict[str, List[Node]]:
        """
        Repair 후 수행되는 Local Search 다듬기 모듈
        1) Direct CE-SE Matching
        2) Container Depot Optimization
        3) Inter-route Relocate (차량 간 작업 재배치)
        """
        current_sol = copy.deepcopy(solution)
        current_cost, _, _ = self.calculate_total_cost(current_sol)
        improved = True

        while improved:
            improved = False

            # 1. Direct CE-SE Match 적용 시도
            cand_sol = self.operator_direct_ce_se_match(current_sol)
            cand_cost, tv, cv = self.calculate_total_cost(cand_sol)
            if cand_cost < current_cost and tv == 0 and cv == 0:
                current_sol = cand_sol
                current_cost = cand_cost
                improved = True
                continue

            # 2. Container Depot 변경 시도
            cand_sol = self.operator_change_container_depot(current_sol)
            cand_cost, tv, cv = self.calculate_total_cost(cand_sol)
            if cand_cost < current_cost and tv == 0 and cv == 0:
                current_sol = cand_sol
                current_cost = cand_cost
                improved = True
                continue

            # 3. Inter-route Relocate 시도
            cand_sol, rel_cost = self._ls_inter_route_relocate(current_sol, current_cost)
            if rel_cost < current_cost:
                current_sol = cand_sol
                current_cost = rel_cost
                improved = True
                continue

        return current_sol

    def _ls_inter_route_relocate(self, solution: Dict[str, List[Node]], current_cost: float) -> Tuple[Dict[str, List[Node]], float]:
        """차량 A의 요청 하나를 차량 B로 이관해보는 Local Search"""
        best_sol = copy.deepcopy(solution)
        best_cost = current_cost

        active_fleets = [f.id for f in self.list_fleet if len(solution[f.id]) > 0]
        if len(active_fleets) < 2:
            return best_sol, best_cost

        for _ in range(10):
            f_from, f_to = random.sample(self.list_fleet, 2)
            if not solution[f_from.id]:
                continue

            reqs_in_from = list({n.request for n in solution[f_from.id]})
            if not reqs_in_from:
                continue

            target_req = random.choice(reqs_in_from)

            # f_from에서 대상 요청 제거
            temp_sol = copy.deepcopy(solution)
            temp_sol[f_from.id] = [n for n in temp_sol[f_from.id] if n.request.id != target_req.id]

            # f_to 차량에 최적 위치 탐색
            ins_cost, p_idx, d_idx = self._find_best_insertion_in_fleet(temp_sol, target_req, f_to.id, use_noise=False)
            if ins_cost < float('inf') and p_idx != -1:
                temp_sol[f_to.id].insert(p_idx, Node(target_req, 'P'))
                temp_sol[f_to.id].insert(d_idx, Node(target_req, 'D'))

                eval_cost, tv, cv = self.calculate_total_cost(temp_sol)
                if eval_cost < best_cost and tv == 0 and cv == 0:
                    best_cost = eval_cost
                    best_sol = temp_sol

        return best_sol, best_cost

    # ==================== Container Depot Custom Operators ====================
    def operator_change_container_depot(self, solution: Dict[str, List[Node]]) -> Dict[str, List[Node]]:
        """ce, se 요청의 컨테이너 창고(terminal_index)를 다른 창고로 무작위 변경하는 오퍼레이터"""
        new_sol = copy.deepcopy(solution)
        if not self.list_container_depot or len(self.list_container_depot) < 2:
            return new_sol

        empty_reqs = [r for r in self.list_request if r.type in ['ce', 'se']]
        if not empty_reqs:
            return new_sol

        selected_req = random.choice(empty_reqs)
        current_cd = selected_req.terminal_index
        other_cds = [cd for cd in self.list_container_depot if cd.id != current_cd]
        if not other_cds:
            return new_sol

        new_cd = random.choice(other_cds)
        selected_req.terminal_index = new_cd.id

        for fleet_id, route in new_sol.items():
            for node in route:
                if node.req_id == selected_req.id:
                    if (node.request.type == 'ce' and node.node_type == 'D') or \
                       (node.request.type == 'se' and node.node_type == 'P'):
                        node.location_id = f"cd{new_cd.id}"

        return new_sol

    def operator_direct_ce_se_match(self, solution: Dict[str, List[Node]]) -> Dict[str, List[Node]]:
        """동일 경로 내 ce(공컨테이너 수거)와 se(공컨테이너 배송)를 창고 방문 없이 직접 연결(Direct Transfer)"""
        new_sol = copy.deepcopy(solution)
        
        for fleet_id, route in new_sol.items():
            ce_p_nodes = [n for n in route if n.request.type == 'ce' and n.node_type == 'P' and not n.is_bypassed]
            se_d_nodes = [n for n in route if n.request.type == 'se' and n.node_type == 'D' and not n.is_bypassed]

            if not ce_p_nodes or not se_d_nodes:
                continue

            for ce_p in ce_p_nodes:
                for se_d in se_d_nodes:
                    idx_ce_p = route.index(ce_p)
                    idx_se_d = route.index(se_d)

                    if idx_ce_p < idx_se_d:
                        ce_d_node = next((n for n in route if n.req_id == ce_p.req_id and n.node_type == 'D'), None)
                        se_p_node = next((n for n in route if n.req_id == se_d.req_id and n.node_type == 'P'), None)

                        if ce_d_node and se_p_node and not ce_d_node.is_bypassed and not se_p_node.is_bypassed:
                            ce_d_node.is_bypassed = True
                            se_p_node.is_bypassed = True
                            return new_sol
        return new_sol

    def operator_split_ce_se_match(self, solution: Dict[str, List[Node]]) -> Dict[str, List[Node]]:
        """Direct 연결되어 있던 ce-se 쌍을 해제하여 다시 컨테이너 창고를 방문하도록 복원"""
        new_sol = copy.deepcopy(solution)
        bypassed_nodes = [n for route in new_sol.values() for n in route if n.is_bypassed]
        
        if bypassed_nodes:
            target = random.choice(bypassed_nodes)
            for route in new_sol.values():
                for n in route:
                    if n.req_id == target.req_id:
                        n.is_bypassed = False
        return new_sol

    # ==================== Destroy Operators ====================
    def random_removal(self, solution: Dict[str, List[Node]], num_remove: int) -> Tuple[Dict[str, List[Node]], List[Request]]:
        destroyed = copy.deepcopy(solution)
        assigned_requests = list({n.request for route in solution.values() for n in route})
        removed_requests = []

        num_to_remove = min(num_remove, len(assigned_requests))
        if num_to_remove == 0:
            return destroyed, removed_requests

        selected_reqs = random.sample(assigned_requests, num_to_remove)
        for req in selected_reqs:
            for fleet_id in destroyed:
                destroyed[fleet_id] = [n for n in destroyed[fleet_id] if n.request.id != req.id]
            removed_requests.append(req)

        return destroyed, removed_requests

    def worst_removal(self, solution: Dict[str, List[Node]], num_remove: int) -> Tuple[Dict[str, List[Node]], List[Request]]:
        destroyed = copy.deepcopy(solution)
        assigned_requests = list({n.request for route in solution.values() for n in route})
        removed_requests = []

        if num_remove <= 0 or not assigned_requests:
            return destroyed, removed_requests

        base_cost, _, _ = self.calculate_total_cost(solution)
        request_removal_benefit = {}

        for req in assigned_requests:
            temp_sol = copy.deepcopy(solution)
            for fleet_id in temp_sol:
                temp_sol[fleet_id] = [n for n in temp_sol[fleet_id] if n.request.id != req.id]
            new_cost, _, _ = self.calculate_total_cost(temp_sol)
            request_removal_benefit[req] = base_cost - new_cost

        sorted_requests = sorted(request_removal_benefit.items(), key=lambda x: x[1], reverse=True)
        num_to_remove = min(num_remove, len(sorted_requests))

        for i in range(num_to_remove):
            req, _ = sorted_requests[i]
            for fleet_id in destroyed:
                destroyed[fleet_id] = [n for n in destroyed[fleet_id] if n.request.id != req.id]
            removed_requests.append(req)

        return destroyed, removed_requests

    def time_window_removal(self, solution: Dict[str, List[Node]], num_remove: int) -> Tuple[Dict[str, List[Node]], List[Request]]:
        destroyed = copy.deepcopy(solution)
        assigned_requests = list({n.request for route in solution.values() for n in route})
        removed_requests = []

        if num_remove <= 0 or not assigned_requests:
            return destroyed, removed_requests

        request_violations = defaultdict(float)

        for fleet_id, route in solution.items():
            fleet = next(f for f in self.list_fleet if f.id == fleet_id)
            current_time = fleet.e
            current_location = f'ts{fleet_id}'

            for node in route:
                if node.is_bypassed:
                    continue
                arc_key = (current_location, current_time)
                if arc_key in self.dict_arc and node.location_id in self.dict_arc[arc_key]:
                    travel_time = self.dict_arc[arc_key][node.location_id][0]
                    arrival_time = current_time + travel_time
                    start_service = max(arrival_time, node.e)
                    finish_time = start_service + node.service_time

                    violation = 0
                    if start_service > node.l:
                        violation = start_service - node.l
                    elif finish_time > node.l and node.l > 0:
                        violation = finish_time - node.l

                    request_violations[node.request] += violation
                    current_time = finish_time
                    current_location = node.location_id

        sorted_requests = sorted(request_violations.items(), key=lambda x: x[1], reverse=True)
        num_to_remove = min(num_remove, len(sorted_requests))

        for i in range(num_to_remove):
            req, _ = sorted_requests[i]
            for fleet_id in destroyed:
                destroyed[fleet_id] = [n for n in destroyed[fleet_id] if n.request.id != req.id]
            removed_requests.append(req)

        return destroyed, removed_requests

    def related_removal(self, solution: Dict[str, List[Node]], num_remove: int) -> Tuple[Dict[str, List[Node]], List[Request]]:
        destroyed = copy.deepcopy(solution)
        assigned_requests = list({n.request for route in solution.values() for n in route})
        removed_requests = []

        if num_remove <= 0 or not assigned_requests:
            return destroyed, removed_requests

        seed_req = random.choice(assigned_requests)

        request_relatedness = []
        for req in assigned_requests:
            dist = math.sqrt((seed_req.x - req.x)**2 + (seed_req.y - req.y)**2)
            spatial_rel = 1.0 / (dist + 1e-5)

            time_center_seed = (seed_req.e + seed_req.l) / 2
            time_center_req = (req.e + req.l) / 2
            time_rel = 1.0 / (1.0 + abs(time_center_seed - time_center_req))

            relatedness = self.location_related_factor * spatial_rel + (1 - self.location_related_factor) * time_rel
            request_relatedness.append((req, relatedness))

        sorted_requests = sorted(request_relatedness, key=lambda x: x[1], reverse=True)
        num_to_remove = min(num_remove, len(sorted_requests))

        for i in range(num_to_remove):
            req, _ = sorted_requests[i]
            for fleet_id in destroyed:
                destroyed[fleet_id] = [n for n in destroyed[fleet_id] if n.request.id != req.id]
            removed_requests.append(req)

        return destroyed, removed_requests

    # ==================== Repair Operators ====================
    def greedy_repair(self, destroyed: Dict[str, List[Node]], removed_requests: List[Request]) -> Dict[str, List[Node]]:
        solution = copy.deepcopy(destroyed)
        if not removed_requests:
            return solution

        for req in removed_requests:
            best_cost = float('inf')
            best_fleet = None
            best_p, best_d = -1, -1

            for fleet in self.list_fleet:
                cost, p_idx, d_idx = self._find_best_insertion_in_fleet(solution, req, fleet.id, use_noise=True)
                if cost < best_cost:
                    best_cost = cost
                    best_fleet = fleet.id
                    best_p, best_d = p_idx, d_idx

            if best_fleet is not None:
                solution[best_fleet].insert(best_p, Node(req, 'P'))
                solution[best_fleet].insert(best_d, Node(req, 'D'))
            else:
                f_id = random.choice([f.id for f in self.list_fleet])
                solution[f_id].append(Node(req, 'P'))
                solution[f_id].append(Node(req, 'D'))

        return solution

    def regret_repair(self, destroyed: Dict[str, List[Node]], removed_requests: List[Request], k=2) -> Dict[str, List[Node]]:
        solution = copy.deepcopy(destroyed)
        rem_reqs = copy.deepcopy(removed_requests)

        while rem_reqs:
            best_req = None
            best_regret = -float('inf')
            best_insertion_info = None

            for req in rem_reqs:
                fleet_costs = []
                for fleet in self.list_fleet:
                    b_cost, b_p, b_d = self._find_best_insertion_in_fleet(solution, req, fleet.id, use_noise=True)
                    if b_cost < float('inf'):
                        fleet_costs.append((b_cost, fleet.id, b_p, b_d))

                if not fleet_costs:
                    continue

                fleet_costs.sort(key=lambda x: x[0])
                
                if len(fleet_costs) >= k:
                    regret = sum(fc[0] for fc in fleet_costs[1:k]) - (k - 1) * fleet_costs[0][0]
                else:
                    regret = fleet_costs[0][0] * k

                if regret > best_regret:
                    best_regret = regret
                    best_req = req
                    best_insertion_info = fleet_costs[0]

            if best_req and best_insertion_info:
                _, f_id, b_p, b_d = best_insertion_info
                solution[f_id].insert(b_p, Node(best_req, 'P'))
                solution[f_id].insert(b_d, Node(best_req, 'D'))
                rem_reqs.remove(best_req)
            else:
                req = rem_reqs.pop()
                f_id = random.choice([f.id for f in self.list_fleet])
                solution[f_id].append(Node(req, 'P'))
                solution[f_id].append(Node(req, 'D'))

        return solution

    def simulated_annealing(self, current_cost: float, new_cost: float, temperature: float) -> bool:
        if new_cost < current_cost:
            return True
        return random.random() < math.exp((current_cost - new_cost) / temperature)

    def update_weights(self, decay=None):
        decay = decay or self.decay
        for op in self.destroy_ops:
            if self.usage['destroy'][op] > 0:
                reward = self.scores['destroy'][op] / self.usage['destroy'][op]
                self.weights['destroy'][op] = decay * self.weights['destroy'][op] + (1 - decay) * reward
        
        for op in self.repair_ops:
            if self.usage['repair'][op] > 0:
                reward = self.scores['repair'][op] / self.usage['repair'][op]
                self.weights['repair'][op] = decay * self.weights['repair'][op] + (1 - decay) * reward
        
        self.scores = {'destroy': defaultdict(float), 'repair': defaultdict(float)}
        self.usage = {'destroy': defaultdict(int), 'repair': defaultdict(int)}

    def build_route_nodes(self, solution: Dict[str, List[Node]]) -> Dict[str, List[str]]:
        """차량별 P/D 방문 순서를 문자열 리스트로 변환"""
        route_nodes = {}
        for fleet_id, route in solution.items():
            nodes = [f"ts{fleet_id}"]
            for n in route:
                if not n.is_bypassed:
                    nodes.append(f"{n.node_type}:{n.location_id}[req{n.req_id}]")
            nodes.append(f"te{fleet_id}")
            route_nodes[fleet_id] = nodes
        return route_nodes

    def solve(self) -> Tuple[Dict[str, List[Node]], float, List[float], List[float]]:
        current_solution = copy.deepcopy(self.best_solution)
        current_cost, _, _ = self.calculate_total_cost(current_solution)
        temperature = self.start_temp
        
        print(f"Starting Enhanced Interruptible PDP ALNS | Fleets: {len(self.list_fleet)}, Requests: {len(self.list_request)}")
        
        for iter in range(self.max_iter):
            # 1. Destroy 연산자 선택 및 수행
            destroy_op = random.choices(
                self.destroy_ops,
                weights=[self.weights['destroy'][op] for op in self.destroy_ops]
            )[0]
            
            repair_op = random.choices(
                self.repair_ops,
                weights=[self.weights['repair'][op] for op in self.repair_ops]
            )[0]
            
            num_remove = random.randint(self.destroy_min, self.destroy_max)
            
            if destroy_op == 'random':
                destroyed, removed = self.random_removal(current_solution, num_remove)
            elif destroy_op == 'worst':
                destroyed, removed = self.worst_removal(current_solution, num_remove)
            elif destroy_op == 'time_window':
                destroyed, removed = self.time_window_removal(current_solution, num_remove)
            else:
                destroyed, removed = self.related_removal(current_solution, num_remove)
            
            # 2. Repair 연산자 수행 (Noise 적용)
            if repair_op == 'greedy':
                new_solution = self.greedy_repair(destroyed, removed)
            else:
                new_solution = self.regret_repair(destroyed, removed)

            # 3. Local Search (지역 탐색) 수행 - 해 집중 개선
            new_solution = self.local_search(new_solution)

            # 4. 해 수용 여부 판정 (Simulated Annealing)
            new_cost, tv, cv = self.calculate_total_cost(new_solution)
            
            if self.simulated_annealing(current_cost, new_cost, temperature):
                current_solution = new_solution
                current_cost = new_cost
                
                if new_cost < self.best_cost:
                    self.best_solution = copy.deepcopy(new_solution)
                    self.best_cost = new_cost
                    self.scores['destroy'][destroy_op] += 3.0
                    self.scores['repair'][repair_op] += 3.0
                else:
                    self.scores['destroy'][destroy_op] += 1.0
                    self.scores['repair'][repair_op] += 1.0
                
                self.usage['destroy'][destroy_op] += 1
                self.usage['repair'][repair_op] += 1
            
            self.history.append(current_cost)
            self.history_best.append(self.best_cost)
            
            if iter > 0 and iter % 100 == 0:
                self.update_weights()
            
            temperature = max(self.min_temp, temperature * self.cooling_rate)
            
            if iter > 0 and iter % 50 == 0:
                print(f"Iter {iter:4d} | Temp: {temperature:7.2f} | "
                      f"Curr Cost: {current_cost:10.2f} | Best Cost: {self.best_cost:10.2f} | "
                      f"TV: {tv:7.2f} | CV: {cv:7.2f} | "
                      f"Ops: {destroy_op} + {repair_op}")
                
            # 변화없을 시 조기 종료
            if (len(self.history) > 1000) and (self.history_best[-1] == self.history_best[-1000]):
                print(f"No improvement for 1000 iterations. Early stopping at iteration {iter}.")
                break
        
        print(f"Optimization finished | Best cost: {self.best_cost:.2f}")
        return self.best_solution, self.best_cost, self.history, self.history_best


# %%
def read_requests(file_path: Path):
    list_request = []
    len_cf = 0
    len_sf = 0
    service_time_empty = 0

    with open(file_path, "r", encoding="utf-8-sig") as data:
        reader = csv.DictReader(data)
        for line in reader:
            r = Request()
            r.terminal_index = line["terminal index"].zfill(3)
            r.id = line["index"].zfill(3)
            r.x = float(line["x"])
            r.y = float(line["y"])
            r.e = int(line["e"])
            r.l = int(line["l"])
            r.e_t = int(line["terminal e"])
            r.l_t = int(line["terminal l"])
            r.demand = int(line["demand"])
            r.type = line["type"][0] + line["container"][0]
            r.service_time = int(line["service time"])
            r.service_time_t = int(line["terminal service time"])
            list_request.append(r)

            if r.type == "cf":
                len_cf += 1
            elif r.type == "sf":
                len_sf += 1
            if r.type in ["ce", "se"]:
                service_time_empty = r.service_time_t

    return list_request, len_cf, len_sf, int(round(service_time_empty))


def read_fleets(file_path: Path):
    list_fleet = []
    min_e = float("inf")
    max_l = 0

    with open(file_path, "r", encoding="utf-8-sig") as data:
        reader = csv.DictReader(data)
        for line in reader:
            f = Fleet()
            f.id = line["index"].zfill(3)
            f.x = float(line["x"])
            f.y = float(line["y"])
            f.e = int(line["e"])
            f.l = int(line["l"])
            f.capacity = int(line["capacity"])
            list_fleet.append(f)

            min_e = min(min_e, f.e)
            max_l = max(max_l, f.l)

    return list_fleet, min_e, max_l


def read_depot(file_path: Path):
    list_save = []
    with open(file_path, "r", encoding="utf-8-sig") as data:
        reader = csv.DictReader(data)
        for line in reader:
            d = Depot()
            d.id = line["index"].zfill(3)
            d.x = float(line["x"])
            d.y = float(line["y"])
            list_save.append(d)
    return list_save


def build_arc_dictionary(
    travel_time_file: Path,
    list_fleet,
    time_unit: int = 1,
    cost_transportation: float = 1.0,
    cost_waiting: float = 0.5,
):
    dict_arc = defaultdict(dict)

    def add_arc(origin, destination, t, tt, cost, distance):
        dict_arc[(origin, t)][destination] = [tt, cost, cost, cost, distance]

    with open(travel_time_file, "r", encoding="utf-8-sig") as data:
        reader = csv.DictReader(data)
        for line in reader:
            start = line["start"]
            end = line["end"]
            t = int(line["time"])
            distance = float(line.get("distance", 0))
            tt = int(round(float(line["travel time"]) / time_unit) * time_unit)
            tt = max(tt, time_unit)

            cost = tt * cost_transportation
            if start == end:
                cost = tt * cost_waiting

            if start.startswith("td") and end.startswith("td") and start[-3:] == end[-3:]:
                fleet_id = int(start[-3:])
                ts = f"ts{start[-3:]}"
                te = f"te{end[-3:]}"
                add_arc(ts, ts, t, tt, 0, 0)
                add_arc(ts, te, t, tt, 0, 0)
                add_arc(te, te, t, max(list_fleet[fleet_id].l - t, time_unit), 0, 0)
            elif start.startswith("td") and not end.startswith("td"):
                add_arc(f"ts{start[-3:]}", end, t, tt, cost, distance)
            elif not start.startswith("td") and end.startswith("td"):
                add_arc(start, f"te{end[-3:]}", t, tt, cost, distance)
            else:
                add_arc(start, end, t, tt, cost, distance)

    return dict_arc


def plot_solution(solution: Dict[str, List[Node]], list_request: List[Request], 
                  list_terminal: List[Depot], list_container_depot: List[Depot], 
                  list_fleet: List[Fleet], dir_forder_save: Path, save_name: str):
    
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. 모든 지점의 (location_id -> (x, y)) 좌표 룩업 딕셔너리 구축
    loc_coords = {}
    for fleet in list_fleet:
        loc_coords[f"ts{fleet.id}"] = (fleet.x, fleet.y)
        loc_coords[f"te{fleet.id}"] = (fleet.x, fleet.y)
    for terminal in list_terminal:
        loc_coords[f"tm{terminal.id}"] = (terminal.x, terminal.y)
    for cd in list_container_depot:
        loc_coords[f"cd{cd.id}"] = (cd.x, cd.y)
    for req in list_request:
        loc_coords[f"cf{req.id}"] = (req.x, req.y)
        loc_coords[f"sf{req.id}"] = (req.x, req.y)
        loc_coords[f"ce{req.id}"] = (req.x, req.y)
        loc_coords[f"se{req.id}"] = (req.x, req.y)

    # 2. 지점별 산점도 (Scatter Plot) 및 라벨링
    for req in list_request:
        ax.scatter(req.x, req.y, color='black', s=30, zorder=4)
        ax.annotate(f'{req.type}{req.id}', (req.x, req.y), xytext=(5, -8), textcoords='offset points', fontsize=8)

    for terminal in list_terminal:
        ax.scatter(terminal.x, terminal.y, color='blue', marker='s', s=80, zorder=5)
        ax.annotate(f'TM_{terminal.id}', (terminal.x, terminal.y), xytext=(-25, -15), textcoords='offset points', fontsize=9, color='blue', weight='bold')

    for cd in list_container_depot:
        ax.scatter(cd.x, cd.y, color='orange', marker='^', s=80, zorder=5)
        ax.annotate(f'CD_{cd.id}', (cd.x, cd.y), xytext=(8, -3), textcoords='offset points', fontsize=9, color='darkorange', weight='bold')

    for fleet in list_fleet:
        ax.scatter(fleet.x, fleet.y, color='gray', marker='o', s=60, zorder=4)
        ax.annotate(f'Depot_{fleet.id}', (fleet.x, fleet.y), xytext=(-30, -15), textcoords='offset points', fontsize=8, color='gray')

    # 3. 차량별 이동 경로 (Path) 그리기
    colors = plt.cm.tab10.colors
    
    for idx, fleet in enumerate(list_fleet):
        route = solution.get(fleet.id, [])
        if not route:
            continue

        x_coords = [fleet.x]
        y_coords = [fleet.y]

        for node in route:
            if node.is_bypassed:
                continue
            
            if node.location_id in loc_coords:
                nx, ny = loc_coords[node.location_id]
                x_coords.append(nx)
                y_coords.append(ny)

        x_coords.append(fleet.x)
        y_coords.append(fleet.y)

        if len(x_coords) > 2:
            color = colors[idx % len(colors)]
            ax.plot(x_coords, y_coords, color=color, linewidth=1.8, alpha=0.8, 
                    label=f'Fleet {fleet.id}', zorder=3)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Drayage ALNS Route Optimization Result", fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    plt.tight_layout()

    save_path = dir_forder_save / f"route-{save_name}.png"
    fig.savefig(save_path, dpi=300)
    print(f"Route plot saved to: {save_path}")
    #plt.show()
    #plt.close(fig)


def main():
    print("Loading data...")

    time_unit = 1
    cost_transportation = 1.0
    cost_waiting = 0.5

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    dir_forder_raw = project_dir / "data"
    files = [
        '4',
        #'10',
        #'20', '20-4', '20-5', '20-6', '20-7',
        #'30-1', '30-3', '30-4', '30-5', '30-9',
        #'40-1', '40-3', '40-6', '40-8', '40-9',
        #'50-1', '50-2', '50-4', '50-5', '50-6',
        #'60-1', '60-2', '60-3', '60-4', '60-5',
    ]

    for file_prefix in files:

        print(f"\nProcessing file prefix: {file_prefix}")

        dir_request = f"{file_prefix}_request.csv"
        dir_fleet = f"{file_prefix}_fleet.csv"
        dir_terminal = f"{file_prefix}_terminal.csv"
        dir_container_depot = f"{file_prefix}_container depot.csv"
        dir_travel_time = f"{file_prefix}_traveltime_modified.csv"
    
        dir_forder_save = project_dir / "results" / "ALNS"
        save_name = f"nspucc-{file_prefix}"
        dir_forder_save.mkdir(parents=True, exist_ok=True)
    
        list_request, len_cf, len_sf, service_time_empty = read_requests(dir_forder_raw / dir_request)
        list_fleet, min_e, max_l = read_fleets(dir_forder_raw / dir_fleet)
        list_terminal = read_depot(dir_forder_raw / dir_terminal)
        list_container_depot = read_depot(dir_forder_raw / dir_container_depot)
        dict_arc = build_arc_dictionary(
            dir_forder_raw / dir_travel_time,
            list_fleet,
            time_unit=time_unit,
            cost_transportation=cost_transportation,
            cost_waiting=cost_waiting,
        )
    
        print(
            f"Loaded requests={len(list_request)}, fleets={len(list_fleet)}, "
            f"terminals={len(list_terminal)}, depots={len(list_container_depot)}, arcs={len(dict_arc)}"
        )
    
        solver = DrayageALNSSolver(
            list_fleet,
            list_request,
            list_terminal,
            list_container_depot,
            dict_arc,
            time_unit=time_unit,
            cost_transportation=cost_transportation,
            cost_waiting=cost_waiting,
        )
        
        # 실행 시간 측정 시작
        start_time = time.time()
        best_solution, best_cost, history, history_best = solver.solve()
        best_route_nodes = solver.build_route_nodes(best_solution)
        execution_time = time.time() - start_time
        print(f"Execution Time: {execution_time:.2f} seconds")
    
        print("\nBest Solution (Interleaved Pickup and Delivery Sequences):")
        for fleet_id, route_nodes in best_route_nodes.items():
            if len(route_nodes) > 2:  # Depot 출발/복귀 외 방문 노드가 있는 차량만 출력
                print(f"\nFleet {fleet_id} Route:")
                print("  -> ".join(route_nodes))
    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history, "b-", linewidth=1)
        ax.set_title("ALNS Optimization Progress (Enhanced Interruptible PDP)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Total Cost")
        ax.grid(True)
        fig.savefig(dir_forder_save / f"ALNS-history-{save_name}-{time.time()}.png")
        plt.close(fig)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history_best, "r-", linewidth=1)
        ax.set_title("ALNS Optimization Progress (Enhanced Interruptible PDP)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Total Cost")
        ax.grid(True)
        fig.savefig(dir_forder_save / f"ALNS-history-best-{save_name}-{time.time()}.png")
        plt.close(fig)
    
        plot_solution(
            solution=best_solution,
            list_request=list_request,
            list_terminal=list_terminal,
            list_container_depot=list_container_depot,
            list_fleet=list_fleet,
            dir_forder_save=dir_forder_save,
            save_name=save_name
        )
        
        # 3. 상세 결과 및 요약 파일 저장 (신규 추가)
        result_summary = {
            "Best Cost": best_cost,
            "Execution Time (s)": execution_time,
            "Number of Iterations": len(history),
            "Number of Fleets": len(list_fleet),
            "Number of Requests": len(list_request),
            "Number of Terminals": len(list_terminal),
            "Number of Container Depots": len(list_container_depot),
        }
        with open(dir_forder_save / f"ALNS-summary-{save_name}-{time.time()}.txt", "w", newline="", encoding="utf-8-sig") as txtfile:
            writer = csv.writer(txtfile, delimiter='\t')
            writer.writerow(['Metric', 'Value'])
            for key, value in result_summary.items():
                writer.writerow([key, value])
            
            # 차량 경로
            for fleet_id, route_nodes in best_route_nodes.items():
                if len(route_nodes) > 2:  # Depot 출발/복귀 외 방문 노드가 있는 차량만 출력
                    writer.writerow([f"Fleet {fleet_id} Route:"])
                    writer.writerow(["  -> ".join(route_nodes)])

if __name__ == "__main__":
    main()