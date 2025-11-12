import networkx as nx
import numpy as np
import random
import pandas as pd
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import fisherz
from causalnex.structure.notears import from_pandas
from sklearn.metrics import precision_recall_fscore_support
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.Node import Node
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge

from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List
from causallearn.search.FCMBased import lingam



def check_multicollinearity(df):
    # 计算VIF，VIF值越大，共线性越严重（通常VIF>10视为高共线性）
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data.sort_values("VIF", ascending=False)

bio_vars = ['PTAU','ABETA42','TAU'] # 生物标志物
attri_vars = ['Sex', 'Age', 'PTEDUCAT', 'APOE4'] # 人口统计学变量
volume_vars = ['SegVentricles', 'WholeBrain', 'GreyMatter'] # 体积变量

import pdb

# ---------------------- 3. 多种因果发现算法运行 ----------------------
# 3.1 CausalLearn - FCI（约束型算法）

def apply_prior_knowledge(fci_nx_graph: nx.DiGraph, 
                         required_edges: list[tuple], 
                         forbidden_edges: list[tuple]) -> nx.DiGraph:
    """
    应用先验知识约束：
    - required_edges: 必须存在的边列表，如[(u, v), ...]表示u必须指向v
    - forbidden_edges: 禁止存在的边列表，如[(u, v), ...]表示u不能指向v
    """
    # 1. 移除所有禁止的边
    for u, v in forbidden_edges:
        if fci_nx_graph.has_edge(u, v):
            fci_nx_graph.remove_edge(u, v)
        # 处理无向/双向边的反向
        if fci_nx_graph.has_edge(v, u) and (v, u) in forbidden_edges:
            fci_nx_graph.remove_edge(v, u)
    
    # 2. 确保必须存在的边
    for u, v in required_edges:
        # 如果边不存在则添加
        if not fci_nx_graph.has_edge(u, v):
            fci_nx_graph.add_edge(u, v, type="required")
        # 确保方向正确（移除可能的反向边）
        if fci_nx_graph.has_edge(v, u):
            fci_nx_graph.remove_edge(v, u)
    
    return fci_nx_graph

def resolve_directions_to_dag(constrained_graph: nx.DiGraph) -> nx.DiGraph:
    """
    解决剩余边的方向，确保生成DAG：
    1. 优先保留有向边
    2. 对无向边，根据节点顺序或拓扑约束确定方向
    3. 移除双向边或转换为单向边（根据先验或领域知识）
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(constrained_graph.nodes())
    
    # 1. 先添加确定的有向边和必须存在的边
    directed_edges = [
        (u, v) for u, v, attr in constrained_graph.edges(data=True)
        if attr["type"] in ["directed", "required"]
    ]
    dag.add_edges_from(directed_edges)
    
    # 2. 处理无向边（尝试添加不形成环的方向）
    undirected_edges = [
        (u, v) for u, v, attr in constrained_graph.edges(data=True)
        if attr["type"] == "undirected" and u < v  # 避免重复处理
    ]
    
    for u, v in undirected_edges:
        # 检查添加u->v是否形成环
        if not nx.has_path(dag, v, u):
            dag.add_edge(u, v, type="resolved_undirected")
        # 检查添加v->u是否形成环
        elif not nx.has_path(dag, u, v):
            dag.add_edge(v, u, type="resolved_undirected")
        # 若双向都有路径（可能形成环），根据领域知识选择或移除
        else:
            # 这里简化处理，实际应根据领域知识决策
            print(f"警告：边 {u}-{v} 可能形成环，已移除")
    
    # 3. 处理部分有向边和双向边（根据实际场景调整）
    other_edges = [
        (u, v) for u, v, attr in constrained_graph.edges(data=True)
        if attr["type"] in ["partially_directed", "bidirectional"]
    ]
    
    for u, v in other_edges:
        # 双向边优先根据先验知识转换，否则移除
        if (u, v) not in directed_edges and (v, u) not in directed_edges:
            # 这里简化处理，实际应结合领域知识
            if not nx.has_path(dag, v, u):
                dag.add_edge(u, v, type="resolved_other")
    
    # 最终检查并移除环（如果仍存在）
    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        print(f"检测到环: {cycles}，尝试移除边以消除环")
        # 移除环中的一条边（简化处理）
        for cycle in cycles:
            u, v = cycle[0], cycle[1]
            if dag.has_edge(u, v):
                dag.remove_edge(u, v)
                break
    
    return dag

def run_fci(data, bk1=None, bk2=None):
    data_arr = data.values
    node_names = list(data.columns)
    if not (bk1 is None):
        G, edges = fci(data_arr, independence_test_method='fisherz', node_names=node_names, background_knowledge=bk1)  # 使用Fisher-Z检验
    else:
        G, edges = fci(data_arr, independence_test_method='fisherz', node_names=node_names)  # 使用Fisher-Z检验
    #if bk is not None:
    #    pdb.set_trace()
    #    orient_by_background_knowledge(G, background_knowledge=bk)
    
    # 转换为NetworkX图
    fci_graph = nx.DiGraph()
    fci_graph.add_nodes_from(node_names)
    
    ##    graph : a GeneralGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
    ##                graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
    ##                graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
     ##               graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
    
    for i in range(len(node_names)):
        for j in range(len(node_names)):

            val_ji = G.graph[j, i]
            val_ij = G.graph[i, j]
            u_name = node_names[i]
            v_name = node_names[j]
            
            # i --> j
            if val_ji == 1 and val_ij == -1:
                fci_graph.add_edge(u_name, v_name, type="directed")
            # i --- j（无向边）
            elif val_ij == -1 and val_ji == -1:
                fci_graph.add_edge(u_name, v_name, type="undirected")
                fci_graph.add_edge(v_name, u_name, type="undirected")
            # i <-> j（双向边，潜在混杂）
            elif val_ij == 1 and val_ji == 1:
                fci_graph.add_edge(u_name, v_name, type="bidirectional")
                fci_graph.add_edge(v_name, u_name, type="bidirectional")
            # i o-> j（部分有向边）
            elif val_ji == 1 and val_ij == 2:
                fci_graph.add_edge(u_name, v_name, type="partially_directed")
                fci_graph.add_edge(node_names[i], node_names[j])  # i -> j
        
    ## 得到部分有向图
    if bk2 is not None:
        bk_valid_e, bk_forbidden_e = bk2
        fci_graph = apply_prior_knowledge(fci_graph, required_edges=bk_valid_e, forbidden_edges=bk_forbidden_e)
    
    fci_graph = resolve_directions_to_dag(fci_graph)

    return fci_graph


# 3.2 CausalLearn - GES（评分型算法）
def run_ges(data, bk=None):
    data_arr = data.values
    node_names = list(data.columns)
    
    cg = ges(X=data_arr, node_names=node_names)  # 默认线性高斯评分
    # 转换为NetworkX图
    ges_graph = nx.DiGraph()
    ges_graph.add_nodes_from(node_names)
    ##    Record['G']: learned causal graph, where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicates  i --> j ,
    ##                Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.
    
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if cg['G'].graph[i, j] == 1 and cg['G'].graph[j, i] == -1:
                ges_graph.add_edge(node_names[i], node_names[j])  # i -> j
    
    #for u, v in cg.G.get_edges():
    #    ges_graph.add_edge(u, v)
    return ges_graph

# 3.3 CausalNex - NOTEARS（结构方程型算法）
def run_notears(data):
    sm = from_pandas(data)
    # 转换为NetworkX图
    notears_graph = nx.DiGraph()
    notears_graph.add_nodes_from(data.columns)
    for u, v, data in sm.edges(data=True):
        if data.get("weight", 0) != 0:  # 权重非0表示有边
            notears_graph.add_edge(u, v)
    return notears_graph

# 3.4 DirectLiNGAM

def run_lingam(data, bk=None):
    data_arr = data.values
    node_names = list(data.columns)
    bk_valid_e, bk_forbidden_e = bk
    prior_knowledge2 = np.zeros((len(node_names), len(node_names)), dtype=int)
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    #for u, v in bk_valid_e:
    #    prior_knowledge2[name_to_idx[u], name_to_idx[v]] = 1  # u -> v
    #for u, v in bk_forbidden_e:
    #    prior_knowledge2[name_to_idx[u], name_to_idx[v]] = -1  # u -/-> v
    
    #pdb.set_trace()
    model1 = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge2)
    model1.fit(data_arr)
    lingam_adj = model1.adjacency_matrix_.T  # 转置以匹配u->v表示法
    lingam_graph = nx.DiGraph()
    lingam_graph.add_nodes_from(node_names)
    #pdb.set_trace()
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if lingam_adj[i, j] != 0:
                lingam_graph.add_edge(node_names[i], node_names[j])
    return lingam_graph

# pc 算法
def run_pc(data, bk=None):
    """
    运行PC算法（基于约束的因果发现）
    :param data: pandas.DataFrame，输入数据（行：样本，列：变量）
    :param bk: tuple，先验知识 (必须边列表, 禁止边列表)，如 ([(u1,v1)], [(u2,v2)])
    :return: nx.DiGraph，因果图（u->v表示u是v的原因）
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    import networkx as nx

    # 1. 数据格式转换（causal-learn需numpy数组）
    data_arr = data.values
    node_names = list(data.columns)


    # 3. 运行PC算法（默认用Gaussian条件独立性检验，适用于连续数据）
    # 若为离散数据，可将test='gaussian'改为test='chi2'
    cg = pc(
        data_arr,
        test='gaussian',  # 检验方法（连续数据用gaussian，离散用chi2）
        alpha=0.05,       # 显著性水平（控制假阳性）
        background_knowledge=bk,
        node_names=node_names
    )

    # 4. 将causal-learn的CausalGraph转为nx.DiGraph
    pc_graph = nx.DiGraph()
    pc_graph.add_nodes_from(node_names)

    G = cg.G
    for i in range(len(node_names)):
        for j in range(len(node_names)):

            val_ji = G.graph[j, i]
            val_ij = G.graph[i, j]
            u_name = node_names[i]
            v_name = node_names[j]
            
            # i --> j
            if val_ji == 1 and val_ij == -1:
                pc_graph.add_edge(u_name, v_name, type="directed")
            # i --- j（无向边）
            elif val_ij == -1 and val_ji == -1:
                pc_graph.add_edge(u_name, v_name, type="undirected")
                pc_graph.add_edge(v_name, u_name, type="undirected")
            # i <-> j（双向边，潜在混杂）
            elif val_ij == 1 and val_ji == 1:
                pc_graph.add_edge(u_name, v_name, type="bidirectional")
                pc_graph.add_edge(v_name, u_name, type="bidirectional")
            # i o-> j（部分有向边）
            elif val_ji == 1 and val_ij == 2:
                pc_graph.add_edge(u_name, v_name, type="partially_directed")
                pc_graph.add_edge(node_names[i], node_names[j])  # i -> j

    return pc_graph

# ANM 算法
def run_anm(data, bk=None):
    """
    运行ANM（加性噪声模型，基于独立性检验的因果发现）
    :param data: pandas.DataFrame，输入数据（行：样本，列：变量）
    :param bk: tuple，先验知识 (必须边列表, 禁止边列表)
    :return: nx.DiGraph，因果图
    """
    from sklearn.ensemble import RandomForestRegressor
    from scipy.stats import pearsonr
    import numpy as np
    import networkx as nx

    # 1. 初始化参数
    node_names = list(data.columns)
    d = len(node_names)
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    data_arr = data.values
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(node_names)
    vadlid_e, forbidden_e = bk if bk is not None else ([], [])
    #pdb.set_trace()

    # 2. 定义ANM独立性检验函数（判断X是否为Y的原因）
    def is_cause(X, Y):
        """
        检验X→Y是否满足ANM假设：Y = f(X) + N，且N与X独立
        :param X: (n,) 候选原因变量
        :param Y: (n,) 候选结果变量
        :return: bool，True表示X→Y成立
        """
        # 步骤1：用随机森林拟合f(X)（非线性函数）
        X_reshaped = X.reshape(-1, 1)  # 转为2D输入（适配sklearn）
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_reshaped, Y)
        Y_pred = model.predict(X_reshaped)
        
        # 步骤2：计算噪声N = Y - Y_pred
        N = Y - Y_pred
        
        # 步骤3：检验N与X的独立性（皮尔逊相关系数，p值>0.05表示独立）
        corr, p_val = pearsonr(X, N)
        return p_val > 0.05  # p值>0.05：无法拒绝“N与X独立”的假设，即X→Y成立

    # 3. 遍历所有变量对，判断因果方向
    for i in range(d):
        for j in range(d):
            if i == j:
                continue  # 跳过自环
            
            u_name = node_names[j]  # 候选原因：u
            v_name = node_names[i]  # 候选结果：v
            u = data_arr[:, j]
            v = data_arr[:, i]

            # 应用先验知识：跳过禁止边，强制必须边
            forbidden = False
            required = False
            # 检查是否为禁止边（u->v禁止）
            if bk is not None:
                if (u_name, v_name) in forbidden_e:
                    forbidden = True
                # 检查是否为必须边（u->v必须）
                if (u_name, v_name) in vadlid_e:
                    required = True
            
            if forbidden:
                continue  # 禁止边：不添加
            if required:
                nx_graph.add_edge(u_name, v_name)
                continue  # 必须边：直接添加
            
            # 无先验知识：用ANM检验u→v是否成立
            if is_cause(u, v):
                # 进一步验证：v→u是否不成立（避免双向边）
                if not is_cause(v, u):
                    nx_graph.add_edge(u_name, v_name)

    return nx_graph

def ensemble_causal_graphs(
    graphs: List[nx.DiGraph],
    methods: List[str],
    vote_threshold: int = 2,
    node_check: bool = True,
    bk_valid_e: list[tuple] = [],
    bk_forbidden_e: list[tuple] = []
) -> nx.DiGraph:
    """
    基于投票机制集成多个因果图（nx.DiGraph）
    
    :param graphs: 输入的因果图列表，必须包含3个nx.DiGraph对象
    :param vote_threshold: 边保留的最小投票数（默认2：3个图中至少2个包含该边则保留）
    :param node_check: 是否强制所有图的节点集一致（默认True：若节点不一致则报错，避免边无对应节点）
    :return: 集成后的最终因果图（nx.DiGraph）
    :raises ValueError: 若输入图数量≠3，或节点集不一致（node_check=True时）
    """
    # -------------------------- 1. 输入合法性校验 --------------------------
    # 校验输入图数量（必须为3个，适配投票逻辑）
    #if len(graphs) != 3:
    #    raise ValueError(f"输入图数量必须为3，当前为{len(graphs)}")
    
    # 提取所有图的节点集
    node_sets = [set(graph.nodes()) for graph in graphs]
    # 校验节点集一致性（避免边对应节点不存在）
    if node_check and not all(ns == node_sets[0] for ns in node_sets):
        raise ValueError(
            f"所有输入图的节点集必须一致！\n"
            f"图1节点：{node_sets[0]}\n"
            f"图2节点：{node_sets[1]}\n"
            f"图3节点：{node_sets[2]}"
        )
    
    # -------------------------- 2. 收集所有候选边并统计投票 --------------------------
    # 候选边集：合并三个图中所有出现过的有向边（去重）
    all_candidate_edges = set()
    for graph in graphs:
        all_candidate_edges.update(graph.edges())  # 每个graph.edges()返回所有有向边(u, v)
    
    # 统计每条候选边的投票数
    edge_votes = {}
    edge_methods = {}
    for edge in all_candidate_edges:
        u, v = edge  # 有向边：u→v
        # 统计包含该边的图数量
        vote_count = sum(1 for graph in graphs if graph.has_edge(u, v))
        edge_votes[edge] = vote_count
        # 记录包含该边的算法名称
        edge_methods[edge] = [methods[i] for i, graph in enumerate(graphs) if graph.has_edge(u, v)]
    
    # -------------------------- 3. 生成最终集成图 --------------------------
    # 初始化最终图（节点集与输入图一致）
    final_graph = nx.DiGraph()
    final_graph.add_nodes_from(node_sets[0])  # 用第一个图的节点集（已校验一致）
    
    # 根据投票阈值添加边
    for edge, vote_count in edge_votes.items():
        if vote_count >= vote_threshold:
            final_graph.add_edge(*edge)  # 投票达标，保留该边
            print(f"边 {edge[0]}→{edge[1]}：{vote_count}票（≥{vote_threshold}票），保留; 来自算法：{', '.join(edge_methods[edge])}")
        else:
            print(f"边 {edge[0]}→{edge[1]}：{vote_count}票（<{vote_threshold}票），剔除; 来自算法：{', '.join(edge_methods[edge])}")
    
    # -------------------------- 4. 输出集成结果摘要 --------------------------
    total_candidates = len(all_candidate_edges)
    retained_edges = len(final_graph.edges())
    print(f"\n集成完成！候选边总数：{total_candidates}，保留边数：{retained_edges}，剔除边数：{total_candidates - retained_edges}")
    #print(f"最终边集1：{set([tuple(sorted(edge)) for edge in final_graph.edges])}")
    post_g = post_process_by_bk(final_graph, bk_valid_e, bk_forbidden_e)
    print(f"应用先验知识后，最终边数：{len(post_g.edges())}，剔除边数：{retained_edges - len(post_g.edges())}")
    #print(f"最终边集2：{set([tuple(sorted(edge)) for edge in post_g.edges])}")
    #quit()
    
    
    return post_g

# ---------------------- 4. 评估指标计算（Precision、Recall、F1） ----------------------
def evaluate_graph(pred_graph, gt_graph, method_name=""):
    # 提取边集合（统一为元组并排序，避免顺序影响）
    gt_edges = set(gt_graph.edges)
    pred_edges = set(pred_graph.edges)
    
    
    #print(method_name, pred_edges)
    
    
    # 真阳性（TP）、假阳性（FP）、假阴性（FN）
    tp = len(gt_edges & pred_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    
    
    # 计算Precision、Recall、F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1, tp, fp, fn

dir_path = '/home/yujiali/cf_mri_2/mia_added_comaparison_methods/causal_discovery'
#random_run_name = ['46', '735']

def get_prior_knowledge(nodes:list[str], Nodes:List[Node]):
    
    def get_time_point_and_variable_name(node_name):
        if node_name.endswith('T0'):
            return 'T0', node_name[:-3]
        elif node_name.endswith('T1'):
            return 'T1', node_name[:-3]
        else:
            return None, node_name  # 无时间点信息，返回None
    
    
    bk_valid_e = []
    bk_forbidden_e = []
    for node1 in nodes:
        for node2 in nodes:
            
            
            time1, vara_name1 = get_time_point_and_variable_name(node1)
            time2, vara_name2 = get_time_point_and_variable_name(node2)
            #print(node1, vara_name1, time1, vara_name2, time2)

            if time1 == 'T1' and time2 == 'T0':
                bk_forbidden_e.append((node1, node2))
            if time1 == 'T0' and time2 == 'T1':
                if vara_name1 == vara_name2:
                    bk_valid_e.append((node1, node2))
            
            if vara_name1 in volume_vars and (vara_name2 in attri_vars or vara_name2 in bio_vars):
                bk_forbidden_e.append((node1, node2))
    
    #pdb.set_trace()
    bk = BackgroundKnowledge()
    for node1 in Nodes:
        for node2 in Nodes:
            time1, vara_name1 = get_time_point_and_variable_name(node1.get_name())
            time2, vara_name2 = get_time_point_and_variable_name(node2.get_name())
            
            if time1 == 'T1' and time2 == 'T0':
                bk.add_forbidden_by_node(node1, node2)
            if time1 == 'T0' and time2 == 'T1':
                if vara_name1 == vara_name2:
                    bk.add_required_by_node(node1, node2)
            if vara_name1 in volume_vars and vara_name2 in bio_vars:
                bk.add_forbidden_by_node(node1, node2)
            if vara_name2 in attri_vars and (vara_name1 in attri_vars or vara_name1 in bio_vars):
                    bk.add_forbidden_by_node(node1, node2)
            
    
    return bk_valid_e, bk_forbidden_e, bk

def post_process_by_bk(graph: nx.DiGraph, bk_valid_e: list[tuple], bk_forbidden_e: list[tuple]) -> nx.DiGraph:

    """
    根据先验知识对因果图进行后处理：
    - 移除所有禁止的边
    - 确保必须存在的边
    :param graph: 输入的因果图（nx.DiGraph）
    :param bk_valid_e: 必须存在的边列表，如[(u, v), ...]表示u必须指向v
    :param bk_forbidden_e: 禁止存在的边列表，如[(u, v), ...]表示u不能指向v
    :return: 处理后的因果图（nx.DiGraph）
    """
    # 1. 移除所有禁止的边
    new_graph = graph.copy()
    for u, v in bk_forbidden_e:
        if new_graph.has_edge(u, v):
            new_graph.remove_edge(u, v)
            
            print(f"移除禁止边: {u} -> {v}")
            #print(f"原始边集: {graph.edges()}")
            #print(f"当前边集: {new_graph.edges()}")
            #print(set([tuple(sorted(edge)) for edge in graph.edges]))
            #print(set([tuple(sorted(edge)) for edge in new_graph.edges()]))
            #pdb.set_trace()
        
            #pdb.set_trace()
    # 2. 确保必须存在的边
    for u, v in bk_valid_e:
        # 如果边不存在则添加
        if not new_graph.has_edge(u, v):
            new_graph.add_edge(u, v)
            
        # 确保方向正确（移除可能的反向边）
        if new_graph.has_edge(v, u):
            new_graph.remove_edge(v, u)
            print(f"确保必须边: {u} -> {v}")
    
    return new_graph

import os
import argparse

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, default=None, help='data csv path')
    parser.add_argument('--output_dir', type=str, default=None, help='output dir path')
    args = parser.parse_args()
    
    
    data = pd.read_csv(args.data_csv)


    vif_result = check_multicollinearity(data)

    # 移除VIF极高的特征（例如VIF接近无穷大的特征）
    high_vif_feature = vif_result[vif_result["VIF"] > 100]["feature"].values
    if len(high_vif_feature) > 0:
        print(f"检测到高共线性特征: {high_vif_feature}")
        #quit()
        data = data.drop(columns=high_vif_feature)
    print(f"已移除高共线性特征: {high_vif_feature}")

    cg_without_background_knowledge = pc(data.values, 0.05, fisherz, True, 0, 0, node_names=data.columns.tolist())
    nodes = cg_without_background_knowledge.G.get_nodes()
    bk_valid_e, bk_forbidden_e, bk = get_prior_knowledge(nodes=data.columns.tolist(), Nodes=nodes)


    fci_graph = run_fci(data, bk1=bk, bk2=(bk_valid_e, bk_forbidden_e))
    ges_graph = run_ges(data, bk=None)
    notears_graph = run_notears(data)
    lingam_graph = run_lingam(data, bk=(bk_valid_e, bk_forbidden_e))
    pc_graph = run_pc(data, bk=bk)
    anm_graph = run_anm(data, bk=(bk_valid_e, bk_forbidden_e))
    
    ensemble_graph = ensemble_causal_graphs([fci_graph, ges_graph, lingam_graph], vote_threshold=1, node_check=True, methods=['FCI', 'GES', 'DirectLiNGAM'],
                                            bk_valid_e=bk_valid_e, bk_forbidden_e=bk_forbidden_e)
    
    ensemble_graph.saveosml(os.path.join(args.output_dir, 'final_ensemble_graph.osm'))
    
    


