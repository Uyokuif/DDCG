"""
DGAP with a Dual-Critic architecture, inspired by TD3.
This system uses a Critic-Ensemble to score agent actions.

- load_model(): Loads the CriticE and CriticQ models.
- calsco_dgap_v2(): Scores actions using the hierarchical D2C model.
- calsco_dual(), calsco(), score_search_dual(), score_search(): Compatibility wrappers.

Fusion Strategies for CriticQ ensemble:
- disagreement_dynamic: (Recommended) Dynamically weights critics based on the standard deviation of their scores. It trusts the conservative critic more when there is high disagreement.
- adaptive: Uses an optimistic strategy for critical actions and a conservative one for basic actions.
- conservative: TD3-style, takes the minimum score to avoid overestimation.
- average: Simple average of scores.
"""

import copy
import glob
import os, sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import re
import pdb
import pickle
import json
import random
from copy import deepcopy

from utils_bc import utils_interactive_eval
from utils_bc.utils_graph import filter_redundant_nodes
from envs.utils.check_logical import check_env_bug
from gpt_policy import GPTPolicy, split_goal
from sim_compute import Similarity
from memory_graph import MemoryGraph
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM , AutoModelForSequenceClassification, RobertaTokenizer
from safetensors.torch import load_file

# Dual DGAP system configuration
DUAL_DGAP_FUSION_STRATEGY = "disagreement_dynamic"  # Options: disagreement_dynamic, adaptive, conservative, average
DUAL_DGAP_DEBUG = False  # Whether to display dual model scoring details



def load_model():
    """Load DGAP-v2 trained models - CriticE + CriticQ Ensemble architecture"""
    try:
        from transformers import RobertaModel
        from safetensors.torch import load_file

        # DGAP-v2 model path, updated for full fine-tuned models
        model_dir = "/home/msj/planning/d2c/Discriminator/VirtualHome/roberta-base/models/full_ft"
        
        print(f"Loading DGAP-v2 models from: {model_dir}")
        
        class CriticE(torch.nn.Module):
            """Executability Critic - Binary Classification Task"""
            def __init__(self, model_name='roberta-base', dropout=0.3):
                super(CriticE, self).__init__()
                self.roberta = RobertaModel.from_pretrained(model_name)
                # Per training script, full fine-tuning was used.
                
                hidden_size = self.roberta.config.hidden_size
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size // 2, 1),
                    torch.nn.Sigmoid()
                )
            
            def forward(self, input_ids, attention_mask):
                outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                logits = self.classifier(pooled_output)
                
                class LogitsOutput:
                    def __init__(self, logits):
                        self.logits = logits.squeeze()
                return LogitsOutput(logits)
        
        class CriticQ(torch.nn.Module):
            """Quality Critic - Multi-architecture version based on the training script."""
            def __init__(self, model_name='roberta-base', dropout=0.3, critic_id=0):
                super(CriticQ, self).__init__()
                self.roberta = RobertaModel.from_pretrained(model_name)
                # Per training script, full fine-tuning was used.

                hidden_size = self.roberta.config.hidden_size
                
                # Dynamically build the regressor based on the critic_id from the training script
                if critic_id == 0:  # Conservative
                    self.regressor = torch.nn.Sequential(
                        torch.nn.Dropout(dropout + 0.1),
                        torch.nn.Linear(hidden_size, hidden_size // 2),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout + 0.1),
                        torch.nn.Linear(hidden_size // 2, 1),
                        torch.nn.Sigmoid()
                    )
                elif critic_id == 1:  # Aggressive
                    self.regressor = torch.nn.Sequential(
                        torch.nn.Dropout(max(0.1, dropout - 0.1)),
                        torch.nn.Linear(hidden_size, hidden_size // 2),
                        torch.nn.ELU(),
                        torch.nn.Dropout(max(0.1, dropout - 0.1)),
                        torch.nn.Linear(hidden_size // 2, hidden_size // 4),
                        torch.nn.ELU(),
                        torch.nn.Dropout(max(0.1, dropout - 0.1)),
                        torch.nn.Linear(hidden_size // 4, 1),
                        torch.nn.Sigmoid()
                    )
                else:  # Balanced (critic_id=2)
                    self.regressor = torch.nn.Sequential(
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_size, hidden_size // 2),
                        torch.nn.GELU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_size // 2, hidden_size // 4),
                        torch.nn.GELU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Linear(hidden_size // 4, 1),
                        torch.nn.Sigmoid()
                    )
            
            def forward(self, input_ids, attention_mask):
                outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                score = self.regressor(pooled_output)
                
                class LogitsOutput:
                    def __init__(self, logits):
                        self.logits = logits.squeeze()
                return LogitsOutput(score)
        
        # Load CriticE model
        print("Loading CriticE model...")
        critic_e = CriticE()
        critic_e_path = f"{model_dir}/critic_e/model.safetensors"
        if os.path.exists(critic_e_path):
            state_dict = load_file(critic_e_path, device="cpu")
            critic_e.load_state_dict(state_dict)
            print("CriticE model weights loaded successfully.")
        else:
            print(f"Warning: CriticE model file not found at {critic_e_path}. Using random initialization.")
        
        # Load CriticQ ensemble
        critic_q_ensemble = []
        critic_q_names = ['critic_q_conservative', 'critic_q_balanced', 'critic_q_aggressive']
        
        for name in critic_q_names:
            print(f"Loading {name} model...")
            
            # Determine critic_id based on name to build the correct architecture
            if 'conservative' in name:
                critic_id = 0
            elif 'aggressive' in name:
                critic_id = 1
            else: # 'balanced'
                critic_id = 2

            critic_q = CriticQ(critic_id=critic_id)
            critic_q_path = f"{model_dir}/{name}/model.safetensors"
            if os.path.exists(critic_q_path):
                state_dict = load_file(critic_q_path, device="cpu")
                critic_q.load_state_dict(state_dict)
                print(f"{name} weights loaded successfully.")
            else:
                print(f"Warning: {name} model file not found at {critic_q_path}. Using random initialization.")
            critic_q_ensemble.append(critic_q)
        
        # Load tokenizer
        print("Loading tokenizer...")
        try:
            tokenizer_path = f"{model_dir}/critic_e"
            if os.path.exists(tokenizer_path):
                roberta_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                print(f"Tokenizer loaded from {tokenizer_path}")
            else:
                print(f"Warning: Tokenizer not found in {tokenizer_path}. Falling back to roberta-base.")
                roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        except Exception as e:
            print(f"Error loading tokenizer: {e}. All model loading failed.")
            raise e

        # Set to evaluation mode
        critic_e.eval()
        for critic_q in critic_q_ensemble:
            critic_q.eval()
        
        print("DGAP-v2 model system loaded successfully (CriticE + CriticQ Ensemble).")
        
        return (critic_e, critic_q_ensemble), roberta_tokenizer
        
    except Exception as e:
        print(f"Fatal: DGAP-v2 model loading failed: {e}")
        print("Attempting to fall back to original RoBERTa model...")
        
        try:
            # This is a fallback and likely to fail if the path is incorrect.
            fallback_path = "/home/qhf/Virualhome/roberta/checkpoint-6100"
            print(f"Loading fallback model from: {fallback_path}")
            roberta_model = AutoModelForSequenceClassification.from_pretrained(fallback_path)
            roberta_tokenizer = AutoTokenizer.from_pretrained(fallback_path)
            print("Fallback to original RoBERTa model successful.")
            return roberta_model, roberta_tokenizer
        except Exception as fallback_e:
            print(f"Fatal: The fallback model at '{fallback_path}' also failed to load: {fallback_e}")
            print("The root cause is the DGAP-v2 loading failure above.")
            raise e


def calsco_dgap_v2(models, roberta_tokenizer, task_goal, recent_action, agent_action, fusion_strategy="disagreement_dynamic"):
    """
    DGAP-v2分层评分函数 - CriticE + CriticQ Ensemble架构
    
    Args:
        models: (CriticE, CriticQ_ensemble) 或单模型
        roberta_tokenizer: tokenizer
        task_goal: 任务目标
        recent_action: 历史动作
        agent_action: 当前动作
        fusion_strategy: 融合策略 ["adaptive", "conservative", "average"]
    
    Returns:
        最终评分 (int)
    """
    # 构建输入文本 (与原版完全一致)
    des = task_goal + ". "
    for action in recent_action:
        action = re.sub(r'\(\d+\)', '(1)', action)
        des += action + ", "
    des = des[:-2] + ". Action: " + agent_action
    
    inputs = roberta_tokenizer([des], return_tensors='pt')
    
    # 判断是DGAP-v2模型还是单模型
    if isinstance(models, tuple) and len(models) == 2:
        # DGAP-v2分层架构
        critic_e, critic_q_ensemble = models
        
        with torch.no_grad():
            # Stage 1: CriticE可执行性检查
            executability_output = critic_e(**inputs)
            # 自定义模型输出 (LogitsOutput) - 已经在模型内部处理了sigmoid
            executability_score = executability_output.logits.item() if executability_output.logits.dim() == 0 else executability_output.logits[0].item()
            
            # 如果不可执行(< 0.5)，返回低分
            if executability_score < 0.5:
                if DUAL_DGAP_DEBUG:
                    print(f"🔧 DGAP-v2: 不可执行 (CriticE: {executability_score:.3f}) -> 1")
                return 1
            
            # Stage 2: CriticQ集成质量评估
            quality_scores = []
            for critic_q in critic_q_ensemble:
                quality_output = critic_q(**inputs)
                # 自定义模型输出 (LogitsOutput) - 已经在模型内部处理了sigmoid
                quality_score = quality_output.logits.item() if quality_output.logits.dim() == 0 else quality_output.logits[0].item()
                quality_scores.append(quality_score)
            
            # 根据融合策略处理CriticQ集成
            if fusion_strategy == "disagreement_dynamic":
                # Disagreement-Aware Dynamic Weighting Algorithm
                # Step 1: Get scores (order: conservative, balanced, aggressive)
                q_c, q_b, q_a = quality_scores[0], quality_scores[1], quality_scores[2]
                
                # Step 2: Quantify Disagreement
                disagreement = np.std([q_c, q_b, q_a])
                
                # Step 3: Calculate Dynamic Weights
                k = 2.0  # Hyperparameter to scale disagreement effect
                alpha = k * disagreement
                
                base_s_c, base_s_b, base_s_a = 1.0, 0.5, 0.0 # Base personality scores
                
                dynamic_s_c = base_s_c + alpha
                dynamic_s_b = base_s_b
                dynamic_s_a = base_s_a - alpha
                
                # Use softmax to get normalized weights
                dynamic_scores = torch.tensor([dynamic_s_c, dynamic_s_b, dynamic_s_a])
                weights = F.softmax(dynamic_scores, dim=0)
                
                # Step 4: Calculate final weighted score
                ensemble_quality = torch.dot(weights, torch.tensor(quality_scores)).item()
            elif fusion_strategy == "conservative":
                ensemble_quality = min(quality_scores)
            elif fusion_strategy == "average":
                ensemble_quality = sum(quality_scores) / len(quality_scores)
            elif fusion_strategy == "adaptive":
                critical_actions = ['putin', 'switchon', 'open', 'close']
                is_critical = any(keyword in agent_action.lower() for keyword in critical_actions)
                ensemble_quality = max(quality_scores) if is_critical else (sum(quality_scores) / len(quality_scores))
            else:
                ensemble_quality = sum(quality_scores) / len(quality_scores)
            
            # 转换到DGAP评分范围[1-10]  
            # 训练时: score/10.0 -> [0,1], 推理时: [0,1]*10 -> [0,10], 然后clamp到[1,10]
            final_score = ensemble_quality * 10.0  # [0,1] -> [0,10]
            final_score = max(1.0, min(10.0, final_score))  # clamp到[1,10]
            
            # 调试信息输出
            if DUAL_DGAP_DEBUG:
                print(f"🔧 DGAP-v2: CriticE={executability_score:.3f}, "
                      f"CriticQ={quality_scores} -> {ensemble_quality:.3f} -> {final_score:.1f} ({fusion_strategy})")
        
    else:
        # 单模型降级
        with torch.no_grad():
            final_score = models(**inputs).logits[0][0].item()
    
    return round(final_score)


def calsco_dual(models, roberta_tokenizer, task_goal, recent_action, agent_action, fusion_strategy="adaptive"):
    """
    兼容函数：自动检测模型类型并调用对应函数
    """
    # 检测是否为DGAP-v2模型结构
    if isinstance(models, tuple) and len(models) == 2:
        first_model, second_model = models
        if isinstance(second_model, list):  # CriticQ ensemble是列表
            return calsco_dgap_v2(models, roberta_tokenizer, task_goal, recent_action, agent_action, fusion_strategy)
        else:
            # 原始双模型结构，保持原逻辑
            pass
    
    # 原始双DGAP逻辑 (保持不变)
    des = task_goal + ". "
    for action in recent_action:
        action = re.sub(r'\(\d+\)', '(1)', action)
        des += action + ", "
    des = des[:-2] + ". Action: " + agent_action
    
    inputs = roberta_tokenizer([des], return_tensors='pt')
    
    # 判断是双模型还是单模型
    if isinstance(models, tuple) and len(models) == 2:
        # 双模型评分
        model1, model2 = models
        
        with torch.no_grad():
            score1 = model1(**inputs).logits[0][0].item()
            score2 = model2(**inputs).logits[0][0].item()
        
        # 融合策略
        if fusion_strategy == "conservative":
            # TD3风格：取最小值，避免过估计
            final_score = min(score1, score2)
        elif fusion_strategy == "average":
            # 简单平均
            final_score = (score1 + score2) / 2
        elif fusion_strategy == "adaptive":
            # 🔥 自适应策略 - 最佳表现
            critical_actions = ['putin', 'switchon', 'open', 'close']
            if any(keyword in agent_action.lower() for keyword in critical_actions):
                # 关键动作：采用乐观策略
                final_score = max(score1, score2)
            else:
                # 基础动作：采用保守策略
                final_score = min(score1, score2)
        else:
            final_score = (score1 + score2) / 2
            
        # 调试信息输出
        if DUAL_DGAP_DEBUG:
            print(f"🔧 双DGAP: {score1:.1f}, {score2:.1f} -> {final_score:.1f} ({fusion_strategy})")
        
    else:
        # 单模型降级
        with torch.no_grad():
            final_score = models(**inputs).logits[0][0].item()
    
    return round(final_score)


# 向后兼容：保留原始函数名
def calsco(roberta_model, roberta_tokenizer, task_goal, recent_action, agent_action):
    """原始calsco函数 - 向后兼容"""
    if isinstance(roberta_model, tuple):
        # 如果传入的是双模型，使用双DGAP
        return calsco_dual(roberta_model, roberta_tokenizer, task_goal, recent_action, agent_action)
    else:
        # 原始单模型逻辑
        des = task_goal + ". "
        for action in recent_action:
            action = re.sub(r'\(\d+\)', '(1)', action)
            des += action + ", "
        des = des[:-2] + ". Action: " + agent_action
        inputs = roberta_tokenizer([des], return_tensors='pt')
        score = roberta_model(**inputs).logits[0][0].item()
        return round(score)


def score_search_dual(models, roberta_tokenizer, task_goal, action_list, fusion_strategy="adaptive"):
    """
    双模型批量评分函数 - 支持DGAP-v2分层架构
    
    Args:
        models: (CriticE, CriticQ_ensemble) 或 (model1, model2) 或单模型
        roberta_tokenizer: tokenizer
        task_goal: 任务目标
        action_list: [(action, recent_actions), ...]
        fusion_strategy: 融合策略
    
    Returns:
        scores: 评分列表
    """
    # 检测是否为DGAP-v2模型结构
    if isinstance(models, tuple) and len(models) == 2:
        first_model, second_model = models
        if isinstance(second_model, list):  # CriticQ ensemble是列表
            # 使用DGAP-v2评分
            scores = []
            for action, recent_actions in action_list:
                score = calsco_dgap_v2(models, roberta_tokenizer, task_goal, recent_actions, action, fusion_strategy)
                scores.append(score)
            return scores
    
    # 原始双DGAP批量评分逻辑
    scores = []
    for action, recent_actions in action_list:
        score = calsco_dual(models, roberta_tokenizer, task_goal, recent_actions, action, fusion_strategy)
        scores.append(score)
    return scores


# 向后兼容：保留原始函数名
def score_search(roberta_model, roberta_tokenizer, lid_goals, recent_action, curr_goal, env_graph):
    """原始score_search函数 - 向后兼容"""
    # 原始单模型逻辑（不管是否为双模型，都使用相同的搜索逻辑）
    test_objs = [{"name": node['class_name'], "id": node['id']} for node in env_graph['nodes']]
    actions = ['walk', 'find', 'open', 'grab', 'close', 'switchon']
    exec_action_lists = []
    for obj in test_objs:
        for action in actions:
            if action == 'find':  # 将 'find' 动作转换为 'walk'
                action = 'walk'
            action_script = "[{}] <{}> ({})".format(action, obj['name'], obj['id'])
            exec_action_lists.append(action_script)
    actions = ['putback', 'putin']
    for i in range(len(test_objs)):
        for j in range(len(test_objs)):
            if i != j:  # 确保不是同一个物体
                for action in actions:
                    item1 = test_objs[i]
                    item2 = test_objs[j]
                    action_script = "[{}] <{}> ({}) <{}> ({})".format(
                        action, item1['name'], item1['id'], item2['name'], item2['id'])
                    exec_action_lists.append(action_script)
    mem_graph = MemoryGraph(None)
    mem_graph.set_graph(env_graph)
    
    des = curr_goal + ". "
    for action in recent_action:
        action = re.sub(r'\(\d+\)', '(1)', action)
        des += action + ", "
    des = des[:-2] + ". Action: " 
    action_scores = []

    for i in range(len(exec_action_lists)):
        inputs = roberta_tokenizer([des + exec_action_lists[i]], return_tensors='pt')
        
        # 根据模型类型计算分数
        if isinstance(roberta_model, tuple):
            # 双模型：使用calsco_dual
            score = calsco_dual(roberta_model, roberta_tokenizer, curr_goal, recent_action, exec_action_lists[i])
        else:
            # 单模型
            outputs = roberta_model(**inputs)
            score = outputs.logits[0][0].item()  
        action_scores.append((score, exec_action_lists[i]))

    top_20_actions = sorted(action_scores, key=lambda x: x[0], reverse=True)[:20]

    for score, action in top_20_actions:
        if mem_graph.simulate_action(action) is True:
            break
    return round(score), action

    
    
    
    
    
    
    
    


def sample_model_action(args, action_logits, object_logits, resampling, obs, agent_id, type='multinomial'):
    if type == 'argmax':
        agent_action = int(action_logits.argmax())
        agent_obj = int(object_logits.argmax())
    elif type == 'multinomial':
        action_dist = torch.distributions.Multinomial(logits=action_logits, total_count=1)
        obj_dist = torch.distributions.Multinomial(logits=object_logits, total_count=1)
        agent_action = int(torch.argmax(action_dist.sample(), dim=-1))
        agent_obj = int(torch.argmax(obj_dist.sample(), dim=-1))
    elif type == 'multinomial_random':
        p = random.uniform(0, 1)
        if p < args.model_exploration_p:

            count = 0
            while 1:

                if resampling == -1 and count == 0:
                    agent_action = int(torch.argmax(action_logits))
                else:
                    agent_action = int(torch.multinomial(action_logits, 1))

                ## randomly select an action if stuck at a single action
                if count > 50 or resampling > 50:
                    agent_action = random.choice(list(args.vocabulary_action_name_word_index_dict.values()))

                object_logits_tem = deepcopy(object_logits)

                if agent_action == args.vocabulary_action_name_word_index_dict['none']:
                    agent_obj = None
                else:
                    agent_obj_space, agent_obj = utils_interactive_eval.get_valid_action_space(args, agent_action, obs,
                                                                                               agent_id)

                    if agent_obj_space is not None:
                        not_agent_obj_space = [idx for idx in list(range(object_logits_tem.shape[1])) if
                                               idx not in agent_obj_space]
                        object_logits_tem[0][torch.tensor(not_agent_obj_space)] = -99999
                        object_logits_tem = F.softmax(object_logits_tem, -1)

                        if resampling == -1 and count == 0:
                            agent_obj = int(torch.argmax(object_logits_tem))
                        else:
                            agent_obj = int(torch.multinomial(object_logits_tem, 1))

                        assert agent_obj in agent_obj_space
                        break

                count += 1
        else:
            count = 0
            while 1:
                action_logits_uniform = torch.ones_like(action_logits) / action_logits.shape[1]
                agent_action = int(torch.multinomial(action_logits_uniform, 1))
                count += 1

                if agent_action == args.vocabulary_action_name_word_index_dict['none']:
                    agent_obj = None
                else:
                    agent_obj_space, agent_obj = utils_interactive_eval.get_valid_action_space(args, agent_action, obs,
                                                                                               agent_id)

                if agent_obj is not None:
                    break

    agent_action = args.vocabulary_action_name_index_word_dict[agent_action]
    resampling += 1
    return agent_action, agent_obj, resampling


def sample_action(args, obs, agent_id, action_logits, object_logits, all_actions, all_cur_observation, logging):
    graph_nodes = obs[agent_id]['nodes']
    agent_action = None
    agent_obj = None
    valid_action = False
    resampling = -1
    sample_model_action_type = 'multinomial_random'

    while 1:
        if agent_action == None or agent_obj == None or agent_obj >= len(graph_nodes):
            agent_action, agent_obj, resampling = sample_model_action(args, action_logits, object_logits, resampling,
                                                                      obs, agent_id, type=sample_model_action_type)
        else:
            selected_node = graph_nodes[agent_obj]

            print(agent_action, selected_node['class_name'])
            action_obj_str, bad_action_flag = utils_interactive_eval.can_perform_action(agent_action,
                                                                                        o1=selected_node['class_name'],
                                                                                        o1_id=selected_node['id'],
                                                                                        agent_id=agent_id + 1,
                                                                                        graph=obs[agent_id],
                                                                                        teleport=True)

            bad_action_flag_v2, ignore_walk = utils_interactive_eval.check_logical_before_unity(agent_id,
                                                                                                cur_action=action_obj_str,
                                                                                                actions_sofar=all_actions,
                                                                                                observations_sofar=all_cur_observation,
                                                                                                logging=logging,
                                                                                                verbose=False)

            if bad_action_flag or bad_action_flag_v2 or ignore_walk:
                agent_action, agent_obj, resampling = sample_model_action(args, action_logits, object_logits,
                                                                          resampling, obs, agent_id,
                                                                          type=sample_model_action_type)
            else:
                valid_action = True
                break

    if not valid_action:
        ignore_walk = False
        action_obj_str = None

    return action_obj_str, ignore_walk, resampling


def compute_task_complexity(task_goal, graph):
    min_steps = 0
    for goal in task_goal:
        goal_num = task_goal[goal]
        # print(goal, goal_num)
        if 'close' in goal:
            min_steps += 1
        elif 'turn' in goal:
            min_steps += 1
        elif 'inside' in goal:
            obj_name = goal.split('_')[1]
            obj_num = goal_num
            inside_num = 0
            out_num = 0
            # pan duan obj wei zhi
            for node in graph['nodes']:
                if node['class_name'] == obj_name:
                    obj_id = node['id']
                    from_obj_edges = [edge for edge in graph['edges'] if edge['from_id'] == obj_id]
                    for edge in reversed(from_obj_edges):
                        if edge['relation_type'] == 'INSIDE':
                            inside_num += 1
                            break
                        elif edge['relation_type'] == 'ON':
                            out_num += 1
                            break
            if obj_num <= out_num:
                min_steps += 4 * goal_num
            else:
                min_steps += 4 * out_num + 5 * (obj_num - out_num)
            min_steps = min_steps + 1 + obj_num
        elif 'on' in goal:
            obj_name = goal.split('_')[1]
            obj_num = goal_num
            inside_num = 0
            out_num = 0
            # pan duan obj wei zhi
            for node in graph['nodes']:
                if node['class_name'] == obj_name:
                    obj_id = node['id']
                    from_obj_edges = [edge for edge in graph['edges'] if edge['from_id'] == obj_id]
                    for edge in reversed(from_obj_edges):
                        print(edge)
                        if edge['relation_type'] == 'INSIDE':
                            inside_num += 1
                            break
                        elif edge['relation_type'] == 'ON':
                            out_num += 1
                            break
            if obj_num <= out_num:
                min_steps += 4 * obj_num
            else:
                min_steps += 4 * out_num + 5 * (obj_num - out_num)
            min_steps = min_steps + obj_num
    return min_steps


def interactive_interface_fn(args, vh_envs, iteri, agent_model, data_info, logging, tokenizer):
    # control flags
    if_gpt = True
    if_exe_all_action = True
    verbose = True
    valid_run = 0
    success_count = 0
    save_output = []
    camera_num = vh_envs.comm.camera_count()[1]
    save_data_all = []
    if_script = False
    roberta_model, roberta_tokenizer = load_model()
    
    # 🎲 随机取样改进：生成随机任务ID列表
    # 假设数据集有足够多的任务，我们随机选择test_examples个任务
    import random
    random.seed(args.seed)  # 使用命令行指定的随机种子
    
    # 获取数据集总任务数（这里假设有足够大的范围，实际应该从数据集获取）
    total_tasks = 1000  # 可以根据实际数据集大小调整
    if hasattr(vh_envs, 'num_tasks'):
        total_tasks = vh_envs.num_tasks
    
    # 随机选择task_examples个任务ID
    selected_task_ids = random.sample(range(1, total_tasks + 1), min(args.test_examples, total_tasks))
    logging.info(f"🎲 随机选择的任务ID: {selected_task_ids}")
    
    # 原始顺序取样代码（已注释）
    # i = 0
    # while 1:
    #     i += 1
    #     print(f'Running test case {i}. Total valid runs: {valid_run}')
    #     
    #     if i > args.test_examples:
    #         break
    #     task_id = i  # 原始：顺序取样
    
    # 🎲 新的随机取样循环
    for idx, task_id in enumerate(selected_task_ids):
        current_test = idx + 1
        print(f'Running test case {current_test}/{args.test_examples} (Task ID: {task_id}). Total valid runs: {valid_run}')
        
        all_cur_observation = []
        all_actions = []
        all_rewards = []
        all_frames = []

        if True:
            obs, env_graph = vh_envs.reset(task_id=task_id)
            obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
            all_cur_observation.append(deepcopy(obs))

            steps = 0
            valid_run_tem = False
            success_run_tem = False

            if if_script:
                with open('script.txt', 'r') as file:
                    script = file.readlines()
                    exe_index = 0
            
            if if_gpt:
                gpt_policy = GPTPolicy(logging)
                gpt_policy.set_graph(env_graph)
                gpt_policy.set_goal(vh_envs.task_goal[0])
                if if_exe_all_action:
                    gpt_policy.generate_recurrent_plan()
                else:
                    gpt_policy.split_task_goal, gpt_policy.split_task_goal_num = split_goal(logging, gpt_policy.task_goal)

            while (1):
                if verbose:
                    logging.info('-' * 100)
                recent_action = []
                recent_dis = []
                agent_id = 0
                agent_actions = {}
                agent_rewards = {}
                agent_ignore_walk = {}
                ignore_walk = None

                action_obj_str = ''
                if if_gpt:
                    if if_exe_all_action:
                        gpt_action_obj_str = gpt_policy.get_action_from_chatgpt()
                        if gpt_action_obj_str:
                            logging.info(f'[INFO] GPT predicted action: {gpt_action_obj_str}')
                    else:
                        gpt_action_obj_str = gpt_policy.get_action_from_chatgpt()
                        if gpt_action_obj_str == '':
                            if gpt_policy.goal_exe_index < gpt_policy.split_task_goal_num:
                                current_task = gpt_policy.split_task_goal[gpt_policy.goal_exe_index]
                                gpt_policy.goal_exe_index += 1
                                gpt_policy.generate_plan(current_task, roberta_model, roberta_tokenizer)
                            gpt_action_obj_str = gpt_policy.get_action_from_chatgpt()
                    action_obj_str = gpt_action_obj_str
                
                if if_script:
                    action_obj_str = script[exe_index]
                    exe_index += 1

                agent_actions[agent_id] = action_obj_str
                agent_ignore_walk[agent_id] = ignore_walk

                if if_gpt and 'gpt_policy' in locals():
                    task_goal_str = gpt_policy.task_goal.split("(id:")[0]
                else:
                    task_goal_str = str(vh_envs.task_goal[0])
                
                t_score = calsco(roberta_model, roberta_tokenizer, task_goal_str, recent_action, agent_actions[0])
                
                # The score from D2C is [1, 10]. A low score indicates a need for search.
                # A threshold of 4 was used in the original DGAP paper.
                if t_score < 4:
                    logging.info(f"Low score ({t_score}), initiating search...")
                    t_score, agent_actions[0] = score_search(roberta_model, roberta_tokenizer, vh_envs.task_goal[0], recent_action, task_goal_str, env_graph)
                
                recent_dis.append(t_score)
                recent_action.append(agent_actions[0])
                obs, rewards, dones, infos, success = vh_envs.step(agent_actions, ignore_walk=agent_ignore_walk, logging=logging)

                if rewards is None:
                    logging.error('Interactive eval: Unity action failed!')
                    logging.error(f'[INFO] Failed reason: {json.dumps(obs)}')
                    valid_run_tem = False
                    break

                obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
                if not check_env_bug(agent_actions[0], obs[0], agent_i=0, logging=logging):
                    logging.error('Interactive eval: check_env_bug outside unity failed!')
                    valid_run_tem = False
                    break

                reward = torch.tensor(rewards)
                if reward[0] is not None:
                    agent_rewards[0] = reward[0]

                all_cur_observation.append(deepcopy(obs))
                all_actions.append(deepcopy(agent_actions))
                all_rewards.append(deepcopy(agent_rewards))

                if verbose:
                    env_task_goal_write = [f'{k}_{v}' for k, v in vh_envs.task_goal[0].items() if v > 0]
                    logging.info(f'Example {current_test} (Task ID: {task_id}), Step {steps}, Goal: {env_task_goal_write}')
                    logging.info(f'  Action: {agent_actions[0]}')
                    logging.info(f'  Reward: {agent_rewards.get(0)}')
                    if agent_actions[0] is not None:
                        logging.info(f'  Ignore Walk: {agent_ignore_walk.get(0)}')

                steps += 1
                if np.any(dones):
                    valid_run_tem = True
                    if infos[0]['is_success']:
                        success_run_tem = True
                    break

            if valid_run_tem:
                valid_run += 1
                for tem in all_actions:
                    logging.info(f"Final action sequence: {tem}")
                if success_run_tem:
                    success_count += 1
                    print('-' * 50)
                    print('----> SUCCESS <----')
                    print('-' * 50)

    # 🎲 随机取样完成后的最终统计
    if args.interactive_eval:
        total_tested = len(selected_task_ids)
        sr = 100. * success_count / total_tested if total_tested != 0 else 0
        exec_rate = 100. * valid_run / total_tested if total_tested != 0 else 0
        log_message = (f"🎲 Random Sampling Evaluation Summary for {args.save_dir} on {args.subset}:\n"
                       f"  Total / Valid / Success: {total_tested} / {valid_run} / {success_count}\n"
                       f"  Selected Task IDs: {selected_task_ids}\n"
                       f"  Exec Rate : {exec_rate:.2f}%\n"
                       f"  SR : {sr:.2f}%")
        logging.info(log_message)

    return sr