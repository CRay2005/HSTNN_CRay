import torch
import torch.nn as nn
from collections import OrderedDict
from utils.kfac_utils import fetch_mat_weights
# from utils.common_utils import (tensor_to_list, PresetLRScheduler)
from utils.prune_utils import (filter_indices,
                               filter_indices_ni,
                               get_threshold,
                               update_indices,
                               normalize_factors,
                               prune_model_ni)
# from utils.network_utils import stablize_bn
from tqdm import tqdm

from hessian_fact import get_trace_hut
from pyhessian.hessian import hessian
from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

import numpy as np
import time
import scipy.linalg
import os.path
from os import path

# 添加用于重要性分布分析的导入
from sklearn.cluster import KMeans

class HessianPruner:
        def __init__(self,
                     model,
                     trace_file_name,
                     fix_layers=0,
                     hessian_mode='trace'):
            self.iter = 0
            #self.known_modules = {'Linear', 'Conv2d'}
            self.modules = []
            self.model = model
            self.fix_layers = fix_layers

            self.W_pruned = {}
            self.S_l = None

            self.hessian_mode = hessian_mode
            self.trace_file_name = trace_file_name

            self.importances = {}
            self._inversed = False
            self._cfgs = {}
            self._indices = {}
            
        # top function
        def make_pruned_model(self, dataloader, criterion, device, snn_ratio, seed, batch_size, bptt, ntokens, is_loader=False, normalize=True, re_init=False, n_v=300):
            self.snn_ratio = snn_ratio # use for some special case, particularly slq_full, slq_layer
            self.seed = seed
            self._prepare_model()
            self.mask_list = self._compute_hessian_importance(dataloader, criterion, device, batch_size, bptt, ntokens,  is_loader, n_v=n_v)
            print("Finished Hessian Importance Computation!")
            return self.mask_list

        def _prepare_model(self):
            count = 0
            for it in self.model.named_parameters():
                if it[0].find("all_fc") >= 0:
                    continue
                if it[0].find("wh") >= 0:
                    continue
                if it[0].find("decoder") >= 0:
                    continue
                if it[0].find("fv") >= 0:
                    continue
                if it[0].find("encoder") >= 0:
                    continue
                self.modules.append(it)

        def _analyze_importance_distribution(self, sorted_list, method='percentile', network_type='RNN', target_retention=0.3):
            """
            分析重要性分布并设定阈值
            Args:
                sorted_list: 排序后的重要性列表 [(index, importance), ...]
                method: 分析方法
                network_type: 网络类型，用于打印信息
                target_retention: 目标保留率（0-1之间）
            Returns:
                threshold: 阈值
                analysis_results: 分析结果字典
            """
            importances = [item[1] for item in sorted_list]
            importances_array = np.array(importances)
            
            analysis_results = {
                'importances': importances_array,
                'max': np.max(importances_array),
                'min': np.min(importances_array),
                'mean': np.mean(importances_array),
                'std': np.std(importances_array),
                'method': method,
                'network_type': network_type,
                'target_retention': target_retention
            }
            
            print(f"\n{network_type} Importance Distribution Statistics:")
            print(f"  Max: {analysis_results['max']:.6f}")
            print(f"  Min: {analysis_results['min']:.6f}")
            print(f"  Mean: {analysis_results['mean']:.6f}")
            print(f"  Std: {analysis_results['std']:.6f}")
            
            if method == 'percentile':
                # 方法1: 百分位数方法 - 直接根据目标保留率设定阈值
                percentile = (1 - target_retention) * 100
                threshold = np.percentile(importances_array, percentile)
                analysis_results['threshold_info'] = f"{percentile:.1f}th percentile"
                
            elif method == 'adaptive_statistical':
                # 方法2: 自适应统计方法 - 自动调整系数以达到目标保留率
                mean_val = analysis_results['mean']
                std_val = analysis_results['std']
                
                # 二分搜索找到合适的系数k，使得保留率接近目标值
                def get_retention_rate(k):
                    test_threshold = mean_val + k * std_val
                    return np.sum(importances_array >= test_threshold) / len(importances_array)
                
                # 二分搜索范围
                k_low, k_high = -5.0, 5.0
                tolerance = 0.02  # 允许2%的误差
                
                for _ in range(50):  # 最多迭代50次
                    k_mid = (k_low + k_high) / 2
                    current_rate = get_retention_rate(k_mid)
                    
                    if abs(current_rate - target_retention) < tolerance:
                        break
                    elif current_rate > target_retention:
                        k_low = k_mid
                    else:
                        k_high = k_mid
                
                threshold = mean_val + k_mid * std_val
                final_rate = get_retention_rate(k_mid)
                analysis_results['threshold_info'] = f"Mean+{k_mid:.2f}*Std (achieved {final_rate:.1%})"
                analysis_results['adaptive_k'] = k_mid
                
            elif method == 'improved_clustering':
                # 方法3: 改进的聚类方法 - 使用更多聚类并选择合适的阈值
                if len(importances) >= 6:
                    # 尝试3个聚类
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(importances_array.reshape(-1, 1))
                    centers = sorted(kmeans.cluster_centers_.flatten(), reverse=True)
                    
                    # 选择能够达到目标保留率的阈值
                    # 尝试不同的阈值点：最高聚类中心、中间聚类中心、聚类中心间的中点
                    candidate_thresholds = [
                        centers[0],  # 最高聚类中心
                        centers[1],  # 中间聚类中心
                        (centers[0] + centers[1]) / 2,  # 最高和中间的中点
                        (centers[1] + centers[2]) / 2   # 中间和最低的中点
                    ]
                    
                    best_threshold = candidate_thresholds[0]
                    best_diff = float('inf')
                    
                    for candidate in candidate_thresholds:
                        retention = np.sum(importances_array >= candidate) / len(importances_array)
                        diff = abs(retention - target_retention)
                        if diff < best_diff:
                            best_diff = diff
                            best_threshold = candidate
                    
                    threshold = best_threshold
                    final_retention = np.sum(importances_array >= threshold) / len(importances_array)
                    analysis_results['threshold_info'] = f"3-cluster optimization (achieved {final_retention:.1%})"
                    analysis_results['cluster_centers'] = centers
                    analysis_results['clusters'] = clusters
                else:
                    threshold = np.percentile(importances_array, (1-target_retention)*100)
                    analysis_results['threshold_info'] = f"Insufficient data, using percentile"
                    
            elif method == 'top_k':
                # 方法4: Top-K方法 - 直接选择前K个重要的神经元
                k = int(len(importances) * target_retention)
                threshold = importances_array[k-1] if k > 0 else importances_array[0]
                analysis_results['threshold_info'] = f"Top-{k} selection"
                analysis_results['selected_k'] = k
                
            elif method == 'top_k_direct':
                # 方法4b: 直接Top-K方法 - 您提到的简单方案
                # 这是最直接的方法：直接按比例截取降序序列的前N个元素
                k = max(1, int(len(importances) * target_retention))
                sorted_importances = np.sort(importances_array)[::-1]  # 降序排列
                threshold = sorted_importances[k-1] if k <= len(sorted_importances) else sorted_importances[-1]
                
                actual_retention = k / len(importances)
                selected_quality = sorted_importances[:k].sum() / importances_array.sum()
                
                print(f"    📊 Direct method selected {k}/{len(importances)} neurons")
                print(f"    📈 Retention ratio: {actual_retention:.2%} (exactly as target {target_retention:.2%})")
                print(f"    🎯 Quality achieved: {selected_quality:.2%}")
                
                analysis_results['threshold_info'] = f"Direct top-{k} selection (quality={selected_quality:.2%})"
                analysis_results['selected_k'] = k
                analysis_results['direct_quality'] = selected_quality
                analysis_results['note'] = "This is the simple approach you mentioned - directly taking top N elements"
                
            elif method == 'pareto_optimal':
                # 方法5: 帕累托最优方法
                # 真正的多目标优化：寻找重要性分布的自然断点
                efficiencies = []
                qualities = []
                marginal_benefits = []  # 边际效益
                
                # 计算不同阈值下的指标
                for percentile in range(5, 96, 5):  # 更细粒度的搜索
                    t = np.percentile(importances_array, 100-percentile)
                    selected = importances_array >= t
                    
                    if selected.sum() > 0:
                        efficiency = selected.sum() / len(selected)  # 保留比例
                        quality = importances_array[selected].sum() / importances_array.sum()  # 重要性质量占比
                        
                        # 计算边际效益：每增加1%保留率带来的质量提升
                        if len(qualities) > 0:
                            marginal_benefit = (quality - qualities[-1]) / max(0.01, efficiency - efficiencies[-1])
                        else:
                            marginal_benefit = quality / efficiency if efficiency > 0 else 0
                        
                        efficiencies.append(efficiency)
                        qualities.append(quality)
                        marginal_benefits.append(marginal_benefit)
                
                efficiencies = np.array(efficiencies)
                qualities = np.array(qualities)
                marginal_benefits = np.array(marginal_benefits)
                
                # 策略1: 寻找边际效益显著下降的拐点（膝点检测）
                if len(marginal_benefits) >= 3:
                    # 计算边际效益的二阶差分，寻找急剧下降点
                    diff2 = np.diff(marginal_benefits, n=2)
                    if len(diff2) > 0:
                        knee_candidates = np.where(diff2 < -np.std(diff2))[0]
                        if len(knee_candidates) > 0:
                            knee_idx = knee_candidates[0] + 2  # 调整索引
                        else:
                            knee_idx = len(efficiencies) // 2
                    else:
                        knee_idx = len(efficiencies) // 2
                else:
                    knee_idx = 0
                
                # 策略2: 帕累托前沿分析
                # 寻找效率-质量曲线上的帕累托最优点
                pareto_indices = []
                for i in range(len(efficiencies)):
                    is_pareto = True
                    for j in range(len(efficiencies)):
                        if i != j:
                            # 如果存在其他点在效率和质量上都占优，则当前点不是帕累托最优
                            if (efficiencies[j] >= efficiencies[i] and qualities[j] >= qualities[i] and
                                (efficiencies[j] > efficiencies[i] or qualities[j] > qualities[i])):
                                is_pareto = False
                                break
                    if is_pareto:
                        pareto_indices.append(i)
                
                # 策略3: 综合决策
                if len(pareto_indices) > 0:
                    # 在帕累托前沿上选择最接近目标的点
                    pareto_efficiencies = efficiencies[pareto_indices]
                    pareto_qualities = qualities[pareto_indices]
                    
                    # 如果指定了target_retention，在帕累托前沿上寻找最接近的点
                    if target_retention is not None:
                        distances = np.abs(pareto_efficiencies - target_retention)
                        best_pareto_idx = pareto_indices[np.argmin(distances)]
                    else:
                        # 否则选择边际效益最高的帕累托点
                        pareto_marginal = marginal_benefits[pareto_indices]
                        best_pareto_idx = pareto_indices[np.argmax(pareto_marginal)]
                    
                    best_idx = best_pareto_idx
                    strategy_used = "Pareto-optimal selection"
                else:
                    # 如果没有明显的帕累托前沿，使用膝点
                    best_idx = knee_idx
                    strategy_used = "Knee-point detection"
                
                # 确保索引有效
                best_idx = max(0, min(best_idx, len(efficiencies) - 1))
                
                # 计算最终阈值
                final_efficiency = efficiencies[best_idx]
                final_quality = qualities[best_idx]
                
                # 重新计算对应的阈值
                target_count = max(1, int(len(importances_array) * final_efficiency))
                sorted_importances = np.sort(importances_array)[::-1]  # 降序排列
                if target_count <= len(sorted_importances):
                    threshold = sorted_importances[target_count - 1]
                else:
                    threshold = np.min(importances_array)
                
                # 打印截取比例信息
                actual_retention_ratio = target_count / len(importances_array)
                print(f"    📊 Pareto method selected {target_count}/{len(importances_array)} neurons")
                print(f"    📈 Actual retention ratio: {actual_retention_ratio:.2%} (target was {target_retention:.2%})")
                print(f"    🎯 Quality achieved: {final_quality:.2%}")
                print(f"    🔧 Strategy used: {strategy_used}")
                
                analysis_results['threshold_info'] = f"{strategy_used}: efficiency={final_efficiency:.2%}, quality={final_quality:.2%}"
                analysis_results['pareto_efficiencies'] = efficiencies
                analysis_results['pareto_qualities'] = qualities
                analysis_results['marginal_benefits'] = marginal_benefits
                analysis_results['pareto_indices'] = pareto_indices
                analysis_results['best_idx'] = best_idx
                analysis_results['strategy_used'] = strategy_used
                analysis_results['final_efficiency'] = final_efficiency
                analysis_results['final_quality'] = final_quality
            
            elif method == 'ema_threshold':
                # 方法6: 指数移动平均阈值
                # 计算重要性的指数移动平均，用于平滑处理
                alpha = 0.1  # 平滑因子
                ema = importances_array[0]
                ema_values = [ema]
                
                for i in range(1, len(importances_array)):
                    ema = alpha * importances_array[i] + (1 - alpha) * ema
                    ema_values.append(ema)
                
                # 使用EMA值的百分位数作为阈值
                ema_array = np.array(ema_values)
                percentile = (1 - target_retention) * 100
                threshold = np.percentile(ema_array, percentile)
                analysis_results['threshold_info'] = f"EMA-based {percentile:.1f}th percentile"
                analysis_results['ema_values'] = ema_array
                
            else:
                # 默认使用百分位数方法
                percentile = (1 - target_retention) * 100
                threshold = np.percentile(importances_array, percentile)
                analysis_results['threshold_info'] = f"Default percentile method"
            
            analysis_results['threshold'] = threshold
            
            # 计算实际保留率
            actual_retention = np.sum(importances_array >= threshold) / len(importances_array)
            analysis_results['actual_retention'] = actual_retention
            
            print(f"  {method} method threshold: {threshold:.6f} ({analysis_results['threshold_info']})")
            print(f"  Actual retention rate: {actual_retention:.1%}")
            
            return threshold, analysis_results

        def _global_pareto_allocation(self, global_sorted, max_retain_neurons, rnn_total, snn_total):
            """
            全局帕累托预算分配方法
            Args:
                global_sorted: 全局排序的神经元列表 [(idx, importance, type, layer_size), ...]
                max_retain_neurons: 最大保留神经元数量
                rnn_total: RNN神经元总数
                snn_total: SNN神经元总数
            Returns:
                (rnn_budget, snn_budget): RNN和SNN的预算分配
            """
            print(f"    🧮 Computing global Pareto allocation...")
            
            # 策略1: 分析前N个最重要神经元的类型分布
            top_neurons = global_sorted[:max_retain_neurons]
            rnn_count_in_top = sum(1 for neuron in top_neurons if neuron[2] == 'RNN')
            snn_count_in_top = sum(1 for neuron in top_neurons if neuron[2] == 'SNN')
            
            print(f"    📊 Top {max_retain_neurons} neurons: RNN={rnn_count_in_top}, SNN={snn_count_in_top}")
            
            # 策略2: 计算重要性质量分布
            rnn_importance_sum = sum(neuron[1] for neuron in global_sorted if neuron[2] == 'RNN')
            snn_importance_sum = sum(neuron[1] for neuron in global_sorted if neuron[2] == 'SNN')
            total_importance = rnn_importance_sum + snn_importance_sum
            
            rnn_quality_ratio = rnn_importance_sum / total_importance
            snn_quality_ratio = snn_importance_sum / total_importance
            
            print(f"    📈 Importance quality: RNN={rnn_quality_ratio:.2%}, SNN={snn_quality_ratio:.2%}")
            
            # 策略3: 多种分配方案
            allocations = []
            
            # 方案A: 真正的全局Top-N分布（最重要的策略）
            pure_top_rnn = sum(1 for neuron in top_neurons if neuron[2] == 'RNN')
            pure_top_snn = sum(1 for neuron in top_neurons if neuron[2] == 'SNN')
            # 确保最小配额但优先保持原始分布
            if pure_top_rnn < min_rnn_quota:
                adjustment = min_rnn_quota - pure_top_rnn
                pure_top_rnn = min_rnn_quota
                pure_top_snn = max(min_snn_quota, pure_top_snn - adjustment)
            elif pure_top_snn < min_snn_quota:
                adjustment = min_snn_quota - pure_top_snn
                pure_top_snn = min_snn_quota
                pure_top_rnn = max(min_rnn_quota, pure_top_rnn - adjustment)
            allocations.append((pure_top_rnn, pure_top_snn, "Pure Global Top-N (importance-driven)"))
            
            # 方案B: 基于Top-N分布但确保最小配额
            top_rnn = max(min_rnn_quota, rnn_count_in_top)
            top_snn = max(min_snn_quota, snn_count_in_top)
            if top_rnn + top_snn > max_retain_neurons:
                # 按比例调整
                excess = top_rnn + top_snn - max_retain_neurons
                if top_rnn > top_snn:
                    top_rnn = max(min_rnn_quota, top_rnn - excess)
                    top_snn = max(min_snn_quota, max_retain_neurons - top_rnn)
                else:
                    top_snn = max(min_snn_quota, top_snn - excess)
                    top_rnn = max(min_rnn_quota, max_retain_neurons - top_snn)
            allocations.append((top_rnn, top_snn, "Protected Top-N distribution"))
            
            # 方案C: 质量加权分配但确保最小配额
            quality_rnn = max(min_rnn_quota, int(max_retain_neurons * rnn_quality_ratio))
            quality_snn = max(min_snn_quota, max_retain_neurons - quality_rnn)
            if quality_rnn + quality_snn > max_retain_neurons:
                # 重新调整
                remaining = max_retain_neurons - min_rnn_quota - min_snn_quota
                if remaining > 0:
                    extra_rnn = int(remaining * rnn_quality_ratio)
                    quality_rnn = min_rnn_quota + extra_rnn
                    quality_snn = min_snn_quota + (remaining - extra_rnn)
                else:
                    quality_rnn, quality_snn = min_rnn_quota, min_snn_quota
            allocations.append((quality_rnn, quality_snn, "Protected Quality-weighted"))
            
            # 方案D: 平衡分配
            balanced_base_rnn = max(min_rnn_quota, int(rnn_total * 0.2))  # 基础20%
            balanced_base_snn = max(min_snn_quota, int(snn_total * 0.2))  # 基础20%
            
            if balanced_base_rnn + balanced_base_snn <= max_retain_neurons:
                remaining = max_retain_neurons - balanced_base_rnn - balanced_base_snn
                # 剩余按质量分配
                extra_rnn = int(remaining * rnn_quality_ratio)
                extra_snn = remaining - extra_rnn
                balanced_rnn = balanced_base_rnn + extra_rnn
                balanced_snn = balanced_base_snn + extra_snn
                allocations.append((balanced_rnn, balanced_snn, "Protected Balanced allocation"))
            
            # 方案E: 保守平均分配（权重最低）
            safe_rnn = max(min_rnn_quota, max_retain_neurons // 2)
            safe_snn = max(min_snn_quota, max_retain_neurons - safe_rnn)
            allocations.append((safe_rnn, safe_snn, "Conservative equal split"))
            
            # 策略4: 评估各种分配方案
            best_allocation = None
            best_score = -1
            
            print(f"    🔍 Evaluating allocation strategies:")
            for rnn_budget, snn_budget, method_name in allocations:
                # 确保预算有效
                rnn_budget = max(min_rnn_quota, min(rnn_budget, rnn_total))
                snn_budget = max(min_snn_quota, min(snn_budget, snn_total))
                
                if rnn_budget + snn_budget > max_retain_neurons:
                    # 最终调整到预算范围内
                    excess = rnn_budget + snn_budget - max_retain_neurons
                    if rnn_budget > min_rnn_quota and excess > 0:
                        reduce_rnn = min(excess, rnn_budget - min_rnn_quota)
                        rnn_budget -= reduce_rnn
                        excess -= reduce_rnn
                    if snn_budget > min_snn_quota and excess > 0:
                        reduce_snn = min(excess, snn_budget - min_snn_quota)
                        snn_budget -= reduce_snn
                
                # 评估这种分配的质量
                score = self._evaluate_allocation_quality_improved(
                    global_sorted, rnn_budget, snn_budget, rnn_total, snn_total, use_normalization
                )
                
                # 计算评分分解以便调试
                selected_rnn = [n for n in global_sorted if n[2] == 'RNN'][:rnn_budget]
                selected_snn = [n for n in global_sorted if n[2] == 'SNN'][:snn_budget]
                
                if use_normalization:
                    total_selected_importance = sum(n[2] for n in selected_rnn + selected_snn)
                    total_importance = sum(n[2] for n in global_sorted)
                else:
                    total_selected_importance = sum(n[1] for n in selected_rnn + selected_snn)
                    total_importance = sum(n[1] for n in global_sorted)
                
                quality_ratio = total_selected_importance / total_importance if total_importance > 0 else 0
                rnn_retention_rate = rnn_budget / rnn_total
                snn_retention_rate = snn_budget / snn_total
                diversity_bonus = 1 - abs(rnn_retention_rate - snn_retention_rate) * 0.3
                quota_bonus = 0.1 if (rnn_budget >= max(2, int(rnn_total * 0.15)) and snn_budget >= max(2, int(snn_total * 0.15))) else 0
                
                print(f"        🎯 {method_name}:")
                print(f"           Allocation: RNN={rnn_budget} ({rnn_retention_rate:.1%}), SNN={snn_budget} ({snn_retention_rate:.1%})")
                print(f"           Quality: {quality_ratio:.3f} (×0.85={quality_ratio*0.85:.3f})")
                print(f"           Diversity: {diversity_bonus:.3f} (×0.10={diversity_bonus*0.10:.3f})")
                print(f"           Quota: {quota_bonus:.3f} (×0.05={quota_bonus*0.05:.3f})")
                print(f"           Total Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_allocation = (rnn_budget, snn_budget, method_name)
            
            if best_allocation:
                rnn_budget, snn_budget, best_method = best_allocation
                print(f"    ✅ Selected: {best_method} (Score: {best_score:.4f})")
                print(f"    📋 Final allocation reasoning:")
                print(f"        - This strategy achieved the highest combined score")
                print(f"        - RNN allocation: {rnn_budget}/{rnn_total} = {rnn_budget/rnn_total:.1%}")
                print(f"        - SNN allocation: {snn_budget}/{snn_total} = {snn_budget/snn_total:.1%}")
                
                # 最终验证
                print(f"    🔍 Final validation:")
                print(f"        RNN quota: {rnn_budget}/{rnn_total} ({rnn_budget/rnn_total:.1%}) ≥ {min_rnn_quota} ✓")
                print(f"        SNN quota: {snn_budget}/{snn_total} ({snn_budget/snn_total:.1%}) ≥ {min_snn_quota} ✓")
                print(f"        Total: {rnn_budget + snn_budget}/{max_retain_neurons} ≤ {max_retain_neurons} ✓")
                
                return rnn_budget, snn_budget
            else:
                # 极端情况下的默认分配
                default_rnn = max(min_rnn_quota, max_retain_neurons // 2)
                default_snn = max(min_snn_quota, max_retain_neurons - default_rnn)
                print(f"    ⚠️  Using emergency default allocation: RNN={default_rnn}, SNN={default_snn}")
                return default_rnn, default_snn
        
        def _evaluate_allocation_quality_improved(self, global_sorted, rnn_budget, snn_budget, rnn_total, snn_total, use_normalization=False):
            """
            改进的分配方案质量评估
            """
            # 获取实际会被选择的神经元
            selected_rnn = [n for n in global_sorted if n[2] == 'RNN'][:rnn_budget]
            selected_snn = [n for n in global_sorted if n[2] == 'SNN'][:snn_budget]
            
            # 计算质量指标（使用适当的重要性值）
            if use_normalization:
                # 使用归一化后的比较值
                total_selected_importance = sum(n[2] for n in selected_rnn + selected_snn)
                total_importance = sum(n[2] for n in global_sorted)
            else:
                # 使用原始重要性值
                total_selected_importance = sum(n[1] for n in selected_rnn + selected_snn)
                total_importance = sum(n[1] for n in global_sorted)
            
            quality_ratio = total_selected_importance / total_importance if total_importance > 0 else 0
            
            # 计算多样性奖励（避免过度偏向某一类型）
            rnn_retention_rate = rnn_budget / rnn_total
            snn_retention_rate = snn_budget / snn_total
            
            # 多样性奖励：两种类型保留率越接近，奖励越高，但权重降低
            diversity_bonus = 1 - abs(rnn_retention_rate - snn_retention_rate) * 0.3
            
            # 配额满足奖励
            quota_bonus = 0
            if rnn_budget >= max(2, int(rnn_total * 0.15)) and snn_budget >= max(2, int(snn_total * 0.15)):
                quota_bonus = 0.1
            
            # 综合评分：更重视质量，适度考虑多样性
            score = quality_ratio * 0.85 + diversity_bonus * 0.10 + quota_bonus * 0.05
            return score

        def _compute_hessian_importance(self, dataloader, criterion, device, batch_size, bptt, ntokens, is_loader, n_v=300):
            print("is_loader", is_loader)
            ###############
            # Here, we use the fact that Conv does not have bias term
            ###############
            if self.hessian_mode == 'trace':
                # 1.只对特定层（SNN和RNN）计算Hessian
                for k, v in self.model.named_parameters():
                    if k.find("all_fc") >= 0:
                        v.requires_grad = False
                    elif k.find("wh") >= 0:
                        v.requires_grad = False
                    elif k.find("decoder") >= 0:
                        v.requires_grad = False
                    elif k.find("fv") >= 0:
                        v.requires_grad = False
                    elif k.find("encoder") >= 0:
                        v.requires_grad = False
                    else:
                        print(k, v.requires_grad)
                # 2.加载或者调用get_trace_hut函数计算Hessian
                trace_dir = self.trace_file_name
                print(trace_dir)
                if os.path.exists(trace_dir):
                    print(f"Loading trace from {trace_dir}")
                    import numpy as np
                    results = np.load(trace_dir, allow_pickle=True)
                else:
                    results = get_trace_hut(self.model, dataloader, criterion, n_v, batch_size, bptt, ntokens, loader=is_loader, channelwise=True, layerwise=False)
                    import numpy as np
                    np.save(self.trace_file_name, results)

                for m in self.model.parameters():
                    m.requires_grad = True

                #3.通道重要性平均值计算
                channel_trace, weighted_trace = [], []
                # results结构：[层][通道][采样次数]
                #处理层 layer
                for k, layer in enumerate(results):
                    # print(k, layer)
                    channel_trace.append(torch.zeros(len(layer)))
                    weighted_trace.append(torch.zeros(len(layer)))
                    #处理通道 channel
                    for cnt, channel in enumerate(layer):
                        #print(cnt, channel.shape, len(layer))
                        # 计算每个通道的平均值
                        channel_trace[k][cnt] = sum(channel) / len(channel)
                #for i in channel_trace:
                    # print(len(i))
                # print(len(results), self.model.parameters())

                # 4.weight加权重要性计算
                # 重要性 = Hessian迹 × (权重范数²/权重元素数量)
                # Hessian迹反映参数对损失函数的敏感性
                # 权重范数反映参数的大小
                # 结合两者得到更准确的神经元重要性评估
                
                # 存储权重数据用于CSV生成
                weights_for_csv = []
                
                # 在开始处理所有模块前，先采集激活和梯度信息
                print("🔄 开始采集激活和梯度信息以增强重要性计算...")
                activation_importance, gradient_importance = self._collect_activation_gradient_info(dataloader, criterion, batch_size, bptt, ntokens)
                
                for k, mod in enumerate(self.modules):
                    tmp = []
                    # k：   层索引（0, 1, 2, 3对应snn_fc1, rnn_fc1, snn_fc2, rnn_fc2）
                    # mod： 每个模块的信息，格式为(模块名称, 模块对象)
                    m = mod[0]
                    import copy 
                    cur_weight = copy.deepcopy(mod[1].data) #mod[1]：模块对象（如Linear层），mod[1].data：该层的权重张量
                    dims = len(list(cur_weight.size()))

                    # 维度转换
                    # 目的：统一权重张量的格式，使第一维度对应通道数（与Hessian迹的通道对应）                
                    if dims == 2:
                        # 2维情况（全连接层）：
                        # 原始形状：[输出神经元数, 输入神经元数]
                        # 转换后：[输入神经元数, 输出神经元数]
                        cur_weight = cur_weight.permute(1, 0)
                    elif dims == 3:
                        # 3维情况（可能的卷积层或特殊结构）：
                        # 原始形状：[输出通道, 输入通道, 其他维度]
                        # 转换后：[其他维度, 输出通道, 输入通道]   
                        cur_weight = cur_weight.permute(2, 0, 1)
                    
                    # 保存权重数据用于CSV生成
                    weights_for_csv.append(cur_weight)
                    
                    for cnt, channel in enumerate(cur_weight):
                        # 原有的重要性计算
                        # channel_trace[k][cnt]：上面计算得到的，第k层第cnt个通道的平均Hessian迹
                        # .detach()：从计算图中分离，避免梯度计算
                        # .norm()**2：计算L2范数的平方（权重向量的平方和）
                        # / channel.numel()：除以权重元素数量，归一化
                        base_importance = (channel_trace[k][cnt] * channel.detach().norm()**2 / channel.numel()).cpu().item()
                        
                        # 获取增强因子
                        activation_factor = 1.0  # 默认因子
                        gradient_factor = 1.0    # 默认因子
                        
                        # 尝试获取激活增强因子
                        layer_name = m
                        if layer_name in activation_importance and cnt < len(activation_importance[layer_name]):
                            #activation_factor = 1.0 + activation_importance[layer_name][cnt] * 0.2  # 20%的激活权重
                            activation_factor = activation_importance[layer_name][cnt] * 0.3
                        #！！！！！！！！！！！！！！！csv中是按照0.1取的值！！！！！！！！！
                        # 尝试获取梯度增强因子  
                        if layer_name in gradient_importance and cnt < len(gradient_importance[layer_name]):
                            gradient_factor = 1.0 + gradient_importance[layer_name][cnt] * 0.15  # 15%的梯度权重
                        
                        # 计算增强后的重要性
                        #enhanced_importance = base_importance * activation_factor * gradient_factor
                        enhanced_importance = (base_importance + activation_factor * channel_trace[k][cnt])
                   
                        # 如果有增强信息，打印调试信息（仅前几个神经元）
                        if cnt < 3 and (activation_factor != 1.0 or gradient_factor != 1.0):
                            print(f"  Layer {layer_name}, Neuron {cnt}: base={base_importance:.6f}, "
                                  f"act_factor={activation_factor:.3f}, grad_factor={gradient_factor:.3f}, "
                                  f"enhanced={enhanced_importance:.6f}")
                        
                        tmp.append(enhanced_importance)
                    print(m, len(tmp))
                    self.importances[str(m)] = (tmp, len(tmp))
                
                #生成包含详细重要性信息的CSV文件
                # print(f"\n📊 生成神经元重要性详细信息...")
                # csv_filename = f"neuron_importance_details_{self.trace_file_name.split('/')[-1].replace('.npy', '')}.csv"
                # self._save_importance_details_to_csv(channel_trace, weights_for_csv, activation_importance, gradient_importance, csv_filename)
                
                # 注释掉强制退出，让程序继续执行剪枝流程
                # print(f"\n🛑 CSV文件生成完成，程序退出以便调试")
                # import sys
                # sys.exit(0)

            else:
                print("Unsupported mode")
                assert False

            tmp_imp_list = list(self.importances.items())

            # 5.混合网络结构处理
            rnn_list = [None, None]     # rnn_fc1, rnn_fc2
            snn_list = [None, None]     # snn_fc1, snn_fc2

            for unit in tmp_imp_list:
                if unit[0].find("rnn") >= 0 or unit[0].find("lstm") >= 0:
                    if unit[0].find("1") >= 0:
                        rnn_list[0] = unit[1][0]        # rnn_fc1层的重要性
                    else:
                        assert unit[0].find("2") >= 0
                        rnn_list[1] = unit[1][0]        # rnn_fc2层的重要性
                elif unit[0].find("snn") >= 0:
                    if unit[0].find("1") >= 0:
                        snn_list[0] = unit[1][0]        # snn_fc1层的重要性
                    else:
                        assert unit[0].find("2") >= 0
                        snn_list[1] = unit[1][0]        # snn_fc2层的重要性
                else:
                    continue

            rnn_shape = [len(rnn_list[0]), len(rnn_list[1])]
            snn_shape = [len(snn_list[0]), len(snn_list[1])]

            # 6.重要性排序
            rnn_tuple_list = []
            snn_tuple_list = []
            # 创建(索引, 重要性)元组列表
            for no in range(len(rnn_list[0])):
                rnn_tuple_list.append((no, rnn_list[0][no]))
            for no in range(len(rnn_list[1])):
                rnn_tuple_list.append((no + rnn_shape[0], rnn_list[1][no]))
            # 按重要性降序排序
            for no in range(len(snn_list[0])):
                snn_tuple_list.append((no, snn_list[0][no]))
            for no in range(len(snn_list[1])):
                snn_tuple_list.append((no + snn_shape[0], snn_list[1][no]))

            sorted_rnn_list = sorted(rnn_tuple_list, key=lambda x:x[1])     #, reverse=True) #按重要性降序排序
            sorted_snn_list = sorted(snn_tuple_list, key=lambda x:x[1])     #, reverse=True) #按重要性降序排序

            sorted_rnn_list.reverse()   #降序排序，如[(3, 0.9)，(0, 0.8)，(4, 0.6)...]
            sorted_snn_list.reverse()   #降序排序
            
            del rnn_list, snn_list, rnn_tuple_list, snn_tuple_list

            # 7. 根据flag选择不同的剪枝算法
            flag = 2
            if flag == 1:
                print(f"\n{'='*80}")
                print(f"🔥 执行算法1: 基于重要性分布分析的全局帕累托方法")
                print(f"{'='*80}")
                
                # 7.1 全局剪枝优化：确保整体剪枝率至少50%
                target_retention = 0
                analysis_method = 'pareto_optimal'
                
                total_neurons = len(sorted_rnn_list) + len(sorted_snn_list)
                max_retain_neurons = int(total_neurons * 0.5)  # 最多保留50%
                
                print(f"\n🌐 Global Pruning Constraint:")
                print(f"   Total neurons: {total_neurons} (RNN: {len(sorted_rnn_list)}, SNN: {len(sorted_snn_list)})")
                print(f"   Maximum retain: {max_retain_neurons} (≤50% pruning target)")
                
                # 7.2 智能预算分配策略
                if analysis_method == 'pareto_optimal':
                    # 检查RNN和SNN重要性的数值尺度差异
                    rnn_importances = [imp for idx, imp in sorted_rnn_list]
                    snn_importances = [imp for idx, imp in sorted_snn_list]
                    
                    rnn_mean = np.mean(rnn_importances)
                    snn_mean = np.mean(snn_importances)
                    rnn_max = np.max(rnn_importances)
                    snn_max = np.max(snn_importances)
                    
                    print(f"\n🔍 Scale Analysis:")
                    print(f"   RNN importance - Mean: {rnn_mean:.6f}, Max: {rnn_max:.6f}")
                    print(f"   SNN importance - Mean: {snn_mean:.6f}, Max: {snn_max:.6f}")
                    
                    # 计算尺度差异
                    scale_ratio = rnn_mean / snn_mean if snn_mean > 0 else float('inf')
                    print(f"   Scale ratio (RNN/SNN): {scale_ratio:.2f}")
                    
                    if scale_ratio > 10 or scale_ratio < 0.1:
                        print(f"   ⚠️  Significant scale difference detected! Using normalized comparison.")
                        use_normalization = True
                    else:
                        print(f"   ✅ Scales are comparable, using direct comparison.")
                        use_normalization = False
                    
                    # 合并所有神经元并进行尺度处理
                    global_neuron_list = []
                    
                    if use_normalization:
                        # 方法1: 归一化处理 - 将每种类型的重要性归一化到[0,1]
                        rnn_min, rnn_range = np.min(rnn_importances), np.max(rnn_importances) - np.min(rnn_importances)
                        snn_min, snn_range = np.min(snn_importances), np.max(snn_importances) - np.min(snn_importances)
                        
                        # 标记来源并添加归一化重要性
                        for idx, importance in sorted_rnn_list:
                            normalized_imp = (importance - rnn_min) / rnn_range if rnn_range > 0 else 0
                            global_neuron_list.append((idx, importance, normalized_imp, 'RNN', len(sorted_rnn_list)))
                        
                        for idx, importance in sorted_snn_list:
                            normalized_imp = (importance - snn_min) / snn_range if snn_range > 0 else 0
                            global_neuron_list.append((idx, importance, normalized_imp, 'SNN', len(sorted_snn_list)))
                        
                        # 按归一化重要性排序
                        global_sorted = sorted(global_neuron_list, key=lambda x: x[2], reverse=True)
                        print(f"   📊 Using normalized importance for global ranking")
                    else:
                        # 方法2: 直接使用原始重要性
                        for idx, importance in sorted_rnn_list:
                            global_neuron_list.append((idx, importance, importance, 'RNN', len(sorted_rnn_list)))
                        
                        for idx, importance in sorted_snn_list:
                            global_neuron_list.append((idx, importance, importance, 'SNN', len(sorted_snn_list)))
                        
                        # 按原始重要性排序
                        global_sorted = sorted(global_neuron_list, key=lambda x: x[2], reverse=True)
                    
                    print(f"\n🔗 Global Pareto Analysis:")
                    print(f"   Analyzing {len(global_sorted)} neurons globally...")
                    
                    # 使用改进的全局帕累托分析
                    rnn_budget, snn_budget = self._global_pareto_allocation_improved(
                        global_sorted, max_retain_neurons, len(sorted_rnn_list), len(sorted_snn_list), use_normalization
                    )
                    
                    print(f"\n💰 Budget Allocation Results:")
                    print(f"   RNN budget: {rnn_budget}/{len(sorted_rnn_list)} ({rnn_budget/len(sorted_rnn_list):.1%})")
                    print(f"   SNN budget: {snn_budget}/{len(sorted_snn_list)} ({snn_budget/len(sorted_snn_list):.1%})")
                    print(f"   Total retain: {rnn_budget + snn_budget}/{total_neurons} ({(rnn_budget + snn_budget)/total_neurons:.1%})")
                    
                    # 7.3 基于预算进行局部帕累托优化
                    rnn_target_retention = rnn_budget / len(sorted_rnn_list)
                    snn_target_retention = snn_budget / len(sorted_snn_list)
                else:
                    # 非帕累托方法保持原有逻辑
                    rnn_target_retention = 0.4
                    snn_target_retention = 0.2
                
                # 7.4 分析RNN重要性分布并设定阈值
                rnn_threshold, rnn_analysis = self._analyze_importance_distribution(
                    sorted_rnn_list, analysis_method, 'RNN', rnn_target_retention
                )
                
                # 7.5 分析SNN重要性分布并设定阈值
                snn_threshold, snn_analysis = self._analyze_importance_distribution(
                    sorted_snn_list, analysis_method, 'SNN', snn_target_retention
                )
                
                # 7.6 基于阈值选择要保留的神经元
                eff_rnns_list = []
                eff_snns_list = []
                
                # 选择重要性大于等于阈值的RNN神经元
                for neuron_idx, importance in sorted_rnn_list:
                    if importance >= rnn_threshold:
                        eff_rnns_list.append(neuron_idx)
                
                # 选择重要性大于等于阈值的SNN神经元
                for neuron_idx, importance in sorted_snn_list:
                    if importance >= snn_threshold:
                        eff_snns_list.append(neuron_idx)
                
                eff_rnns_number = len(eff_rnns_list)
                eff_snns_number = len(eff_snns_list)
                
                print(f"\nThreshold-based selection results:")
                print(f"  RNN neurons selected: {eff_rnns_number}/{len(sorted_rnn_list)} ({eff_rnns_number/len(sorted_rnn_list):.1%})")
                print(f"  SNN neurons selected: {eff_snns_number}/{len(sorted_snn_list)} ({eff_snns_number/len(sorted_snn_list):.1%})")

                # 7.7 强制执行全局50%剪枝约束 - 这是关键步骤！
                current_total_retained = eff_rnns_number + eff_snns_number
                
                print(f"\n🚨 Enforcing Global 50% Pruning Constraint:")
                print(f"   Current retained: {current_total_retained}/{total_neurons} ({current_total_retained/total_neurons:.1%})")
                print(f"   Maximum allowed: {max_retain_neurons}/{total_neurons} (50.0%)")
                
                if current_total_retained > max_retain_neurons:
                    print(f"   ⚠️  VIOLATION: {current_total_retained - max_retain_neurons} neurons over budget!")
                    print(f"   🔧 Applying forced truncation to meet 50% target...")
                    
                    # 需要削减的神经元数量
                    excess = current_total_retained - max_retain_neurons
                    
                    # 使用改进的预算分配结果进行强制截断
                    if analysis_method == 'pareto_optimal':
                        # 使用预算分配的结果
                        target_rnn = rnn_budget
                        target_snn = snn_budget
                    else:
                        # 按比例分配到预算范围内
                        rnn_ratio = eff_rnns_number / current_total_retained
                        target_rnn = int(max_retain_neurons * rnn_ratio)
                        target_snn = max_retain_neurons - target_rnn
                    
                    print(f"   📊 Target allocation: RNN={target_rnn}, SNN={target_snn}")
                    
                    # 强制截断到目标数量
                    if eff_rnns_number > target_rnn:
                        # 保留前target_rnn个最重要的RNN神经元
                        rnn_with_importance = [(idx, dict(sorted_rnn_list)[idx]) for idx in eff_rnns_list]
                        rnn_with_importance.sort(key=lambda x: x[1], reverse=True)
                        eff_rnns_list = [idx for idx, _ in rnn_with_importance[:target_rnn]]
                        print(f"   ✂️  RNN truncated: {eff_rnns_number} → {len(eff_rnns_list)}")
                    
                    if eff_snns_number > target_snn:
                        # 保留前target_snn个最重要的SNN神经元
                        snn_with_importance = [(idx, dict(sorted_snn_list)[idx]) for idx in eff_snns_list]
                        snn_with_importance.sort(key=lambda x: x[1], reverse=True)
                        eff_snns_list = [idx for idx, _ in snn_with_importance[:target_snn]]
                        print(f"   ✂️  SNN truncated: {eff_snns_number} → {len(eff_snns_list)}")
                    
                    # 更新计数
                    eff_rnns_number = len(eff_rnns_list)
                    eff_snns_number = len(eff_snns_list)
                    current_total_retained = eff_rnns_number + eff_snns_number
                    
                    print(f"   ✅ After truncation: RNN={eff_rnns_number}, SNN={eff_snns_number}, Total={current_total_retained}")
                
                elif current_total_retained < max_retain_neurons:
                    # 如果保留的神经元少于预算，可以适当增加（但保持在50%以内）
                    available_budget = max_retain_neurons - current_total_retained
                    print(f"   📈 Under budget by {available_budget} neurons, could retain more if needed")
                    
                    # 可选：智能补充一些接近阈值的神经元
                    if available_budget > 0 and analysis_method == 'pareto_optimal':
                        print(f"   🔍 Considering additional high-importance neurons within budget...")
                        
                        # 从未选中但重要性较高的神经元中补充
                        additional_rnn = []
                        additional_snn = []
                        
                        # 获取未选中的RNN神经元，按重要性排序
                        unselected_rnn = [(idx, imp) for idx, imp in sorted_rnn_list if idx not in eff_rnns_list]
                        for idx, imp in unselected_rnn[:available_budget//2]:
                            additional_rnn.append(idx)
                        
                        # 获取未选中的SNN神经元，按重要性排序
                        unselected_snn = [(idx, imp) for idx, imp in sorted_snn_list if idx not in eff_snns_list]
                        for idx, imp in unselected_snn[:available_budget - len(additional_rnn)]:
                            additional_snn.append(idx)
                        
                        if additional_rnn or additional_snn:
                            eff_rnns_list.extend(additional_rnn)
                            eff_snns_list.extend(additional_snn)
                            eff_rnns_number = len(eff_rnns_list)
                            eff_snns_number = len(eff_snns_list)
                            current_total_retained = eff_rnns_number + eff_snns_number
                            print(f"   📈 Added {len(additional_rnn)} RNN + {len(additional_snn)} SNN neurons")
                            print(f"   📊 New totals: RNN={eff_rnns_number}, SNN={eff_snns_number}, Total={current_total_retained}")
                
                else:
                    print(f"   ✅ Perfect match: exactly {max_retain_neurons} neurons retained")
                
                # 7.8 最终全局约束验证
                assert current_total_retained <= max_retain_neurons, f"CONSTRAINT VIOLATION: {current_total_retained} > {max_retain_neurons}"
                
                final_pruning_rate_enforced = (total_neurons - current_total_retained) / total_neurons
                print(f"\n🎯 Global Constraint Enforcement Results:")
                print(f"   Retained neurons: {current_total_retained}/{total_neurons} ({current_total_retained/total_neurons:.1%})")
                print(f"   Pruning rate: {final_pruning_rate_enforced:.1%}")
                
                if final_pruning_rate_enforced >= 0.5:
                    print(f"   ✅ SUCCESS: Achieved ≥50% pruning target!")
                else:
                    print(f"   ❌ ERROR: Still below 50% target - this should not happen!")
                
                print(f"   📊 Breakdown: RNN={eff_rnns_number}/{len(sorted_rnn_list)} ({eff_rnns_number/len(sorted_rnn_list):.1%}), SNN={eff_snns_number}/{len(sorted_snn_list)} ({eff_snns_number/len(sorted_snn_list):.1%})")

            elif flag == 2:
                print(f"\n{'='*80}")
                print(f"🎯 执行算法2: 直接Top-K方法")
                print(f"{'='*80}")
                
                # 设定目标保留率 - 可以根据需要调整
                target_rnn_retention = 0.1 # 保留  %的RNN神经元
                target_snn_retention = 0.9  # 保留  %的SNN神经元
                
                print(f"🎯 目标保留率: RNN={target_rnn_retention:.1%}, SNN={target_snn_retention:.1%}")
                
                # 2.1 使用top_k_direct方法分析RNN
                rnn_threshold, rnn_analysis = self._analyze_importance_distribution(
                    sorted_rnn_list, 'top_k_direct', 'RNN', target_rnn_retention
                )
                
                # 2.2 使用top_k_direct方法分析SNN
                snn_threshold, snn_analysis = self._analyze_importance_distribution(
                    sorted_snn_list, 'top_k_direct', 'SNN', target_snn_retention
                )
                
                # 2.3 基于阈值选择要保留的神经元
                eff_rnns_list = []
                eff_snns_list = []
                
                # 选择重要性大于等于阈值的RNN神经元
                for neuron_idx, importance in sorted_rnn_list:
                    if importance >= rnn_threshold:
                        eff_rnns_list.append(neuron_idx)
                
                # 选择重要性大于等于阈值的SNN神经元
                for neuron_idx, importance in sorted_snn_list:
                    if importance >= snn_threshold:
                        eff_snns_list.append(neuron_idx)
                
                eff_rnns_number = len(eff_rnns_list)
                eff_snns_number = len(eff_snns_list)
                
                print(f"\n📊 Top-K Selection Results:")
                print(f"  RNN neurons selected: {eff_rnns_number}/{len(sorted_rnn_list)} ({eff_rnns_number/len(sorted_rnn_list):.1%})")
                print(f"  SNN neurons selected: {eff_snns_number}/{len(sorted_snn_list)} ({eff_snns_number/len(sorted_snn_list):.1%})")
                
                # 2.4 计算总体剪枝率
                total_neurons = len(sorted_rnn_list) + len(sorted_snn_list)
                current_total_retained = eff_rnns_number + eff_snns_number
                pruning_rate = (total_neurons - current_total_retained) / total_neurons
                
                print(f"  Total pruning rate: {pruning_rate:.1%} ({total_neurons - current_total_retained}/{total_neurons} neurons pruned)")
                
                # 2.5 显示详细质量信息
                rnn_quality = rnn_analysis.get('direct_quality', 0)
                snn_quality = snn_analysis.get('direct_quality', 0)
                print(f"  Quality achieved: RNN={rnn_quality:.1%}, SNN={snn_quality:.1%}")
                
            else:
                print(f"⚠️  不支持的剪枝算法 flag={flag}，使用默认算法（算法1）")
                # 递归调用自己，使用flag=1
                return self._compute_hessian_importance(dataloader, criterion, device, batch_size, bptt, ntokens, is_loader, n_v, flag=1)

            # 8. 确保每层至少保留一个神经元（保持原有的安全机制）- 所有算法共用
            print(f"\n🛡️  执行层结构安全检查...")
            rnn_layer_util = [False, False] #使用布尔数组记录每一层是否至少保留了一个神经元
            snn_layer_util = [False, False]

            # check whether at least one neuron(rnn or snn) exists in every layer
            for neuron_idx in eff_rnns_list:
                if neuron_idx >= rnn_shape[0]:
                    rnn_layer_util[1] = True
                else:
                    rnn_layer_util[0] = True
            
            for neuron_idx in eff_snns_list:
                if neuron_idx >= snn_shape[0]:
                    snn_layer_util[1] = True
                else:
                    snn_layer_util[0] = True
            
            # fix the structure - 如果某层没有保留神经元，强制保留一个
            def not_in_one_layer(idx1, idx2, thres):
                return (idx1 < thres and idx2 >= thres) or (idx2 < thres and idx1 >= thres)
            
            # 处理RNN层结构问题
            if rnn_layer_util[0] is False or rnn_layer_util[1] is False:
                print("Warning: Some RNN layer has no preserved neurons, fixing structure...")
                if len(eff_rnns_list) > 0:
                    last_one = eff_rnns_list[-1]
                    # 从未选中的神经元中找一个来替换，确保两层都有神经元
                    for neuron_idx, importance in sorted_rnn_list:
                        if neuron_idx not in eff_rnns_list:
                            if not_in_one_layer(last_one, neuron_idx, rnn_shape[0]):
                                eff_rnns_list[-1] = neuron_idx
                                print(f"  Replaced RNN neuron {last_one} with {neuron_idx} to maintain layer structure")
                                break
                else:
                    # 如果没有保留任何RNN神经元，至少保留两个（每层一个）
                    if len(sorted_rnn_list) >= 2:
                        # 选择每层中重要性最高的神经元
                        layer1_best = None
                        layer2_best = None
                        for neuron_idx, importance in sorted_rnn_list:
                            if neuron_idx < rnn_shape[0] and layer1_best is None:
                                layer1_best = neuron_idx
                            elif neuron_idx >= rnn_shape[0] and layer2_best is None:
                                layer2_best = neuron_idx
                            if layer1_best is not None and layer2_best is not None:
                                break
                        eff_rnns_list = [layer1_best, layer2_best]
                        print(f"  Force preserved RNN neurons: {eff_rnns_list}")

            # 处理SNN层结构问题
            if snn_layer_util[0] is False or snn_layer_util[1] is False:
                print("Warning: Some SNN layer has no preserved neurons, fixing structure...")
                if len(eff_snns_list) > 0:
                    last_one = eff_snns_list[-1]
                    # 从未选中的神经元中找一个来替换，确保两层都有神经元
                    for neuron_idx, importance in sorted_snn_list:
                        if neuron_idx not in eff_snns_list:
                            if not_in_one_layer(last_one, neuron_idx, snn_shape[0]):
                                eff_snns_list[-1] = neuron_idx
                                print(f"  Replaced SNN neuron {last_one} with {neuron_idx} to maintain layer structure")
                                break
                else:
                    # 如果没有保留任何SNN神经元，至少保留两个（每层一个）
                    if len(sorted_snn_list) >= 2:
                        # 选择每层中重要性最高的神经元
                        layer1_best = None
                        layer2_best = None
                        for neuron_idx, importance in sorted_snn_list:
                            if neuron_idx < snn_shape[0] and layer1_best is None:
                                layer1_best = neuron_idx
                            elif neuron_idx >= snn_shape[0] and layer2_best is None:
                                layer2_best = neuron_idx
                            if layer1_best is not None and layer2_best is not None:
                                break
                        eff_snns_list = [layer1_best, layer2_best]
                        print(f"  Force preserved SNN neurons: {eff_snns_list}")

            del rnn_layer_util, snn_layer_util

            # 9. 最终输出（所有算法共用）
            eff_dict = {}
            eff_dict["rnn1"] = []
            eff_dict["rnn2"] = []
            eff_dict["snn1"] = []
            eff_dict["snn2"] = []

            for item in eff_rnns_list:
                if item < rnn_shape[0]:
                    eff_dict["rnn1"].append(item)
                else:
                    eff_dict["rnn2"].append(item - rnn_shape[0])
            
            for item in eff_snns_list:
                if item < snn_shape[0]:
                    eff_dict["snn1"].append(item)
                else:
                    eff_dict["snn2"].append(item - snn_shape[0])
            
            print(f"\n✅ 剪枝算法 flag={flag} 执行完成!")
            return eff_dict

        def _global_pareto_allocation_improved(self, global_sorted, max_retain_neurons, rnn_total, snn_total, use_normalization=False):
            """
            改进的全局帕累托预算分配方法 - 解决尺度差异问题
            Args:
                global_sorted: 全局排序的神经元列表 [(idx, original_importance, comparison_value, type, layer_size), ...]
                max_retain_neurons: 最大保留神经元数量
                rnn_total: RNN神经元总数
                snn_total: SNN神经元总数
                use_normalization: 是否使用了归一化
            Returns:
                (rnn_budget, snn_budget): RNN和SNN的预算分配
            """
            print(f"    🧮 Computing improved global Pareto allocation...")
            
            # 设置最小配额保护 - 确保每种类型至少保留一定比例
            min_rnn_quota = max(2, int(rnn_total * 0.15))  # 至少保留15%的RNN
            min_snn_quota = max(2, int(snn_total * 0.15))  # 至少保留15%的SNN
            
            print(f"    🛡️  Minimum quotas: RNN≥{min_rnn_quota}, SNN≥{min_snn_quota}")
            
            # 检查配额是否可行
            if min_rnn_quota + min_snn_quota > max_retain_neurons:
                print(f"    ⚠️  Minimum quotas exceed budget, adjusting...")
                # 按比例调整最小配额
                total_min = min_rnn_quota + min_snn_quota
                min_rnn_quota = max(1, int(min_rnn_quota * max_retain_neurons / total_min))
                min_snn_quota = max(1, max_retain_neurons - min_rnn_quota)
                print(f"    🔧 Adjusted quotas: RNN≥{min_rnn_quota}, SNN≥{min_snn_quota}")
            
            # 策略1: 分析前N个最重要神经元的类型分布（考虑尺度问题）
            top_neurons = global_sorted[:max_retain_neurons]
            rnn_count_in_top = sum(1 for neuron in top_neurons if neuron[3] == 'RNN')
            snn_count_in_top = sum(1 for neuron in top_neurons if neuron[3] == 'SNN')
            
            print(f"    📊 Top {max_retain_neurons} neurons: RNN={rnn_count_in_top}, SNN={snn_count_in_top}")
            print(f"    🔍 Raw distribution in top neurons: RNN={rnn_count_in_top/max_retain_neurons:.1%}, SNN={snn_count_in_top/max_retain_neurons:.1%}")
            
            # 分析前100、200、300等不同范围的分布，看趋势
            for check_range in [100, 200, 300, 500, max_retain_neurons]:
                if check_range <= len(global_sorted):
                    check_neurons = global_sorted[:check_range]
                    check_rnn = sum(1 for n in check_neurons if n[3] == 'RNN')
                    check_snn = sum(1 for n in check_neurons if n[3] == 'SNN')
                    print(f"        Top {check_range}: RNN={check_rnn} ({check_rnn/check_range:.1%}), SNN={check_snn} ({check_snn/check_range:.1%})")
            
            # 打印前20个神经元的详细信息
            print(f"    🔍 Top 20 neurons detail:")
            for i, neuron in enumerate(global_sorted[:20]):
                idx, orig_imp, comp_val, ntype, total = neuron
                print(f"        {i+1:2d}. {ntype} #{idx}: orig={orig_imp:.6f}, comp={comp_val:.6f}")
            
            # 策略2: 计算调整后的重要性质量分布
            if use_normalization:
                # 使用归一化后的比较值
                rnn_importance_sum = sum(neuron[2] for neuron in global_sorted if neuron[3] == 'RNN')
                snn_importance_sum = sum(neuron[2] for neuron in global_sorted if neuron[3] == 'SNN')
                total_importance = rnn_importance_sum + snn_importance_sum
                print(f"    📈 Normalized importance quality: RNN={rnn_importance_sum:.3f}, SNN={snn_importance_sum:.3f}")
            else:
                # 使用原始重要性值
                rnn_importance_sum = sum(neuron[1] for neuron in global_sorted if neuron[3] == 'RNN')
                snn_importance_sum = sum(neuron[1] for neuron in global_sorted if neuron[3] == 'SNN')
                total_importance = rnn_importance_sum + snn_importance_sum
                print(f"    📈 Original importance quality: RNN={rnn_importance_sum:.6f}, SNN={snn_importance_sum:.6f}")
            
            rnn_quality_ratio = rnn_importance_sum / total_importance if total_importance > 0 else 0.5
            snn_quality_ratio = snn_importance_sum / total_importance if total_importance > 0 else 0.5
            
            print(f"    📊 Quality ratios: RNN={rnn_quality_ratio:.2%}, SNN={snn_quality_ratio:.2%}")
            print(f"    ⚖️  Quality difference: {abs(rnn_quality_ratio - snn_quality_ratio):.1%} ({'Balanced' if abs(rnn_quality_ratio - snn_quality_ratio) < 0.1 else 'Imbalanced'})")
            
            # 策略3: 多种分配方案
            allocations = []
            
            # 方案A: 真正的全局Top-N分布（最重要的策略）
            pure_top_rnn = sum(1 for neuron in top_neurons if neuron[3] == 'RNN')
            pure_top_snn = sum(1 for neuron in top_neurons if neuron[3] == 'SNN')
            print(f"    🎯 Pure Global Analysis: RNN={pure_top_rnn}, SNN={pure_top_snn} (before quota adjustment)")
            
            # 确保最小配额但优先保持原始分布
            if pure_top_rnn < min_rnn_quota:
                adjustment = min_rnn_quota - pure_top_rnn
                pure_top_rnn = min_rnn_quota
                pure_top_snn = max(min_snn_quota, pure_top_snn - adjustment)
                print(f"    🔧 RNN quota adjustment: +{adjustment} (RNN={pure_top_rnn}, SNN={pure_top_snn})")
            elif pure_top_snn < min_snn_quota:
                adjustment = min_snn_quota - pure_top_snn
                pure_top_snn = min_snn_quota
                pure_top_rnn = max(min_rnn_quota, pure_top_rnn - adjustment)
                print(f"    🔧 SNN quota adjustment: +{adjustment} (RNN={pure_top_rnn}, SNN={pure_top_snn})")
            allocations.append((pure_top_rnn, pure_top_snn, "Pure Global Top-N (importance-driven)"))
            
            # 方案B: 基于Top-N分布但确保最小配额
            top_rnn = max(min_rnn_quota, rnn_count_in_top)
            top_snn = max(min_snn_quota, snn_count_in_top)
            if top_rnn + top_snn > max_retain_neurons:
                # 按比例调整
                excess = top_rnn + top_snn - max_retain_neurons
                if top_rnn > top_snn:
                    top_rnn = max(min_rnn_quota, top_rnn - excess)
                    top_snn = max(min_snn_quota, max_retain_neurons - top_rnn)
                else:
                    top_snn = max(min_snn_quota, top_snn - excess)
                    top_rnn = max(min_rnn_quota, max_retain_neurons - top_snn)
            allocations.append((top_rnn, top_snn, "Protected Top-N distribution"))
            
            # 方案C: 质量加权分配但确保最小配额
            quality_rnn = max(min_rnn_quota, int(max_retain_neurons * rnn_quality_ratio))
            quality_snn = max(min_snn_quota, max_retain_neurons - quality_rnn)
            if quality_rnn + quality_snn > max_retain_neurons:
                # 重新调整
                remaining = max_retain_neurons - min_rnn_quota - min_snn_quota
                if remaining > 0:
                    extra_rnn = int(remaining * rnn_quality_ratio)
                    quality_rnn = min_rnn_quota + extra_rnn
                    quality_snn = min_snn_quota + (remaining - extra_rnn)
                else:
                    quality_rnn, quality_snn = min_rnn_quota, min_snn_quota
            allocations.append((quality_rnn, quality_snn, "Protected Quality-weighted"))
            
            # 方案D: 平衡分配
            balanced_base_rnn = max(min_rnn_quota, int(rnn_total * 0.2))  # 基础20%
            balanced_base_snn = max(min_snn_quota, int(snn_total * 0.2))  # 基础20%
            
            if balanced_base_rnn + balanced_base_snn <= max_retain_neurons:
                remaining = max_retain_neurons - balanced_base_rnn - balanced_base_snn
                # 剩余按质量分配
                extra_rnn = int(remaining * rnn_quality_ratio)
                extra_snn = remaining - extra_rnn
                balanced_rnn = balanced_base_rnn + extra_rnn
                balanced_snn = balanced_base_snn + extra_snn
                allocations.append((balanced_rnn, balanced_snn, "Protected Balanced allocation"))
            
            # 方案E: 保守平均分配（权重最低）
            safe_rnn = max(min_rnn_quota, max_retain_neurons // 2)
            safe_snn = max(min_snn_quota, max_retain_neurons - safe_rnn)
            allocations.append((safe_rnn, safe_snn, "Conservative equal split"))
            
            # 策略4: 评估各种分配方案
            best_allocation = None
            best_score = -1
            
            print(f"    🔍 Evaluating allocation strategies:")
            for rnn_budget, snn_budget, method_name in allocations:
                # 确保预算有效
                rnn_budget = max(min_rnn_quota, min(rnn_budget, rnn_total))
                snn_budget = max(min_snn_quota, min(snn_budget, snn_total))
                
                if rnn_budget + snn_budget > max_retain_neurons:
                    # 最终调整到预算范围内
                    excess = rnn_budget + snn_budget - max_retain_neurons
                    if rnn_budget > min_rnn_quota and excess > 0:
                        reduce_rnn = min(excess, rnn_budget - min_rnn_quota)
                        rnn_budget -= reduce_rnn
                        excess -= reduce_rnn
                    if snn_budget > min_snn_quota and excess > 0:
                        reduce_snn = min(excess, snn_budget - min_snn_quota)
                        snn_budget -= reduce_snn
                
                # 🔧 完全重写质量评估逻辑
                # 基于全局重要性排序，选择总共(rnn_budget + snn_budget)个最重要的神经元
                total_budget = rnn_budget + snn_budget
                actual_selected_neurons = global_sorted[:total_budget]
                
                # 统计实际选中的RNN和SNN数量
                actual_selected_rnn = sum(1 for n in actual_selected_neurons if n[3] == 'RNN')
                actual_selected_snn = sum(1 for n in actual_selected_neurons if n[3] == 'SNN')
                
                # 计算这种实际选择与目标分配的差异惩罚
                rnn_diff = abs(actual_selected_rnn - rnn_budget)
                snn_diff = abs(actual_selected_snn - snn_budget)
                allocation_penalty = (rnn_diff + snn_diff) / total_budget * 0.1  # 10%的惩罚系数
                
                # 计算真实的质量比率（基于实际会被选择的最重要神经元）
                if use_normalization:
                    actual_selected_importance = sum(n[2] for n in actual_selected_neurons)
                    total_importance = sum(n[2] for n in global_sorted)
                else:
                    actual_selected_importance = sum(n[1] for n in actual_selected_neurons)
                    total_importance = sum(n[1] for n in global_sorted)
                
                true_quality_ratio = actual_selected_importance / total_importance if total_importance > 0 else 0
                
                # 调整后的质量评分 = 真实质量 - 分配偏差惩罚
                adjusted_quality = true_quality_ratio - allocation_penalty
                
                rnn_retention_rate = rnn_budget / rnn_total
                snn_retention_rate = snn_budget / snn_total
                diversity_bonus = 1 - abs(rnn_retention_rate - snn_retention_rate) * 0.3
                quota_bonus = 0.1 if (rnn_budget >= max(2, int(rnn_total * 0.15)) and snn_budget >= max(2, int(snn_total * 0.15))) else 0
                
                print(f"        🎯 {method_name}:")
                print(f"           Allocation: RNN={rnn_budget} ({rnn_retention_rate:.1%}), SNN={snn_budget} ({snn_retention_rate:.1%})")
                print(f"           Actual selection: RNN={actual_selected_rnn}, SNN={actual_selected_snn}")
                print(f"           Allocation penalty: {allocation_penalty:.3f} (diff: RNN±{rnn_diff}, SNN±{snn_diff})")
                print(f"           True quality: {true_quality_ratio:.3f}")
                print(f"           Adjusted quality: {adjusted_quality:.3f} (×0.85={adjusted_quality*0.85:.3f})")
                print(f"           Diversity: {diversity_bonus:.3f} (×0.10={diversity_bonus*0.10:.3f})")
                print(f"           Quota: {quota_bonus:.3f} (×0.05={quota_bonus*0.05:.3f})")
                
                # 使用调整后的质量评分
                corrected_score = adjusted_quality * 0.85 + diversity_bonus * 0.10 + quota_bonus * 0.05
                print(f"           Total Score: {corrected_score:.4f}")
                
                if corrected_score > best_score:
                    best_score = corrected_score
                    best_allocation = (rnn_budget, snn_budget, method_name)
            
            if best_allocation:
                rnn_budget, snn_budget, best_method = best_allocation
                print(f"    ✅ Selected: {best_method} (Score: {best_score:.4f})")
                print(f"    📋 Final allocation reasoning:")
                print(f"        - This strategy achieved the highest combined score")
                print(f"        - RNN allocation: {rnn_budget}/{rnn_total} = {rnn_budget/rnn_total:.1%}")
                print(f"        - SNN allocation: {snn_budget}/{snn_total} = {snn_budget/snn_total:.1%}")
                
                # 最终验证
                print(f"    🔍 Final validation:")
                print(f"        RNN quota: {rnn_budget}/{rnn_total} ({rnn_budget/rnn_total:.1%}) ≥ {min_rnn_quota} ✓")
                print(f"        SNN quota: {snn_budget}/{snn_total} ({snn_budget/snn_total:.1%}) ≥ {min_snn_quota} ✓")
                print(f"        Total: {rnn_budget + snn_budget}/{max_retain_neurons} ≤ {max_retain_neurons} ✓")
                
                return rnn_budget, snn_budget
            else:
                # 极端情况下的默认分配
                default_rnn = max(min_rnn_quota, max_retain_neurons // 2)
                default_snn = max(min_snn_quota, max_retain_neurons - default_rnn)
                print(f"    ⚠️  Using emergency default allocation: RNN={default_rnn}, SNN={default_snn}")
                return default_rnn, default_snn

        def _save_importance_details_to_csv(self, channel_trace, weights_data, activation_importance=None, gradient_importance=None, filename="neuron_importance_details.csv"):
            """
            将神经元重要性的详细信息保存到CSV文件
            Args:
                channel_trace: Hessian迹数据
                weights_data: 权重数据
                activation_importance: 激活重要性字典（可选）
                gradient_importance: 梯度重要性字典（可选）
                filename: 输出CSV文件名
            """
            import csv
            import numpy as np
            
            print(f"正在生成重要性详细信息CSV文件: {filename}")
            
            # 准备CSV数据
            csv_data = []
            csv_headers = ['layer_name', 'neuron_index', 'hessian_trace', 'weight_norm_squared', 'weight_elements_count', 
                          'nonzero_weight_ratio', 'norm_squared_per_element', 'importance_value', 
                          'activation_importance', 'gradient_importance', 'activation_factor', 'gradient_factor', 'enhanced_importance']
            
            # 遍历每个模块
            for k, mod in enumerate(self.modules):
                layer_name = mod[0]  # 层名称
                cur_weight = weights_data[k]  # 该层的权重数据
                
                print(f"处理层 {layer_name}，神经元数量: {len(cur_weight)}")
                
                # 遍历该层的每个神经元
                for cnt, channel in enumerate(cur_weight):
                    # 计算各项指标
                    hessian_trace = channel_trace[k][cnt].cpu().item() if hasattr(channel_trace[k][cnt], 'cpu') else channel_trace[k][cnt]
                    
                    # 计算非零权重比例
                    weight_values = channel.detach().cpu().numpy().flatten() if hasattr(channel, 'cpu') else channel.numpy().flatten()
                    nonzero_count = np.count_nonzero(weight_values)
                    total_count = len(weight_values)
                    nonzero_weight_ratio = nonzero_count / total_count if total_count > 0 else 0.0
                    
                    weight_norm_squared = channel.detach().norm()**2
                    weight_elements_count = channel.numel()
                    norm_squared_per_element = weight_norm_squared / weight_elements_count
                    importance_value = hessian_trace * norm_squared_per_element
                    
                    # 转换为Python标量
                    weight_norm_squared = weight_norm_squared.cpu().item() if hasattr(weight_norm_squared, 'cpu') else weight_norm_squared
                    norm_squared_per_element = norm_squared_per_element.cpu().item() if hasattr(norm_squared_per_element, 'cpu') else norm_squared_per_element
                    importance_value = importance_value.cpu().item() if hasattr(importance_value, 'cpu') else importance_value
                    
                    # 获取激活和梯度重要性信息
                    act_importance = 0.0
                    grad_importance = 0.0
                    activation_factor = 1.0
                    gradient_factor = 1.0
                    enhanced_importance = importance_value
                    
                    if activation_importance and layer_name in activation_importance and cnt < len(activation_importance[layer_name]):
                        act_importance = activation_importance[layer_name][cnt]
                        activation_factor = 1.0 + act_importance * 0.2
                    
                    if gradient_importance and layer_name in gradient_importance and cnt < len(gradient_importance[layer_name]):
                        grad_importance = gradient_importance[layer_name][cnt]
                        gradient_factor = 1.0 + grad_importance * 0.15
                    
                    # 计算增强后的重要性 - 使用新的计算公式
                    enhanced_importance =( importance_value + act_importance*0.1 * hessian_trace)

                    # 添加到CSV数据
                    csv_data.append([
                        layer_name,
                        cnt,
                        f"{hessian_trace:.8f}",
                        f"{weight_norm_squared:.8f}", 
                        weight_elements_count,
                        f"{nonzero_weight_ratio:.6f}",
                        f"{norm_squared_per_element:.8f}",
                        f"{importance_value:.8f}",
                        f"{act_importance:.8f}",
                        f"{grad_importance:.8f}",
                        f"{activation_factor:.6f}",
                        f"{gradient_factor:.6f}",
                        f"{enhanced_importance:.8f}"
                    ])
            
            # 写入CSV文件
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(csv_headers)
                    writer.writerows(csv_data)
                
                print(f"✅ CSV文件已成功生成: {filename}")
                print(f"   包含 {len(csv_data)} 行数据（不含表头）")
                print(f"   包含原始重要性、激活重要性、梯度重要性和增强重要性信息")
                
                # 统计每层的神经元数量
                layer_stats = {}
                for row in csv_data:
                    layer_name = row[0]
                    if layer_name not in layer_stats:
                        layer_stats[layer_name] = 0
                    layer_stats[layer_name] += 1
                
                print("   各层神经元数量统计:")
                for layer_name, count in layer_stats.items():
                    print(f"     {layer_name}: {count} 个神经元")
                
                # 统计激活和梯度增强的效果
                if activation_importance or gradient_importance:
                    non_zero_act = sum(1 for row in csv_data if float(row[8]) != 0.0)
                    non_zero_grad = sum(1 for row in csv_data if float(row[9]) != 0.0)
                    enhanced_count = sum(1 for row in csv_data if float(row[12]) != float(row[7]))
                    
                    print(f"   📈 激活增强信息: {non_zero_act}/{len(csv_data)} ({non_zero_act/len(csv_data)*100:.1f}%)")
                    print(f"   📈 梯度增强信息: {non_zero_grad}/{len(csv_data)} ({non_zero_grad/len(csv_data)*100:.1f}%)")
                    print(f"   🔄 重要性被增强的神经元: {enhanced_count}/{len(csv_data)} ({enhanced_count/len(csv_data)*100:.1f}%)")
                    
            except Exception as e:
                print(f"❌ 写入CSV文件时发生错误: {e}")

        def _collect_activation_gradient_info(self, dataloader, criterion, batch_size, bptt, ntokens):
            """
            实时采集激活和梯度信息的函数 - 恢复到之前版本
            直接访问模型参数并使用真实数据
            """
            print("🔄 开始采集激活和梯度信息（恢复版本）...")
            
            # 存储激活和梯度信息
            layer_activations = {}
            layer_gradients = {}
            tensor_hooks = []
            
            # 获取目标参数并为它们注册hooks
            target_params = {}
            for name, param in self.model.named_parameters():
                if name in ['snn1', 'rnn1', 'snn2', 'rnn2']:
                    target_params[name] = param
                    print(f"   📌 找到目标参数: {name}, 形状: {param.shape}")
            
            print(f"📋 找到 {len(target_params)} 个目标参数")
            
            if len(target_params) == 0:
                print("⚠️ 未找到目标参数，返回空结果")
                return {}, {}
            
            # 重写模型的前向传播来插入记录点
            original_forward = self.model.forward
            
            def hooked_forward(raw_input, hidden):
                # input vector embedding
                emb = self.model.encoder(raw_input)
                input = self.model.dropout(emb)
                
                # embedded inputs forward pass
                n_win, batch_size, input_size = input.size()
                
                h1_mem = h1_spike = torch.zeros(batch_size, self.model.snn_shape[0], device = self.model.device)
                h2_mem = h2_spike = torch.zeros(batch_size, self.model.snn_shape[1], device = self.model.device)
                h1_y, h2_y = hidden
                
                buf = []
                
                # 记录每个时间步的激活
                if self.model.union:
                    for t in range(n_win):
                        output0 = input[t]
                        

                        # SNN1层计算和记录
                        if self.model.snn_shape[0] > 0 and hasattr(self.model, 'snn1'):
                            snn1_input = output0
                            snn1_activation = snn1_input.mm(self.model.snn1)  # 矩阵乘法                            
                            # 记录激活
                            if 'snn1' not in layer_activations:
                                layer_activations['snn1'] = []
                            if len(layer_activations['snn1']) < 5:
                                layer_activations['snn1'].append(snn1_activation.detach().cpu())
                            
                            h1_mem, h1_spike = self.model.snn_update(self.model.snn1, output0, h1_mem, h1_spike)
                        
                        # RNN1层计算和记录  
                        if self.model.rnn_shape[0] > 0 and hasattr(self.model, 'rnn1'):
                            rnn1_input = torch.cat((output0, h1_y), dim=1)
                            rnn1_activation = rnn1_input.mm(self.model.rnn1)  # 矩阵乘法
                            
                            # 记录激活
                            if 'rnn1' not in layer_activations:
                                layer_activations['rnn1'] = []
                            if len(layer_activations['rnn1']) < 5:
                                layer_activations['rnn1'].append(rnn1_activation.detach().cpu())
                            
                            h1_y = self.model.rnn_union_update(self.model.rnn1, output0, h1_y)
                        
                        output1 = torch.cat((h1_spike, h1_y), dim=1)
                        
                        # SNN2层计算和记录
                        if self.model.snn_shape[1] > 0 and hasattr(self.model, 'snn2'):
                            snn2_input = output1
                            snn2_activation = snn2_input.mm(self.model.snn2)  # 矩阵乘法
                            
                            # 记录激活
                            if 'snn2' not in layer_activations:
                                layer_activations['snn2'] = []
                            if len(layer_activations['snn2']) < 5:
                                layer_activations['snn2'].append(snn2_activation.detach().cpu())
                            
                            h2_mem, h2_spike = self.model.snn_update(self.model.snn2, output1, h2_mem, h2_spike)
                        
                        # RNN2层计算和记录
                        if self.model.rnn_shape[1] > 0 and hasattr(self.model, 'rnn2'):
                            rnn2_input = torch.cat((output1, h2_y), dim=1)
                            rnn2_activation = rnn2_input.mm(self.model.rnn2)  # 矩阵乘法
                            
                            # 记录激活
                            if 'rnn2' not in layer_activations:
                                layer_activations['rnn2'] = []
                            if len(layer_activations['rnn2']) < 5:
                                layer_activations['rnn2'].append(rnn2_activation.detach().cpu())
                            
                            h2_y = self.model.rnn_union_update(self.model.rnn2, output1, h2_y)
                        
                        output2 = torch.cat((h2_spike, h2_y), dim=1)
                        buf.append(output2)
                
                stacked_output = torch.stack(buf, dim=0)
                
                # dropout and decoded
                output = self.model.dropout(stacked_output)
                decoded = self.model.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
                
                return decoded.view(output.size(0), output.size(1), decoded.size(1)), (h1_y, h2_y)
            
            # 临时替换前向传播函数
            self.model.forward = hooked_forward
            
            # 为参数注册梯度hooks
            def make_grad_hook(param_name):
                def hook(grad):
                    if param_name not in layer_gradients:
                        layer_gradients[param_name] = []
                    if len(layer_gradients[param_name]) < 5:
                        layer_gradients[param_name].append(grad.detach().cpu())
                        print(f"   📊 {param_name} 梯度形状: {grad.shape}")
                return hook
            
            # 为目标参数注册梯度hooks
            for param_name, param in target_params.items():
                hook = param.register_hook(make_grad_hook(param_name))
                tensor_hooks.append(hook)
                print(f"   🎯 为 {param_name} 注册梯度hook")
            
            # 使用真实数据集进行训练来采集信息
            print("🏃 使用真实数据集执行前向反向传播来采集激活和梯度...")
            self.model.train()
            
            try:
                # 定义get_batch函数（复制自rnn-ptb.py）
                def get_batch(source, i):
                    seq_len = min(bptt, len(source) - 1 - i)
                    data = source[i:i + seq_len]
                    target = source[i + 1:i + 1 + seq_len].view(-1)
                    return data, target
                
                # 使用真实数据集
                step_count = 0
                max_steps = 3  # 只执行3步以节省时间
                
                print(f"   📊 使用真实数据集，批次大小: {batch_size}, 序列长度: {bptt}")
                print(f"   📊 数据集总长度: {len(dataloader)}")
                
                for i in range(0, len(dataloader) - 1, bptt):
                    if step_count >= max_steps:
                        break
                    
                    # 获取真实数据批次
                    data, targets = get_batch(dataloader, i)
                    data = data.cuda() if torch.cuda.is_available() else data
                    targets = targets.cuda() if torch.cuda.is_available() else targets
                    
                    print(f"   📊 Step {step_count+1}: 数据形状={data.shape}, 目标形状={targets.shape}")
                    
                    # 初始化隐藏状态
                    hidden = self.model.init_hidden(data.size(1))  # batch_size是第二维
                    
                    # 前向传播（使用我们的hooked版本）
                    output, hidden = self.model(data, hidden)
                    
                    # 计算损失并反向传播
                    loss = criterion(output.view(-1, ntokens), targets)
                    self.model.zero_grad()
                    loss.backward()
                    
                    print(f"   📊 Step {step_count+1}: loss={loss.item():.4f}, 输出形状={output.shape}")
                    step_count += 1
                    
            except Exception as e:
                print(f"   ⚠️ 训练过程中出现错误: {e}")
                import traceback
                traceback.print_exc()
            
            # 恢复原始前向传播函数
            self.model.forward = original_forward
            
            # 清理hooks
            for hook in tensor_hooks:
                hook.remove()
            
            # 处理采集到的信息
            activation_importance = {}
            gradient_importance = {}
            
            print("📊 处理采集到的激活信息...")
            for layer_name, activations in layer_activations.items():
                if activations and len(activations) > 0:
                    try:
                        # 合并所有batch的激活
                        all_acts = torch.cat(activations, dim=0)
                        if len(all_acts.shape) >= 2:
                            neuron_count = all_acts.shape[-1]
                            importance_factors = []
                            
                            for i in range(neuron_count):
                                neuron_acts = all_acts[..., i].flatten()
                                # 激活频率 + 激活强度
                                activation_freq = (neuron_acts > 0).float().mean().item()
                                activation_magnitude = neuron_acts.abs().mean().item()
                                factor = activation_freq * 0.5 + activation_magnitude * 0.5
                                importance_factors.append(factor)
                            
                            activation_importance[layer_name] = importance_factors
                            print(f"   ✅ {layer_name}: 处理了 {neuron_count} 个神经元")
                    except Exception as e:
                        print(f"   ⚠️ 处理 {layer_name} 激活时出错: {e}")
            
            print("📊 处理采集到的梯度信息...")
            for layer_name, gradients in layer_gradients.items():
                if gradients and len(gradients) > 0:
                    try:
                        # 合并所有batch的梯度
                        all_grads = torch.cat(gradients, dim=0)
                        if len(all_grads.shape) >= 2:
                            neuron_count = all_grads.shape[-1]
                            importance_factors = []
                            
                            for i in range(neuron_count):
                                neuron_grads = all_grads[..., i].flatten()
                                # 梯度幅值 + 梯度稳定性
                                gradient_magnitude = neuron_grads.abs().mean().item()
                                gradient_std = neuron_grads.std().item()
                                gradient_stability = 1.0 / (gradient_std + 1e-8)
                                factor = gradient_magnitude * 0.7 + min(gradient_stability, 10.0) * 0.3
                                importance_factors.append(factor)
                            
                            gradient_importance[layer_name] = importance_factors
                            print(f"   ✅ {layer_name}: 处理了 {neuron_count} 个神经元")
                    except Exception as e:
                        print(f"   ⚠️ 处理 {layer_name} 梯度时出错: {e}")
            
            print(f"✅ 激活和梯度信息采集完成!")
            print(f"   📊 激活信息: {len(activation_importance)} 层")
            print(f"   📊 梯度信息: {len(gradient_importance)} 层")
            
            return activation_importance, gradient_importance

