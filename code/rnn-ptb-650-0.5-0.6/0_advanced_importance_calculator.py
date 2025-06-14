#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级神经元重要性计算器
提供多种精细的神经元重要性评估方法

作者：AI助手
目的：解决当前重要性计算粗粒度的问题，提供更准确的神经元重要性评估
"""

import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm
import copy
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import torch.nn.functional as F

class AdvancedImportanceCalculator:
    """
    高级神经元重要性计算器
    提供多种精细的重要性评估方法
    """
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = torch.device(device)
        self.activation_hooks = {}
        self.gradient_hooks = {}
        self.layer_activations = defaultdict(list)
        self.layer_gradients = defaultdict(list)
        
        # 注册hooks来收集激活和梯度信息
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向hooks来收集激活和梯度信息"""
        def make_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # 只收集前N批数据，避免内存溢出
                    if len(self.layer_activations[name]) < 50:
                        self.layer_activations[name].append(output.detach().cpu())
            return hook
        
        def make_gradient_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    if len(self.layer_gradients[name]) < 50:
                        self.layer_gradients[name].append(grad_output[0].detach().cpu())
            return hook
        
        # 为目标层注册hooks
        target_layers = ['snn_fc1', 'rnn_fc1', 'snn_fc2', 'rnn_fc2']
        for name, module in self.model.named_parameters():
            layer_name = name.split('.')[0]  # 获取层名
            if layer_name in target_layers:
                # 获取对应的模块
                module_obj = self.model
                for attr in name.split('.')[:-1]:
                    module_obj = getattr(module_obj, attr)
                
                if layer_name not in self.activation_hooks:
                    self.activation_hooks[layer_name] = module_obj.register_forward_hook(
                        make_activation_hook(layer_name)
                    )
                    self.gradient_hooks[layer_name] = module_obj.register_backward_hook(
                        make_gradient_hook(layer_name)
                    )
    
    def compute_comprehensive_importance(self, dataloader, criterion, methods=['basic', 'activation', 'gradient_flow', 'information_bottleneck', 'cooperative']):
        """
        计算综合神经元重要性
        
        Args:
            dataloader: 数据加载器
            criterion: 损失函数
            methods: 要使用的重要性计算方法列表
            
        Returns:
            dict: 包含各种重要性分数的字典
        """
        print(f"\n🔬 开始计算高级神经元重要性...")
        print(f"📋 使用方法: {methods}")
        
        importance_scores = {}
        
        # 1. 基础重要性（现有方法的改进版）
        if 'basic' in methods:
            print("🔸 计算基础重要性（改进版）...")
            importance_scores['basic'] = self._compute_enhanced_basic_importance()
        
        # 2. 基于激活的重要性
        if 'activation' in methods:
            print("🔸 计算基于激活的重要性...")
            importance_scores['activation'] = self._compute_activation_importance(dataloader, criterion)
        
        # 3. 梯度流重要性
        if 'gradient_flow' in methods:
            print("🔸 计算梯度流重要性...")
            importance_scores['gradient_flow'] = self._compute_gradient_flow_importance(dataloader, criterion)
        
        # 4. 信息瓶颈重要性
        if 'information_bottleneck' in methods:
            print("🔸 计算信息瓶颈重要性...")
            importance_scores['information_bottleneck'] = self._compute_information_importance(dataloader)
        
        # 5. 协同重要性
        if 'cooperative' in methods:
            print("🔸 计算协同重要性...")
            importance_scores['cooperative'] = self._compute_cooperative_importance(dataloader, criterion)
        
        # 6. 结构化重要性（考虑网络结构）
        if 'structural' in methods:
            print("🔸 计算结构化重要性...")
            importance_scores['structural'] = self._compute_structural_importance()
        
        # 7. 综合重要性（多方法融合）
        print("🔸 融合多种重要性分数...")
        importance_scores['comprehensive'] = self._fuse_importance_scores(importance_scores)
        
        # 清理hooks和缓存
        self._cleanup_hooks()
        
        return importance_scores
    
    def _compute_enhanced_basic_importance(self):
        """
        增强版基础重要性计算
        改进原有的 Hessian_trace × weight_norm 方法
        """
        basic_importance = {}
        
        # 获取权重参数
        target_layers = ['snn_fc1', 'rnn_fc1', 'snn_fc2', 'rnn_fc2']
        
        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name in target_layers and 'weight' in name:
                weights = param.data.cpu()
                
                # 计算多种权重统计量
                importance_per_neuron = []
                
                # 原始维度：[output_neurons, input_neurons]
                # 遍历每个输出神经元（对应一行权重）
                for neuron_idx in range(weights.shape[0]):
                    neuron_weights = weights[neuron_idx]  # 该神经元的权重向量
                    
                    # 1. L2范数（原有方法）
                    l2_norm = torch.norm(neuron_weights, p=2).item()
                    
                    # 2. L1范数（稀疏性指标）
                    l1_norm = torch.norm(neuron_weights, p=1).item()
                    
                    # 3. 权重方差（衡量权重分布的离散程度）
                    weight_var = torch.var(neuron_weights).item()
                    
                    # 4. 非零权重比例（衡量连接的有效性）
                    nonzero_ratio = (neuron_weights.abs() > 1e-6).float().mean().item()
                    
                    # 5. 权重熵（衡量信息内容）
                    normalized_weights = F.softmax(neuron_weights.abs(), dim=0)
                    weight_entropy = -(normalized_weights * torch.log(normalized_weights + 1e-8)).sum().item()
                    
                    # 6. 权重的有效秩（通过SVD分析）
                    if len(neuron_weights.shape) > 1:
                        U, S, V = torch.svd(neuron_weights.unsqueeze(0))
                        effective_rank = (S > S.max() * 0.01).sum().item()
                    else:
                        effective_rank = 1.0
                    
                    # 综合重要性评分
                    enhanced_importance = (
                        l2_norm * 0.3 +                    # 权重大小
                        weight_var * 0.2 +                 # 权重多样性
                        nonzero_ratio * 0.2 +              # 连接有效性
                        weight_entropy * 0.2 +             # 信息内容
                        effective_rank * 0.1               # 表示能力
                    )
                    
                    importance_per_neuron.append(enhanced_importance)
                
                basic_importance[layer_name] = importance_per_neuron
        
        return basic_importance
    
    def _compute_activation_importance(self, dataloader, criterion):
        """
        基于激活模式的重要性计算
        考虑神经元的实际激活情况和信息传递能力
        """
        print("   📊 收集激活数据...")
        
        # 清空之前的激活记录
        self.layer_activations.clear()
        
        self.model.eval()
        activation_importance = {}
        
        # 收集激活数据
        sample_count = 0
        max_samples = 200  # 限制样本数量避免内存问题
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                if sample_count >= max_samples:
                    break
                
                data, targets = data.to(self.device), targets.to(self.device)
                
                # 初始化隐藏状态
                hidden = self.model.init_hidden(data.size(1))
                
                # 前向传播收集激活
                output, hidden = self.model(data, hidden)
                
                sample_count += data.size(1)  # batch_size
        
        print(f"   📈 分析 {len(self.layer_activations)} 层的激活模式...")
        
        # 分析每层的激活重要性
        for layer_name, activations in self.layer_activations.items():
            if not activations:
                continue
                
            # 将所有批次的激活连接起来
            all_activations = torch.cat(activations, dim=0)  # [total_samples, neurons]
            
            importance_per_neuron = []
            
            for neuron_idx in range(all_activations.shape[-1]):
                neuron_activations = all_activations[..., neuron_idx]
                
                # 1. 激活频率（神经元有多频繁被激活）
                activation_freq = (neuron_activations > 0).float().mean().item()
                
                # 2. 激活强度（平均激活值）
                activation_magnitude = neuron_activations.abs().mean().item()
                
                # 3. 激活方差（激活的变化程度）
                activation_variance = neuron_activations.var().item()
                
                # 4. 激活熵（激活模式的复杂性）
                # 将激活值分为不同区间计算熵
                hist, _ = np.histogram(neuron_activations.numpy(), bins=10, density=True)
                hist = hist + 1e-8  # 避免log(0)
                activation_entropy = -(hist * np.log(hist)).sum()
                
                # 5. 激活的判别能力（通过不同类别的激活差异衡量）
                # 这里简化为激活值的动态范围
                activation_range = neuron_activations.max().item() - neuron_activations.min().item()
                
                # 综合激活重要性
                activation_importance_score = (
                    activation_freq * 0.2 +           # 激活频率
                    activation_magnitude * 0.3 +      # 激活强度
                    activation_variance * 0.2 +       # 激活多样性
                    activation_entropy * 0.2 +        # 激活复杂性
                    activation_range * 0.1            # 判别能力
                )
                
                importance_per_neuron.append(activation_importance_score)
            
            activation_importance[layer_name] = importance_per_neuron
        
        return activation_importance
    
    def _compute_gradient_flow_importance(self, dataloader, criterion):
        """
        基于梯度流的重要性计算
        分析梯度在网络中的传播情况
        """
        print("   🌊 分析梯度流模式...")
        
        # 清空梯度记录
        self.layer_gradients.clear()
        
        self.model.train()
        gradient_importance = {}
        
        # 收集梯度数据
        sample_count = 0
        max_samples = 100  # 梯度计算更消耗资源，限制更少样本
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
            
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 前向传播
            hidden = self.model.init_hidden(data.size(1))
            output, hidden = self.model(data, hidden)
            
            # 计算损失并反向传播
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            
            self.model.zero_grad()
            loss.backward()
            
            sample_count += data.size(1)
        
        print(f"   📉 分析 {len(self.layer_gradients)} 层的梯度特征...")
        
        # 分析每层的梯度重要性
        for layer_name, gradients in self.layer_gradients.items():
            if not gradients:
                continue
            
            # 将所有批次的梯度连接起来
            all_gradients = torch.cat(gradients, dim=0)
            
            importance_per_neuron = []
            
            for neuron_idx in range(all_gradients.shape[-1]):
                neuron_gradients = all_gradients[..., neuron_idx]
                
                # 1. 梯度幅值（梯度的平均大小）
                gradient_magnitude = neuron_gradients.abs().mean().item()
                
                # 2. 梯度稳定性（梯度的一致性）
                gradient_stability = 1.0 / (neuron_gradients.var().item() + 1e-8)
                
                # 3. 梯度信噪比
                signal = neuron_gradients.abs().mean().item()
                noise = neuron_gradients.std().item()
                snr = signal / (noise + 1e-8)
                
                # 4. 梯度方向一致性
                sign_consistency = (neuron_gradients > 0).float().mean().item()
                sign_consistency = max(sign_consistency, 1 - sign_consistency)  # 取更一致的方向
                
                # 综合梯度重要性
                gradient_importance_score = (
                    gradient_magnitude * 0.4 +        # 梯度大小
                    gradient_stability * 0.2 +        # 梯度稳定性
                    snr * 0.2 +                       # 信噪比
                    sign_consistency * 0.2            # 方向一致性
                )
                
                importance_per_neuron.append(gradient_importance_score)
            
            gradient_importance[layer_name] = importance_per_neuron
        
        return gradient_importance
    
    def _compute_information_importance(self, dataloader):
        """
        基于信息瓶颈理论的重要性计算
        分析神经元的信息传递和压缩能力
        """
        print("   📡 计算信息瓶颈重要性...")
        
        # 这是一个简化版的信息瓶颈分析
        # 完整版需要估计互信息，计算量很大
        
        information_importance = {}
        
        # 收集输入输出数据用于信息分析
        input_data = []
        output_data = {}
        
        self.model.eval()
        sample_count = 0
        max_samples = 150
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(dataloader):
                if sample_count >= max_samples:
                    break
                
                data = data.to(self.device)
                input_data.append(data.cpu())
                
                hidden = self.model.init_hidden(data.size(1))
                output, hidden = self.model(data, hidden)
                
                sample_count += data.size(1)
        
        # 分析每层的信息传递能力
        for layer_name, activations in self.layer_activations.items():
            if not activations:
                continue
            
            all_activations = torch.cat(activations, dim=0)
            importance_per_neuron = []
            
            for neuron_idx in range(all_activations.shape[-1]):
                neuron_activations = all_activations[..., neuron_idx].numpy()
                
                # 1. 信息熵（神经元输出的信息量）
                # 离散化激活值来计算熵
                hist, _ = np.histogram(neuron_activations, bins=20, density=True)
                hist = hist + 1e-8
                information_entropy = -(hist * np.log(hist)).sum()
                
                # 2. 信息传递效率（基于激活值的变化）
                if len(neuron_activations) > 1:
                    # 计算相邻时刻激活值的相关性
                    autocorr = np.corrcoef(neuron_activations[:-1], neuron_activations[1:])[0, 1]
                    if np.isnan(autocorr):
                        autocorr = 0
                    information_transfer = 1 - abs(autocorr)  # 低相关性意味着高信息传递
                else:
                    information_transfer = 0.5
                
                # 3. 信息压缩能力（通过PCA分析）
                if neuron_activations.var() > 1e-6:
                    # 简化的信息压缩指标：基于激活值的动态范围
                    dynamic_range = neuron_activations.max() - neuron_activations.min()
                    compression_ability = dynamic_range / (neuron_activations.std() + 1e-8)
                else:
                    compression_ability = 0
                
                # 综合信息重要性
                information_importance_score = (
                    information_entropy * 0.4 +           # 信息量
                    information_transfer * 0.3 +          # 传递效率
                    compression_ability * 0.3             # 压缩能力
                )
                
                importance_per_neuron.append(information_importance_score)
            
            information_importance[layer_name] = importance_per_neuron
        
        return information_importance
    
    def _compute_cooperative_importance(self, dataloader, criterion):
        """
        协同重要性计算
        考虑神经元之间的相互作用和协同效应
        """
        print("   🤝 分析神经元协同效应...")
        
        cooperative_importance = {}
        
        # 使用激活数据分析神经元间相关性
        for layer_name, activations in self.layer_activations.items():
            if not activations:
                continue
            
            all_activations = torch.cat(activations, dim=0)  # [samples, neurons]
            num_neurons = all_activations.shape[-1]
            
            if num_neurons < 2:
                cooperative_importance[layer_name] = [0.5] * num_neurons
                continue
            
            # 计算神经元间的相关矩阵
            activations_np = all_activations.numpy()
            correlation_matrix = np.corrcoef(activations_np.T)
            
            # 处理NaN值
            correlation_matrix = np.nan_to_num(correlation_matrix, 0)
            
            importance_per_neuron = []
            
            for neuron_idx in range(num_neurons):
                # 1. 连接强度（与其他神经元的平均相关性）
                neuron_correlations = correlation_matrix[neuron_idx]
                connection_strength = np.abs(neuron_correlations).mean()
                
                # 2. 独特性（低冗余性）
                # 计算该神经元与其他神经元的最大相关性
                max_correlation = np.max(np.abs(neuron_correlations[neuron_correlations != 1.0]))
                uniqueness = 1 - max_correlation
                
                # 3. 中心性（网络中的重要位置）
                # 基于相关性的度中心性
                degree_centrality = np.sum(np.abs(neuron_correlations) > 0.1)
                degree_centrality = degree_centrality / (num_neurons - 1)
                
                # 4. 协同效应（与多个神经元的协作能力）
                # 计算三元组协同效应的简化版本
                cooperation_count = 0
                for i in range(num_neurons):
                    for j in range(i+1, num_neurons):
                        if i != neuron_idx and j != neuron_idx:
                            # 检查三个神经元之间的协同模式
                            corr_ij = correlation_matrix[i, j]
                            corr_ni = correlation_matrix[neuron_idx, i]
                            corr_nj = correlation_matrix[neuron_idx, j]
                            
                            # 如果三者都有一定相关性，认为存在协同效应
                            if abs(corr_ij) > 0.1 and abs(corr_ni) > 0.1 and abs(corr_nj) > 0.1:
                                cooperation_count += 1
                
                cooperation_ability = cooperation_count / max(1, num_neurons * (num_neurons - 1) // 2)
                
                # 综合协同重要性
                cooperative_importance_score = (
                    connection_strength * 0.3 +       # 连接强度
                    uniqueness * 0.3 +               # 独特性
                    degree_centrality * 0.2 +        # 中心性
                    cooperation_ability * 0.2        # 协同能力
                )
                
                importance_per_neuron.append(cooperative_importance_score)
            
            cooperative_importance[layer_name] = importance_per_neuron
        
        return cooperative_importance
    
    def _compute_structural_importance(self):
        """
        结构化重要性计算
        考虑网络的层次结构和信息流
        """
        print("   🏗️ 分析网络结构重要性...")
        
        structural_importance = {}
        
        # 简化的结构重要性：基于层的位置和连接模式
        layer_order = ['snn_fc1', 'rnn_fc1', 'snn_fc2', 'rnn_fc2']
        
        for layer_idx, layer_name in enumerate(layer_order):
            # 检查该层是否存在
            layer_found = False
            for name, param in self.model.named_parameters():
                if layer_name in name and 'weight' in name:
                    layer_found = True
                    weights = param.data.cpu()
                    num_neurons = weights.shape[0]
                    
                    importance_per_neuron = []
                    
                    for neuron_idx in range(num_neurons):
                        # 1. 层位置重要性（早期层和晚期层更重要）
                        if layer_idx <= 1:  # 第一层
                            position_importance = 0.8
                        else:  # 第二层
                            position_importance = 0.9
                        
                        # 2. 扇出重要性（该神经元连接到下一层的程度）
                        # 在混合网络中，每个神经元都会连接到下一层的所有神经元
                        fanout_importance = 1.0  # 简化假设
                        
                        # 3. 路径重要性（该神经元在信息传递路径中的重要性）
                        # RNN和SNN的cascade连接模式
                        if 'rnn' in layer_name:
                            path_importance = 0.8  # RNN提供连续信息
                        else:
                            path_importance = 0.7  # SNN提供离散信息
                        
                        # 综合结构重要性
                        structural_importance_score = (
                            position_importance * 0.4 +
                            fanout_importance * 0.3 +
                            path_importance * 0.3
                        )
                        
                        importance_per_neuron.append(structural_importance_score)
                    
                    structural_importance[layer_name] = importance_per_neuron
                    break
            
            if not layer_found:
                print(f"   ⚠️ 层 {layer_name} 未找到")
        
        return structural_importance
    
    def _fuse_importance_scores(self, importance_scores):
        """
        融合多种重要性分数
        使用加权平均和归一化
        """
        print("   🔄 融合多种重要性分数...")
        
        # 定义各方法的权重
        method_weights = {
            'basic': 0.25,              # 基础重要性
            'activation': 0.25,         # 激活重要性
            'gradient_flow': 0.2,       # 梯度流重要性
            'information_bottleneck': 0.15,  # 信息瓶颈重要性
            'cooperative': 0.1,         # 协同重要性
            'structural': 0.05          # 结构重要性
        }
        
        fused_importance = {}
        
        # 获取所有层的名称
        all_layers = set()
        for method, scores in importance_scores.items():
            all_layers.update(scores.keys())
        
        for layer_name in all_layers:
            # 收集该层所有方法的分数
            layer_scores = {}
            max_neurons = 0
            
            for method, scores in importance_scores.items():
                if layer_name in scores:
                    layer_scores[method] = scores[layer_name]
                    max_neurons = max(max_neurons, len(scores[layer_name]))
            
            if max_neurons == 0:
                continue
            
            # 归一化各方法的分数到[0,1]范围
            normalized_scores = {}
            for method, scores in layer_scores.items():
                scores_array = np.array(scores)
                if scores_array.max() > scores_array.min():
                    normalized = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
                else:
                    normalized = np.ones_like(scores_array) * 0.5
                normalized_scores[method] = normalized
            
            # 加权融合
            fused_scores = np.zeros(max_neurons)
            total_weight = 0
            
            for method, weight in method_weights.items():
                if method in normalized_scores:
                    fused_scores += normalized_scores[method] * weight
                    total_weight += weight
            
            if total_weight > 0:
                fused_scores /= total_weight
            
            fused_importance[layer_name] = fused_scores.tolist()
        
        return fused_importance
    
    def _cleanup_hooks(self):
        """清理注册的hooks"""
        for hook in self.activation_hooks.values():
            hook.remove()
        for hook in self.gradient_hooks.values():
            hook.remove()
        
        self.activation_hooks.clear()
        self.gradient_hooks.clear()
        self.layer_activations.clear()
        self.layer_gradients.clear()
    
    def analyze_importance_quality(self, importance_scores, ground_truth_method='comprehensive'):
        """
        分析重要性计算的质量
        通过多种指标评估重要性分数的合理性
        """
        print(f"\n🔍 分析重要性计算质量...")
        
        quality_metrics = {}
        
        if ground_truth_method not in importance_scores:
            print(f"⚠️ 参考方法 {ground_truth_method} 不存在，使用第一个方法作为参考")
            ground_truth_method = list(importance_scores.keys())[0]
        
        reference_scores = importance_scores[ground_truth_method]
        
        for method_name, method_scores in importance_scores.items():
            if method_name == ground_truth_method:
                continue
            
            method_quality = {}
            
            for layer_name in reference_scores.keys():
                if layer_name in method_scores:
                    ref_scores = np.array(reference_scores[layer_name])
                    method_score_array = np.array(method_scores[layer_name])
                    
                    # 1. 相关性分析
                    correlation = np.corrcoef(ref_scores, method_score_array)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0
                    
                    # 2. 排序一致性（Spearman相关）
                    from scipy.stats import spearmanr
                    rank_correlation, _ = spearmanr(ref_scores, method_score_array)
                    if np.isnan(rank_correlation):
                        rank_correlation = 0
                    
                    # 3. 分布相似性（KL散度）
                    ref_normalized = ref_scores / (ref_scores.sum() + 1e-8)
                    method_normalized = method_score_array / (method_score_array.sum() + 1e-8)
                    
                    # 计算KL散度
                    kl_div = np.sum(ref_normalized * np.log((ref_normalized + 1e-8) / (method_normalized + 1e-8)))
                    
                    method_quality[layer_name] = {
                        'correlation': correlation,
                        'rank_correlation': rank_correlation,
                        'kl_divergence': kl_div,
                        'quality_score': (correlation + rank_correlation) / 2 - kl_div * 0.1
                    }
            
            quality_metrics[method_name] = method_quality
        
        # 打印质量分析结果
        for method_name, method_quality in quality_metrics.items():
            print(f"\n📊 {method_name} 方法质量分析:")
            for layer_name, metrics in method_quality.items():
                print(f"  {layer_name}:")
                print(f"    相关性: {metrics['correlation']:.3f}")
                print(f"    排序相关性: {metrics['rank_correlation']:.3f}")
                print(f"    KL散度: {metrics['kl_divergence']:.3f}")
                print(f"    综合质量分数: {metrics['quality_score']:.3f}")
        
        return quality_metrics 