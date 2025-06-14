#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的神经元重要性计算方法
可以直接集成到现有的HessianPruner中，提供更精细的重要性评估

主要改进：
1. 多维度权重分析
2. 激活模式考虑
3. 梯度信息利用
4. 网络结构感知
5. 自适应阈值选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict
import math

class ImprovedImportanceCalculator:
    """
    改进的重要性计算器
    可以替换HessianPruner中的重要性计算部分
    """
    
    def __init__(self, use_activation_analysis=True, use_gradient_analysis=True, 
                 use_structural_analysis=True, activation_samples=100):
        """
        初始化改进的重要性计算器
        
        Args:
            use_activation_analysis: 是否使用激活分析
            use_gradient_analysis: 是否使用梯度分析
            use_structural_analysis: 是否使用结构分析
            activation_samples: 激活分析的样本数量
        """
        self.use_activation_analysis = use_activation_analysis
        self.use_gradient_analysis = use_gradient_analysis
        self.use_structural_analysis = use_structural_analysis
        self.activation_samples = activation_samples
        
        # 存储激活和梯度信息
        self.layer_activations = defaultdict(list)
        self.layer_gradients = defaultdict(list)
        self.hooks = []
    
    def compute_enhanced_importance(self, model, modules, channel_trace, dataloader=None, 
                                  criterion=None, device="cuda"):
        """
        计算增强的神经元重要性
        
        Args:
            model: 要分析的模型
            modules: 模块列表（来自HessianPruner._prepare_model）
            channel_trace: Hessian迹信息
            dataloader: 数据加载器（用于激活和梯度分析）
            criterion: 损失函数
            device: 设备
            
        Returns:
            dict: 增强的重要性分数
        """
        print("🔬 计算增强的神经元重要性...")
        
        # 1. 计算基础重要性（改进版的传统方法）
        print("   📊 计算增强基础重要性...")
        basic_importance = self._compute_enhanced_basic_importance(modules, channel_trace)
        
        # 2. 收集激活和梯度信息（如果启用且有数据）
        activation_importance = {}
        gradient_importance = {}
        
        if dataloader is not None and (self.use_activation_analysis or self.use_gradient_analysis):
            print("   📡 收集运行时信息...")
            self._collect_runtime_info(model, dataloader, criterion, device)
            
            if self.use_activation_analysis:
                print("   🎯 分析激活模式...")
                activation_importance = self._compute_activation_importance()
            
            if self.use_gradient_analysis:
                print("   🌊 分析梯度流...")
                gradient_importance = self._compute_gradient_importance()
        
        # 3. 计算结构重要性
        structural_importance = {}
        if self.use_structural_analysis:
            print("   🏗️ 分析网络结构...")
            structural_importance = self._compute_structural_importance(modules)
        
        # 4. 融合所有重要性分数
        print("   🔀 融合重要性分数...")
        fused_importance = self._fuse_importance_scores(
            basic_importance, activation_importance, 
            gradient_importance, structural_importance
        )
        
        # 清理hooks
        self._cleanup()
        
        return fused_importance
    
    def _compute_enhanced_basic_importance(self, modules, channel_trace):
        """
        计算增强的基础重要性
        改进传统的 Hessian_trace × weight_norm 方法
        """
        enhanced_importance = {}
        
        for k, mod in enumerate(modules):
            tmp = []
            m = mod[0]  # 模块名称
            cur_weight = copy.deepcopy(mod[1].data)
            dims = len(list(cur_weight.size()))
            
            # 维度转换（与原代码保持一致）
            if dims == 2:
                cur_weight = cur_weight.permute(1, 0)
            elif dims == 3:
                cur_weight = cur_weight.permute(2, 0, 1)
            
            for cnt, channel in enumerate(cur_weight):
                # 获取Hessian迹
                hessian_trace = channel_trace[k][cnt]
                
                # 1. 原始重要性（L2范数）
                l2_norm_sq = channel.detach().norm()**2
                original_importance = (hessian_trace * l2_norm_sq / channel.numel()).cpu().item()
                
                # 2. 增强指标
                channel_cpu = channel.detach().cpu()
                
                # L1范数（稀疏性）
                l1_norm = torch.norm(channel_cpu, p=1).item()
                
                # 权重方差（多样性）
                weight_variance = torch.var(channel_cpu).item()
                
                # 非零权重比例（连接有效性）
                nonzero_ratio = (channel_cpu.abs() > 1e-6).float().mean().item()
                
                # 权重熵（信息内容）
                abs_weights = channel_cpu.abs()
                if abs_weights.sum() > 1e-8:
                    normalized_weights = abs_weights / abs_weights.sum()
                    weight_entropy = -(normalized_weights * torch.log(normalized_weights + 1e-8)).sum().item()
                else:
                    weight_entropy = 0.0
                
                # 权重动态范围
                weight_range = (channel_cpu.max() - channel_cpu.min()).item()
                
                # 权重的有效维度（简化版PCA）
                if len(channel_cpu.shape) > 1:
                    try:
                        U, S, V = torch.svd(channel_cpu.unsqueeze(0))
                        effective_rank = (S > S.max() * 0.01).sum().item()
                    except:
                        effective_rank = 1.0
                else:
                    effective_rank = 1.0
                
                # 3. 自适应权重计算
                # 根据Hessian迹的大小调整各个指标的权重
                hessian_magnitude = abs(hessian_trace.item())
                
                if hessian_magnitude > 1e-3:  # 高敏感性神经元
                    # 更重视原始重要性和权重大小
                    enhanced_importance_score = (
                        original_importance * 0.5 +        # 原始重要性
                        l2_norm_sq.item() * 0.2 +          # 权重大小
                        weight_variance * 0.15 +           # 权重多样性
                        nonzero_ratio * 0.1 +              # 连接有效性
                        weight_entropy * 0.05              # 信息内容
                    )
                elif hessian_magnitude > 1e-6:  # 中等敏感性神经元
                    # 平衡考虑各个因素
                    enhanced_importance_score = (
                        original_importance * 0.3 +        # 原始重要性
                        weight_variance * 0.25 +           # 权重多样性
                        nonzero_ratio * 0.2 +              # 连接有效性
                        weight_entropy * 0.15 +            # 信息内容
                        effective_rank * 0.1               # 表示能力
                    )
                else:  # 低敏感性神经元
                    # 更重视权重的统计特性
                    enhanced_importance_score = (
                        weight_variance * 0.35 +           # 权重多样性
                        nonzero_ratio * 0.25 +             # 连接有效性
                        weight_entropy * 0.2 +             # 信息内容
                        weight_range * 0.1 +               # 动态范围
                        original_importance * 0.1          # 原始重要性
                    )
                
                tmp.append(enhanced_importance_score)
            
            # 提取层名（如snn_fc1, rnn_fc1等）
            layer_name = m
            enhanced_importance[layer_name] = tmp
        
        return enhanced_importance
    
    def _collect_runtime_info(self, model, dataloader, criterion, device):
        """收集运行时的激活和梯度信息"""
        self.layer_activations.clear()
        self.layer_gradients.clear()
        
        # 注册hooks
        def make_activation_hook(name):
            def hook(module, input, output):
                if len(self.layer_activations[name]) < 20:  # 限制样本数量
                    if isinstance(output, torch.Tensor):
                        self.layer_activations[name].append(output.detach().cpu())
            return hook
        
        def make_gradient_hook(name):
            def hook(module, grad_input, grad_output):
                if len(self.layer_gradients[name]) < 20:  # 限制样本数量
                    if grad_output[0] is not None:
                        self.layer_gradients[name].append(grad_output[0].detach().cpu())
            return hook
        
        # 为关键层注册hooks
        target_layers = ['snn_fc1', 'rnn_fc1', 'snn_fc2', 'rnn_fc2']
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and any(layer in name for layer in target_layers):
                if self.use_activation_analysis:
                    hook = module.register_forward_hook(make_activation_hook(name))
                    self.hooks.append(hook)
                if self.use_gradient_analysis:
                    hook = module.register_backward_hook(make_gradient_hook(name))
                    self.hooks.append(hook)
        
        # 收集数据
        model.train() if self.use_gradient_analysis else model.eval()
        sample_count = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            if sample_count >= self.activation_samples:
                break
            
            if len(batch_data) == 2:
                data, targets = batch_data
            else:
                data, targets = batch_data[0], batch_data[1]
            
            data = data.to(device)
            targets = targets.to(device)
            
            # 前向传播
            hidden = model.init_hidden(data.size(1))
            output, hidden = model(data, hidden)
            
            # 反向传播（如果需要梯度）
            if self.use_gradient_analysis and criterion is not None:
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
                model.zero_grad()
                loss.backward()
            
            sample_count += data.size(1)
            
            if batch_idx >= 10:  # 限制批次数量
                break
    
    def _compute_activation_importance(self):
        """基于激活模式计算重要性"""
        activation_importance = {}
        
        for layer_name, activations in self.layer_activations.items():
            if not activations or len(activations) == 0:
                continue
            
            # 简化层名映射
            simple_name = self._simplify_layer_name(layer_name)
            if simple_name is None:
                continue
            
            try:
                # 拼接所有激活
                all_activations = torch.cat(activations, dim=0)
                if len(all_activations.shape) < 2:
                    continue
                
                neuron_importance = []
                
                # 分析每个神经元的激活模式
                for neuron_idx in range(all_activations.shape[-1]):
                    neuron_acts = all_activations[..., neuron_idx].flatten()
                    
                    # 激活频率
                    activation_freq = (neuron_acts > 0).float().mean().item()
                    
                    # 激活强度
                    activation_magnitude = neuron_acts.abs().mean().item()
                    
                    # 激活稳定性（负变异系数）
                    activation_std = neuron_acts.std().item()
                    activation_mean = neuron_acts.mean().item()
                    if abs(activation_mean) > 1e-8:
                        activation_stability = 1.0 / (abs(activation_std / activation_mean) + 1e-8)
                    else:
                        activation_stability = 1.0
                    
                    # 激活动态范围
                    activation_range = neuron_acts.max().item() - neuron_acts.min().item()
                    
                    # 综合激活重要性
                    importance = (
                        activation_freq * 0.2 +
                        activation_magnitude * 0.3 +
                        activation_stability * 0.2 +
                        activation_range * 0.3
                    )
                    
                    neuron_importance.append(importance)
                
                activation_importance[simple_name] = neuron_importance
                
            except Exception as e:
                print(f"     ⚠️ 激活分析失败 {layer_name}: {e}")
                continue
        
        return activation_importance
    
    def _compute_gradient_importance(self):
        """基于梯度流计算重要性"""
        gradient_importance = {}
        
        for layer_name, gradients in self.layer_gradients.items():
            if not gradients or len(gradients) == 0:
                continue
            
            simple_name = self._simplify_layer_name(layer_name)
            if simple_name is None:
                continue
            
            try:
                # 拼接所有梯度
                all_gradients = torch.cat(gradients, dim=0)
                if len(all_gradients.shape) < 2:
                    continue
                
                neuron_importance = []
                
                for neuron_idx in range(all_gradients.shape[-1]):
                    neuron_grads = all_gradients[..., neuron_idx].flatten()
                    
                    # 梯度幅值
                    gradient_magnitude = neuron_grads.abs().mean().item()
                    
                    # 梯度一致性
                    sign_changes = (neuron_grads[1:] * neuron_grads[:-1] < 0).float().mean().item()
                    gradient_consistency = 1.0 - sign_changes
                    
                    # 梯度信噪比
                    signal = neuron_grads.abs().mean().item()
                    noise = neuron_grads.std().item()
                    snr = signal / (noise + 1e-8)
                    
                    # 综合梯度重要性
                    importance = (
                        gradient_magnitude * 0.5 +
                        gradient_consistency * 0.3 +
                        min(snr, 10.0) * 0.2  # 限制SNR的影响
                    )
                    
                    neuron_importance.append(importance)
                
                gradient_importance[simple_name] = neuron_importance
                
            except Exception as e:
                print(f"     ⚠️ 梯度分析失败 {layer_name}: {e}")
                continue
        
        return gradient_importance
    
    def _compute_structural_importance(self, modules):
        """计算基于网络结构的重要性"""
        structural_importance = {}
        
        # 层的位置权重
        layer_weights = {
            'snn_fc1': 0.8,  # 第一层SNN
            'rnn_fc1': 0.8,  # 第一层RNN
            'snn_fc2': 0.9,  # 第二层SNN
            'rnn_fc2': 0.9   # 第二层RNN
        }
        
        # 网络类型权重
        network_weights = {
            'snn': 0.7,  # SNN的离散特性
            'rnn': 0.8   # RNN的连续特性
        }
        
        for k, mod in enumerate(modules):
            layer_name = mod[0]
            weights = mod[1].data
            
            # 确定层的位置和网络类型
            position_weight = layer_weights.get(layer_name, 0.5)
            network_type = 'snn' if 'snn' in layer_name else 'rnn'
            network_weight = network_weights.get(network_type, 0.5)
            
            # 计算每个神经元的结构重要性
            neuron_importance = []
            for neuron_idx in range(weights.shape[0]):
                # 基础结构重要性
                base_importance = position_weight * network_weight
                
                # 连接密度（该神经元参与的连接数量的相对比例）
                connection_density = 1.0  # 在全连接层中，每个神经元都有相同的连接数
                
                # 层间重要性（考虑cascade连接）
                if 'fc2' in layer_name:
                    # 第二层接收来自第一层的mixed信息，更重要
                    layer_importance = 1.2
                else:
                    layer_importance = 1.0
                
                structural_score = base_importance * connection_density * layer_importance
                neuron_importance.append(structural_score)
            
            structural_importance[layer_name] = neuron_importance
        
        return structural_importance
    
    def _simplify_layer_name(self, layer_name):
        """简化层名，映射到标准格式"""
        if 'snn_fc1' in layer_name or 'snn.0' in layer_name:
            return 'snn_fc1'
        elif 'rnn_fc1' in layer_name or 'rnn.0' in layer_name:
            return 'rnn_fc1'
        elif 'snn_fc2' in layer_name or 'snn.1' in layer_name:
            return 'snn_fc2'
        elif 'rnn_fc2' in layer_name or 'rnn.1' in layer_name:
            return 'rnn_fc2'
        else:
            return None
    
    def _fuse_importance_scores(self, basic_importance, activation_importance, 
                               gradient_importance, structural_importance):
        """融合多种重要性分数"""
        fused_importance = {}
        
        # 定义权重
        weights = {
            'basic': 0.4,       # 基础重要性
            'activation': 0.25, # 激活重要性
            'gradient': 0.2,    # 梯度重要性
            'structural': 0.15  # 结构重要性
        }
        
        # 获取所有层
        all_layers = set(basic_importance.keys())
        
        for layer in all_layers:
            # 获取基础重要性
            basic_scores = basic_importance.get(layer, [])
            num_neurons = len(basic_scores)
            
            if num_neurons == 0:
                continue
            
            # 初始化融合分数
            fused_scores = np.array(basic_scores) * weights['basic']
            total_weight = weights['basic']
            
            # 添加激活重要性
            if layer in activation_importance:
                act_scores = activation_importance[layer]
                if len(act_scores) == num_neurons:
                    # 归一化到与基础重要性相同的尺度
                    act_array = np.array(act_scores)
                    if act_array.max() > act_array.min():
                        act_normalized = (act_array - act_array.min()) / (act_array.max() - act_array.min())
                        # 缩放到基础重要性的范围
                        basic_array = np.array(basic_scores)
                        act_scaled = act_normalized * (basic_array.max() - basic_array.min()) + basic_array.min()
                        fused_scores += act_scaled * weights['activation']
                        total_weight += weights['activation']
            
            # 添加梯度重要性
            if layer in gradient_importance:
                grad_scores = gradient_importance[layer]
                if len(grad_scores) == num_neurons:
                    grad_array = np.array(grad_scores)
                    if grad_array.max() > grad_array.min():
                        grad_normalized = (grad_array - grad_array.min()) / (grad_array.max() - grad_array.min())
                        basic_array = np.array(basic_scores)
                        grad_scaled = grad_normalized * (basic_array.max() - basic_array.min()) + basic_array.min()
                        fused_scores += grad_scaled * weights['gradient']
                        total_weight += weights['gradient']
            
            # 添加结构重要性
            if layer in structural_importance:
                struct_scores = structural_importance[layer]
                if len(struct_scores) == num_neurons:
                    struct_array = np.array(struct_scores)
                    basic_array = np.array(basic_scores)
                    # 结构重要性作为乘法因子
                    fused_scores *= struct_array
                    total_weight += weights['structural']
            
            # 归一化
            if total_weight > 0:
                fused_scores /= total_weight
            
            fused_importance[layer] = fused_scores.tolist()
        
        return fused_importance
    
    def _cleanup(self):
        """清理hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.layer_activations.clear()
        self.layer_gradients.clear()

def create_adaptive_retention_strategy():
    """
    创建自适应保留策略
    根据重要性分布自动调整保留率
    """
    def adaptive_retention_rate(importance_scores, network_type='mixed'):
        """
        自适应确定神经元保留率
        
        Args:
            importance_scores: 重要性分数列表
            network_type: 网络类型 ('rnn', 'snn', 'mixed')
            
        Returns:
            float: 建议的保留率
        """
        if not importance_scores:
            return 0.3  # 默认保留率
        
        scores = np.array(importance_scores)
        
        # 计算分布特征
        mean_score = scores.mean()
        std_score = scores.std()
        cv = std_score / (mean_score + 1e-8)  # 变异系数
        
        # 计算重要性集中度
        sorted_scores = np.sort(scores)[::-1]  # 降序
        cumsum_scores = np.cumsum(sorted_scores)
        total_importance = cumsum_scores[-1]
        
        # 计算达到不同重要性阈值需要的神经元比例
        thresholds = [0.7, 0.8, 0.9]
        threshold_ratios = []
        
        for threshold in thresholds:
            target = total_importance * threshold
            idx = np.argmax(cumsum_scores >= target)
            ratio = (idx + 1) / len(scores)
            threshold_ratios.append(ratio)
        
        # 基于分布特征选择保留策略
        if cv > 2.5:  # 极高变异性：重要性高度集中
            base_retention = threshold_ratios[0]  # 70%重要性阈值
        elif cv > 1.5:  # 高变异性：重要性比较集中
            base_retention = threshold_ratios[1]  # 80%重要性阈值
        elif cv > 0.8:  # 中等变异性：重要性分布中等
            base_retention = threshold_ratios[2]  # 90%重要性阈值
        else:  # 低变异性：重要性分布均匀
            base_retention = 0.6  # 保守策略
        
        # 网络类型调整
        if network_type == 'rnn':
            # RNN更稳定，可以更激进地剪枝
            adjusted_retention = base_retention * 0.85
        elif network_type == 'snn':
            # SNN的离散特性更敏感，保守一些
            adjusted_retention = base_retention * 1.15
        else:  # mixed
            adjusted_retention = base_retention
        
        # 确保在合理范围内
        final_retention = np.clip(adjusted_retention, 0.05, 0.8)
        
        return final_retention
    
    return adaptive_retention_rate

# 使用示例
def integrate_with_hessian_pruner():
    """
    展示如何将改进的重要性计算集成到现有的HessianPruner中
    """
    print("🔧 集成改进的重要性计算到HessianPruner...")
    
    # 在HessianPruner的_compute_hessian_importance方法中，
    # 将原来的重要性计算部分替换为：
    
    example_code = '''
    # 在HessianPruner._compute_hessian_importance方法中：
    
    # 原来的代码：
    # for k, mod in enumerate(self.modules):
    #     tmp = []
    #     for cnt, channel in enumerate(cur_weight):
    #         tmp.append((channel_trace[k][cnt] * channel.detach().norm()**2 / channel.numel()).cpu().item())
    #     self.importances[str(m)] = (tmp, len(tmp))
    
    # 替换为：
    improved_calculator = ImprovedImportanceCalculator(
        use_activation_analysis=True,
        use_gradient_analysis=True,
        use_structural_analysis=True
    )
    
    enhanced_importance = improved_calculator.compute_enhanced_importance(
        model=self.model,
        modules=self.modules,
        channel_trace=channel_trace,
        dataloader=dataloader,  # 需要传入
        criterion=criterion,    # 需要传入
        device=device
    )
    
    # 将结果转换为原格式
    for layer_name, importance_scores in enhanced_importance.items():
        self.importances[layer_name] = (importance_scores, len(importance_scores))
    '''
    
    print("📝 集成代码示例:")
    print(example_code)
    
    print("\n📋 集成步骤:")
    print("1. 在HessianPruner.__init__中添加improved_calculator参数")
    print("2. 在_compute_hessian_importance中替换重要性计算部分")
    print("3. 传入dataloader和criterion参数")
    print("4. 可选：使用自适应保留策略替换固定阈值")

if __name__ == "__main__":
    integrate_with_hessian_pruner() 