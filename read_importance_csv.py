import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置matplotlib参数，使用英文标签避免字体问题
plt.rcParams['axes.unicode_minus'] = False

def read_and_analyze_importance_data():
    """读取并分析神经元重要性数据"""
    
    # 创建输出目录
    output_dir = 'analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取CSV文件
    file_path = '/root/autodl-tmp/hstnn-demo/code/rnn-ptb-650-0.5-0.6/neuron_importance_details_trace_hybrid_1111.csv'
    df = pd.read_csv(file_path)
    
    print(f"Total data points: {len(df)}")
    print(f"Number of unique layers: {df['layer_name'].nunique()}")
    print(f"Data count per layer:")
    print(df['layer_name'].value_counts())
    
    # 获取所有唯一的layer_name
    layer_names = df['layer_name'].unique()
    
    # 要分析的指标
    metrics = ['hessian_trace', 'norm_squared_per_element', 'importance_value']
    
    # 创建子图
    fig, axes = plt.subplots(len(metrics), len(layer_names), figsize=(5*len(layer_names), 4*len(metrics)))
    
    # 如果只有一个layer或一个metric，确保axes是二维数组
    if len(layer_names) == 1:
        axes = axes.reshape(-1, 1)
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(layer_names)))
    
    # 为每个指标和每个layer创建分布图
    for i, metric in enumerate(metrics):
        for j, layer in enumerate(layer_names):
            layer_data = df[df['layer_name'] == layer][metric]
            
            # 创建直方图
            axes[i, j].hist(layer_data, bins=30, alpha=0.7, color=colors[j], edgecolor='black')
            axes[i, j].set_title(f'{layer} - {metric}')
            axes[i, j].set_xlabel(metric)
            axes[i, j].set_ylabel('Frequency')
            axes[i, j].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = layer_data.mean()
            std_val = layer_data.std()
            axes[i, j].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.6f}')
            axes[i, j].legend()
    
    plt.tight_layout()
    plt.suptitle('Distribution of Neuron Importance Metrics by Layer', fontsize=16, y=1.02)
    # 保存图片
    plt.savefig(os.path.join(output_dir, 'distribution_histograms.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 新增：创建激活和梯度增强信息的分布图
    enhancement_metrics = ['importance_value', 'activation_factor', 'enhanced_importance']
    
    # 检查这些列是否存在
    missing_cols = [col for col in enhancement_metrics if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}, skipping enhancement analysis")
    else:
        print("Creating enhancement metrics distribution plots...")
        
        # 创建增强指标分布图
        fig_enh, axes_enh = plt.subplots(len(enhancement_metrics), len(layer_names), 
                                        figsize=(5*len(layer_names), 4*len(enhancement_metrics)))
        
        # 如果只有一个layer或一个metric，确保axes是二维数组
        if len(layer_names) == 1:
            axes_enh = axes_enh.reshape(-1, 1)
        if len(enhancement_metrics) == 1:
            axes_enh = axes_enh.reshape(1, -1)
        
        # 为增强指标使用不同的颜色方案
        enhancement_colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
        
        # 为每个增强指标和每个layer创建分布图
        for i, metric in enumerate(enhancement_metrics):
            for j, layer in enumerate(layer_names):
                layer_data = df[df['layer_name'] == layer][metric]
                
                # 创建直方图
                axes_enh[i, j].hist(layer_data, bins=30, alpha=0.7, color=enhancement_colors[j], edgecolor='black')
                axes_enh[i, j].set_title(f'{layer} - {metric}')
                axes_enh[i, j].set_xlabel(metric)
                axes_enh[i, j].set_ylabel('Frequency')
                axes_enh[i, j].grid(True, alpha=0.3)
                
                # 为RNN层的enhanced_importance调整x轴范围
                if metric == 'enhanced_importance' and layer == 'rnn2':
                    mean_val = layer_data.mean()
                    std_val = layer_data.std()
                    axes_enh[i, j].set_xlim(mean_val - 2*std_val, mean_val + 2*std_val)
                
                # 添加统计信息
                mean_val = layer_data.mean()
                std_val = layer_data.std()
                axes_enh[i, j].axvline(mean_val, color='red', linestyle='--', 
                                      label=f'Mean: {mean_val:.6f}')
                axes_enh[i, j].legend()
                
                # 为factor类型的指标添加基准线
                if 'factor' in metric:
                    axes_enh[i, j].axvline(1.0, color='orange', linestyle=':', alpha=0.8,
                                          label='Baseline (1.0)')
                    axes_enh[i, j].legend()
        
        plt.tight_layout()
        plt.suptitle('Distribution of Enhancement Metrics (Activation & Gradient) by Layer', fontsize=16, y=1.02)
        # 保存增强指标图片
        plt.savefig(os.path.join(output_dir, 'enhancement_distribution_histograms.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 创建增强效果对比的箱线图
        fig_enh_box, axes_enh_box = plt.subplots(1, len(enhancement_metrics), figsize=(6*len(enhancement_metrics), 6))
        
        if len(enhancement_metrics) == 1:
            axes_enh_box = [axes_enh_box]
        
        for i, metric in enumerate(enhancement_metrics):
            data_to_plot = [df[df['layer_name'] == layer][metric] for layer in layer_names]
            
            # 修复matplotlib版本兼容性问题
            bp = axes_enh_box[i].boxplot(data_to_plot, tick_labels=layer_names, patch_artist=True)
            
            # 为每个箱子设置不同颜色
            for patch, color in zip(bp['boxes'], enhancement_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes_enh_box[i].set_title(f'{metric} - Layer Comparison')
            axes_enh_box[i].set_ylabel(metric)
            axes_enh_box[i].grid(True, alpha=0.3)
            axes_enh_box[i].tick_params(axis='x', rotation=45)
            
            # 为factor类型的指标添加基准线
            if 'factor' in metric:
                axes_enh_box[i].axhline(1.0, color='orange', linestyle=':', alpha=0.8,
                                       label='Baseline (1.0)')
                axes_enh_box[i].legend()
        
        plt.tight_layout()
        plt.suptitle('Boxplot Comparison of Enhancement Metrics', fontsize=16, y=1.02)
        # 保存增强指标箱线图
        plt.savefig(os.path.join(output_dir, 'enhancement_boxplot_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 增强效果统计分析
        print("\nEnhancement Metrics Statistical Summary:")
        for layer in layer_names:
            print(f"\n=== {layer} Enhancement Analysis ===")
            layer_data = df[df['layer_name'] == layer]
            
            # 基本统计
            for metric in enhancement_metrics:
                data = layer_data[metric]
                print(f"{metric}:")
                print(f"  Mean: {data.mean():.6f}")
                print(f"  Std: {data.std():.6f}")
                print(f"  Min: {data.min():.6f}")
                print(f"  Max: {data.max():.6f}")
                print(f"  Median: {data.median():.6f}")
            
            # 增强效果分析
            base_importance = layer_data['importance_value']
            enhanced_importance = layer_data['enhanced_importance']
            enhancement_ratio = enhanced_importance / base_importance
            
            print(f"\nEnhancement Effect Analysis:")
            print(f"  Enhancement Ratio (enhanced/base):")
            print(f"    Mean: {enhancement_ratio.mean():.4f}")
            print(f"    Min: {enhancement_ratio.min():.4f}")
            print(f"    Max: {enhancement_ratio.max():.4f}")
            
            # 计算被显著增强的神经元比例（增强超过10%）
            significantly_enhanced = (enhancement_ratio > 1.1).sum()
            total_neurons = len(enhancement_ratio)
            print(f"  Significantly Enhanced Neurons (>10%): {significantly_enhanced}/{total_neurons} ({significantly_enhanced/total_neurons*100:.1f}%)")
            
            # 计算激活和梯度因子的非平凡比例（不等于1.0）
            non_trivial_act = (layer_data['activation_factor'] != 1.0).sum()
            print(f"  Non-trivial Activation Factors: {non_trivial_act}/{total_neurons} ({non_trivial_act/total_neurons*100:.1f}%)")
    
    # 创建箱线图比较不同层之间的分布
    fig2, axes2 = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
    
    if len(metrics) == 1:
        axes2 = [axes2]
    
    for i, metric in enumerate(metrics):
        data_to_plot = [df[df['layer_name'] == layer][metric] for layer in layer_names]
        
        # 修复matplotlib版本兼容性问题
        bp = axes2[i].boxplot(data_to_plot, tick_labels=layer_names, patch_artist=True)
        
        # 为每个箱子设置不同颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes2[i].set_title(f'{metric} - Layer Comparison')
        axes2[i].set_ylabel(metric)
        axes2[i].grid(True, alpha=0.3)
        axes2[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.suptitle('Boxplot Comparison of Neuron Importance Metrics', fontsize=16, y=1.02)
    # 保存图片
    plt.savefig(os.path.join(output_dir, 'boxplot_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计摘要
    print("\nStatistical Summary:")
    for layer in layer_names:
        print(f"\n=== {layer} ===")
        layer_data = df[df['layer_name'] == layer]
        for metric in metrics:
            data = layer_data[metric]
            print(f"{metric}:")
            print(f"  Mean: {data.mean():.6f}")
            print(f"  Std: {data.std():.6f}")
            print(f"  Min: {data.min():.6f}")
            print(f"  Max: {data.max():.6f}")
            print(f"  Median: {data.median():.6f}")
    
    # 相关性分析
    print("\nCorrelation Analysis:")
    for layer in layer_names:
        print(f"\n=== {layer} Correlation Matrix ===")
        layer_data = df[df['layer_name'] == layer][metrics]
        correlation_matrix = layer_data.corr()
        print(correlation_matrix)
    
    # 创建相关性热力图
    if len(layer_names) > 1:
        fig3, axes3 = plt.subplots(1, len(layer_names), figsize=(6*len(layer_names), 5))
        
        if len(layer_names) == 1:
            axes3 = [axes3]
        
        for i, layer in enumerate(layer_names):
            layer_data = df[df['layer_name'] == layer][metrics]
            correlation_matrix = layer_data.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       ax=axes3[i], square=True, fmt='.3f')
            axes3[i].set_title(f'{layer} - Metrics Correlation')
        
        plt.tight_layout()
        plt.suptitle('Correlation Heatmaps by Layer', fontsize=16, y=1.02)
        # 保存图片
        plt.savefig(os.path.join(output_dir, 'correlation_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 保存数据摘要到文件
    summary_file = os.path.join(output_dir, 'neuron_importance_analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Neuron Importance Analysis Summary\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total data points: {len(df)}\n")
        f.write(f"Number of unique layers: {df['layer_name'].nunique()}\n")
        f.write("Data count per layer:\n")
        for layer, count in df['layer_name'].value_counts().items():
            f.write(f"  {layer}: {count}\n")
        
        f.write("\nStatistical Summary:\n")
        for layer in layer_names:
            f.write(f"\n{layer}:\n")
            layer_data = df[df['layer_name'] == layer]
            for metric in metrics:
                data = layer_data[metric]
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {data.mean():.6f}\n")
                f.write(f"    Std: {data.std():.6f}\n")
                f.write(f"    Min: {data.min():.6f}\n")
                f.write(f"    Max: {data.max():.6f}\n")
                f.write(f"    Median: {data.median():.6f}\n")
        
        f.write("\nCorrelation Analysis:\n")
        for layer in layer_names:
            f.write(f"\n{layer} Correlation Matrix:\n")
            layer_data = df[df['layer_name'] == layer][metrics]
            correlation_matrix = layer_data.corr()
            f.write(str(correlation_matrix))
            f.write("\n")
        
        # 添加增强指标摘要（如果存在的话）
        enhancement_metrics = ['importance_value', 'activation_factor', 'enhanced_importance']
        missing_cols = [col for col in enhancement_metrics if col not in df.columns]
        if not missing_cols:
            f.write("\n" + "="*50 + "\n")
            f.write("Enhancement Metrics Analysis\n")
            f.write("="*50 + "\n\n")
            
            for layer in layer_names:
                f.write(f"\n{layer} Enhancement Analysis:\n")
                layer_data = df[df['layer_name'] == layer]
                
                # 基本统计
                for metric in enhancement_metrics:
                    data = layer_data[metric]
                    f.write(f"  {metric}:\n")
                    f.write(f"    Mean: {data.mean():.6f}\n")
                    f.write(f"    Std: {data.std():.6f}\n")
                    f.write(f"    Min: {data.min():.6f}\n")
                    f.write(f"    Max: {data.max():.6f}\n")
                    f.write(f"    Median: {data.median():.6f}\n")
                
                # 增强效果分析
                base_importance = layer_data['importance_value']
                enhanced_importance = layer_data['enhanced_importance']
                enhancement_ratio = enhanced_importance / base_importance
                
                f.write(f"\n  Enhancement Effect Analysis:\n")
                f.write(f"    Enhancement Ratio (enhanced/base):\n")
                f.write(f"      Mean: {enhancement_ratio.mean():.4f}\n")
                f.write(f"      Min: {enhancement_ratio.min():.4f}\n")
                f.write(f"      Max: {enhancement_ratio.max():.4f}\n")
                
                # 计算被显著增强的神经元比例
                significantly_enhanced = (enhancement_ratio > 1.1).sum()
                total_neurons = len(enhancement_ratio)
                f.write(f"    Significantly Enhanced Neurons (>10%): {significantly_enhanced}/{total_neurons} ({significantly_enhanced/total_neurons*100:.1f}%)\n")
                
                # 计算激活和梯度因子的非平凡比例
                non_trivial_act = (layer_data['activation_factor'] != 1.0).sum()
                print(f"    Non-trivial Activation Factors: {non_trivial_act}/{total_neurons} ({non_trivial_act/total_neurons*100:.1f}%)\n")
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to directory: {os.path.abspath(output_dir)}/")
    print(f"Generated files:")
    print(f"  - distribution_histograms.png")
    print(f"  - boxplot_comparison.png") 
    print(f"  - correlation_heatmaps.png")
    print(f"  - enhancement_distribution_histograms.png")
    print(f"  - enhancement_boxplot_comparison.png")
    print(f"  - neuron_importance_analysis_summary.txt")

if __name__ == "__main__":
    read_and_analyze_importance_data() 