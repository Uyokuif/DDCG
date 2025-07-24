#!/usr/bin/env python3
"""
D2C 智能体评估脚本
用于运行带有 D2C (Discriminator-to-Critic) 评分系统的智能体评估
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 添加必要的路径
sys.path.append(os.path.dirname(__file__))

def setup_logging(log_dir="logs"):
    """设置日志记录"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"d2c_evaluation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='D2C 智能体评估')
    
    # 基础参数
    parser.add_argument('--test_examples', type=int, default=10, help='测试样例数量')
    parser.add_argument('--subset', type=str, default='NovelTasks', 
                       choices=['NovelTasks', 'InDistributation', 'OutDistributation'],
                       help='评测数据集子集')
    
    # D2C 参数
    parser.add_argument('--fusion_strategy', type=str, default='adaptive',
                       choices=['adaptive', 'conservative', 'average'],
                       help='D2C 融合策略')
    parser.add_argument('--dgap_debug', action='store_true', default=False,
                       help='是否显示 D2C 评分详情')
    
    # 环境参数
    parser.add_argument('--base_port', type=int, default=8679, help='Unity 环境端口')
    parser.add_argument('--graphics', action='store_true', default=False, help='是否显示图形界面')
    parser.add_argument('--max_episode_length', type=int, default=100, help='最大步数')
    
    # GPT 参数
    parser.add_argument('--use_gpt', action='store_true', default=True, help='是否使用 GPT 规划')
    parser.add_argument('--use_script', action='store_true', default=False, help='是否使用脚本模式')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    logger.info("🚀 启动 D2C 智能体评估系统")
    logger.info(f"📋 评估配置:")
    logger.info(f"   测试样例: {args.test_examples}")
    logger.info(f"   数据子集: {args.subset}")
    logger.info(f"   融合策略: {args.fusion_strategy}")
    logger.info(f"   调试模式: {args.dgap_debug}")
    
    try:
        # 首先测试模型加载
        logger.info("🧪 测试 D2C 模型加载...")
        from test_model_loading import test_directory_structure, test_model_loading
        
        if not test_directory_structure():
            logger.error("❌ 模型目录结构检查失败")
            return 1
            
        if not test_model_loading():
            logger.error("❌ 模型加载测试失败")
            return 1
            
        logger.info("✅ D2C 模型系统检查通过")
        
        # 设置 D2C 全局参数
        import interactive_interface
        interactive_interface.DUAL_DGAP_FUSION_STRATEGY = args.fusion_strategy
        interactive_interface.DUAL_DGAP_DEBUG = args.dgap_debug
        
        # 导入必要模块
        from arguments import get_args
        from utils_bc.utils_interactive_eval import connect_env
        
        # 更新参数 - 暂时修改sys.argv来避免参数冲突
        import sys
        original_argv = sys.argv.copy()
        
        # 移除我们自定义的参数，只保留arguments.py认识的参数
        filtered_argv = ['run_d2c_evaluation.py']
        if '--test_examples' in original_argv:
            idx = original_argv.index('--test_examples')
            filtered_argv.extend(['--test_examples', original_argv[idx + 1]])
        if '--subset' in original_argv:
            idx = original_argv.index('--subset')
            # subset参数需要传给env_task_set
            filtered_argv.extend(['--subset', original_argv[idx + 1]])
            filtered_argv.extend(['--env_task_set', original_argv[idx + 1]])
        else:
            # 默认值
            filtered_argv.extend(['--env_task_set', args.subset])
        if '--base_port' in original_argv:
            idx = original_argv.index('--base_port')
            filtered_argv.extend(['--base-port', original_argv[idx + 1]])  # 注意这里是base-port
        if '--graphics' in original_argv:
            filtered_argv.append('--graphics')
        if '--max_episode_length' in original_argv:
            idx = original_argv.index('--max_episode_length')
            filtered_argv.extend(['--max_episode_length', original_argv[idx + 1]])
            
        sys.argv = filtered_argv
        eval_args = get_args()
        sys.argv = original_argv  # 恢复原始参数
        
        # 连接环境
        logger.info("🔗 连接 VirtualHome 环境...")
        vh_envs = connect_env(eval_args, logger)
        
        # 运行评估
        logger.info("🏃 开始智能体评估...")
        from interactive_interface import interactive_interface_fn
        
        success_rate = interactive_interface_fn(
            args=eval_args,
            vh_envs=vh_envs,
            iteri=0,
            agent_model=None,  # 使用 GPT 策略
            data_info=None,
            logging=logger,
            tokenizer=None
        )
        
        logger.info(f"🎯 评估完成! 成功率: {success_rate:.1f}%")
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断评估")
        return 130
    except Exception as e:
        logger.error(f"❌ 评估过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 