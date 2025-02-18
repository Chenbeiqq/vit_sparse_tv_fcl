import os
import pandas as pd
from datetime import datetime

def test_save_stats_to_excel():
    # 创建一个类的实例
    class MockClass:
        def save_stats_to_excel(self, stats_dict, excel_path='sparse_merge_stats.xlsx'):
            """
            将合并统计信息保存到Excel文件，支持追加新数据

            Args:
                stats_dict: 统计信息字典
                excel_path: Excel文件路径
            """
            # 准备当前统计数据
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rows = []

            for layer_name, stats in stats_dict.items():
                row = {
                    '时间': current_time,
                    '层名称': layer_name,
                    '总索引数': stats['total_indices'],
                    '唯一索引数': stats['unique_indices'],
                    '重复索引数': stats['duplicate_indices'],
                    '重复率': f"{stats['duplicate_ratio']:.2%}",
                    '最大重复次数': stats['max_duplicates'],
                    '最大重复索引数': stats['indices_with_max_duplicates']
                }
                rows.append(row)

            # 创建新的DataFrame
            new_df = pd.DataFrame(rows)

            try:
                # 如果文件存在，读取并追加
                if os.path.exists(excel_path):
                    existing_df = pd.read_excel(excel_path)
                    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    updated_df = new_df

                # 保存到Excel
                updated_df.to_excel(excel_path, index=False)
                print(f"统计数据已保存到 {excel_path}")

            except Exception as e:
                print(f"保存Excel时发生错误: {str(e)}")
                # 尝试使用备份文件名保存
                backup_path = f'sparse_merge_stats_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                new_df.to_excel(backup_path, index=False)
                print(f"已保存备份文件到 {backup_path}")
            # 方法代码粘贴在这里（省略重复）

    mock_instance = MockClass()

    # 测试数据
    test_stats = {
        'layer_1': {
            'total_indices': 100,
            'unique_indices': 90,
            'duplicate_indices': 10,
            'duplicate_ratio': 0.1,
            'max_duplicates': 5,
            'indices_with_max_duplicates': [1, 2, 3]
        },
        'layer_2': {
            'total_indices': 200,
            'unique_indices': 180,
            'duplicate_indices': 20,
            'duplicate_ratio': 0.1,
            'max_duplicates': 4,
            'indices_with_max_duplicates': [4, 5]
        }
    }

    # 测试文件路径
    test_excel_path = r'C:\vit_sparse_tv\wandb\test_sparse_merge_stats.xlsx'

    # 确保没有旧文件
    if os.path.exists(test_excel_path):
        os.remove(test_excel_path)

    # 调用方法
    mock_instance.save_stats_to_excel(test_stats, excel_path=test_excel_path)

    # 验证 Excel 文件是否创建
    assert os.path.exists(test_excel_path), "Excel 文件未创建"

    # 验证内容
    df = pd.read_excel(test_excel_path)
    assert len(df) == 2, "数据行数不正确"
    assert '时间' in df.columns, "缺少时间列"
    assert '层名称' in df.columns, "缺少层名称列"

    # 打印输出，便于调试
    print("测试数据保存成功，内容如下：")
    print(df)

    # 清理测试文件

# 调用测试函数
test_save_stats_to_excel()
