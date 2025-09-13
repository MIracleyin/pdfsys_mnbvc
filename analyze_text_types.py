#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析PDF每页中不同种类文字的个数
支持中文、英文、数字、标点符号、特殊字符等分类统计
"""

import json
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

class TextTypeAnalyzer:
    """文本类型分析器"""
    
    def __init__(self):
        # 定义各种文字类型的正则表达式
        self.patterns = {
            'chinese': re.compile(r'[\u4e00-\u9fff]'),           # 中文字符
            'english_letters': re.compile(r'[a-zA-Z]'),          # 英文字母
            'digits': re.compile(r'[0-9]'),                      # 数字
            'punctuation': re.compile(r'[^\w\s]'),               # 标点符号
            'whitespace': re.compile(r'\s'),                     # 空白字符
            'special_chars': re.compile(r'[^\u4e00-\u9fff\w\s]') # 特殊字符
        }
    
    def categorize_character(self, char: str) -> str:
        """对单个字符进行分类"""
        if self.patterns['chinese'].match(char):
            return 'chinese'
        elif self.patterns['english_letters'].match(char):
            return 'english_letters'
        elif self.patterns['digits'].match(char):
            return 'digits'
        elif self.patterns['whitespace'].match(char):
            return 'whitespace'
        elif self.patterns['punctuation'].match(char):
            return 'punctuation'
        else:
            return 'special_chars'
    
    def analyze_text(self, text: str) -> Dict[str, int]:
        """分析文本中各种类型字符的数量"""
        counts = defaultdict(int)
        
        for char in text:
            char_type = self.categorize_character(char)
            counts[char_type] += 1
        
        return dict(counts)
    
    def count_unique_chinese_chars(self, text: str) -> int:
        """统计文本中不同中文字符的种类数量"""
        chinese_chars = set()
        
        for char in text:
            if self.patterns['chinese'].match(char):
                chinese_chars.add(char)
        
        return len(chinese_chars)
    
    def count_unique_english_words(self, text: str) -> int:
        """统计文本中不同英文单词的种类数量"""
        # 提取英文单词（包含字母的连续字符串）
        english_words = set()
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        for word in words:
            if len(word) >= 2:  # 只统计长度>=2的单词
                english_words.add(word)
        
        return len(english_words)
    
    def detect_document_type(self, text: str) -> str:
        """检测文档类型：中文文献或英文文献"""
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_count = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_count > english_count:
            return 'chinese'
        elif english_count > chinese_count:
            return 'english'
        else:
            return 'mixed'
    
    def analyze_page(self, page_text: str) -> Dict[str, int]:
        """分析单页文本"""
        result = self.analyze_text(page_text)
        # 添加中文字符种类数量
        result['unique_chinese_chars'] = self.count_unique_chinese_chars(page_text)
        # 添加英文单词种类数量
        result['unique_english_words'] = self.count_unique_english_words(page_text)
        # 检测文档类型
        result['doc_type'] = self.detect_document_type(page_text)
        return result
    
    def analyze_pdf(self, pdf_data: Dict) -> Tuple[str, List[Dict[str, int]]]:
        """分析整个PDF的每页文字类型"""
        file_path = pdf_data.get('file_path', 'Unknown')
        text_pages = pdf_data.get('text', [])
        
        page_analyses = []
        all_text = ""
        
        for page_num, page_text in enumerate(text_pages, 1):
            if isinstance(page_text, str):
                page_stats = self.analyze_page(page_text)
                page_stats['page_number'] = page_num
                page_analyses.append(page_stats)
                all_text += page_text + " "
        
        # 检测整个PDF的文档类型
        if page_analyses:
            overall_doc_type = self.detect_document_type(all_text)
            # 为所有页面添加整体文档类型信息
            for page in page_analyses:
                page['overall_doc_type'] = overall_doc_type
        
        return file_path, page_analyses

def process_single_line(line_data):
    """处理单行JSON数据的函数"""
    line_num, line = line_data
    analyzer = TextTypeAnalyzer()
    
    try:
        pdf_data = json.loads(line.strip())
        file_path_info, page_analyses = analyzer.analyze_pdf(pdf_data)
        
        result = {
            'file_path': file_path_info,
            'total_pages': len(page_analyses),
            'pages': page_analyses,
            'summary': calculate_pdf_summary(page_analyses)
        }
        
        return line_num, result, None
        
    except json.JSONDecodeError as e:
        return line_num, None, f"JSON解析错误: {e}"
    except Exception as e:
        return line_num, None, f"处理错误: {e}"

def analyze_jsonl_file(file_path: str, output_file: str = None, max_records: int = None, num_threads: int = 4):
    """分析JSONL文件中的所有PDF - 多线程版本"""
    results = []
    errors = []
    
    print(f"开始分析文件: {file_path} (使用 {num_threads} 个线程)")
    
    # 首先读取所有行
    lines_to_process = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_records and line_num > max_records:
                break
            lines_to_process.append((line_num, line))
    
    total_lines = len(lines_to_process)
    print(f"总共需要处理 {total_lines} 行")
    
    # 使用线程池处理
    processed_count = 0
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_line = {executor.submit(process_single_line, line_data): line_data[0] 
                         for line_data in lines_to_process}
        
        # 处理完成的任务
        for future in as_completed(future_to_line):
            line_num, result, error = future.result()
            
            with lock:
                processed_count += 1
                
                if error:
                    errors.append(f"第 {line_num} 行{error}")
                    print(f"第 {line_num} 行{error}")
                else:
                    results.append(result)
                
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count}/{total_lines} 个PDF文件 ({processed_count/total_lines*100:.1f}%)")
    
    print(f"总共处理了 {len(results)} 个PDF文件, 出现 {len(errors)} 个错误")
    
    if errors:
        print(f"错误详情:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"  {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors)-10} 个错误")
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_file}")
    
    return results

def calculate_pdf_summary(page_analyses: List[Dict[str, int]]) -> Dict[str, int]:
    """计算PDF的总体统计信息"""
    summary = defaultdict(int)
    
    for page in page_analyses:
        for key, value in page.items():
            # 只累加数字类型的值，跳过字符串类型的字段
            if key not in ['page_number', 'doc_type', 'overall_doc_type'] and isinstance(value, (int, float)):
                summary[key] += value
    
    return dict(summary)

def analyze_text_diversity(results: List[Dict], threshold: int = 50):
    """分析文本多样性，统一处理中文字符和英文单词，找出超过指定阈值的页面"""
    high_diversity_pages = []
    total_pages = 0
    chinese_docs = 0
    english_docs = 0
    mixed_docs = 0
    
    for result in results:
        file_path = result['file_path']
        pages = result['pages']
        
        # 统计文档类型
        if pages:
            doc_type = pages[0].get('overall_doc_type', 'unknown')
            if doc_type == 'chinese':
                chinese_docs += 1
            elif doc_type == 'english':
                english_docs += 1
            else:
                mixed_docs += 1
        
        for page in pages:
            total_pages += 1
            unique_chinese = page.get('unique_chinese_chars', 0)
            unique_english = page.get('unique_english_words', 0)
            doc_type = page.get('overall_doc_type', 'unknown')
            
            # 根据文档类型选择合适的指标
            diversity_count = 0
            diversity_type = ""
            
            if doc_type == 'chinese':
                diversity_count = unique_chinese
                diversity_type = "中文字符"
            elif doc_type == 'english':
                diversity_count = unique_english
                diversity_type = "英文单词"
            else:  # mixed
                diversity_count = max(unique_chinese, unique_english)
                diversity_type = "中文字符" if unique_chinese >= unique_english else "英文单词"
            
            if diversity_count > threshold:
                high_diversity_pages.append({
                    'file_path': file_path,
                    'page_number': page['page_number'],
                    'diversity_count': diversity_count,
                    'diversity_type': diversity_type,
                    'doc_type': doc_type,
                    'unique_chinese_chars': unique_chinese,
                    'unique_english_words': unique_english
                })
    
    # 计算比例
    ratio = len(high_diversity_pages) / total_pages if total_pages > 0 else 0
    
    print(f"\n=== 文本多样性分析 (阈值: {threshold}) ===")
    print(f"总页数: {total_pages}")
    print(f"文档类型分布: 中文文献 {chinese_docs}个, 英文文献 {english_docs}个, 混合文献 {mixed_docs}个")
    print(f"超过{threshold}种类的页面数: {len(high_diversity_pages)}")
    print(f"占比: {ratio:.4f} ({ratio*100:.2f}%)")
    
    if high_diversity_pages:
        print(f"\n=== 超过{threshold}种类的页面列表 ===")
        for i, page_info in enumerate(high_diversity_pages, 1):
            doc_type_label = {
                'chinese': '[中文文献]',
                'english': '[英文文献]',
                'mixed': '[混合文献]'
            }.get(page_info['doc_type'], '[未知类型]')
            
            print(f"{i}. {doc_type_label} {page_info['file_path']} - 第{page_info['page_number']}页:")
            print(f"   {page_info['diversity_type']}: {page_info['diversity_count']}种")
            print(f"   (中文字符: {page_info['unique_chinese_chars']}种, 英文单词: {page_info['unique_english_words']}种)")
    
    return {
        'total_pages': total_pages,
        'high_diversity_pages': high_diversity_pages,
        'ratio': ratio,
        'threshold': threshold,
        'doc_type_stats': {
            'chinese': chinese_docs,
            'english': english_docs,
            'mixed': mixed_docs
        }
    }

def print_statistics(results: List[Dict], top_n: int = 5):
    """打印统计信息"""
    if not results:
        print("没有分析结果")
        return
    
    print("\n=== 分析统计 ===")
    print(f"总PDF数量: {len(results)}")
    
    # 计算总页数
    total_pages = sum(r['total_pages'] for r in results)
    print(f"总页数: {total_pages}")
    
    # 找出页数最多的PDF
    max_pages_pdf = max(results, key=lambda x: x['total_pages'])
    print(f"最多页数: {max_pages_pdf['total_pages']} 页 ({max_pages_pdf['file_path']})")
    
    # 统计各类字符的总数
    total_chars = defaultdict(int)
    for result in results:
        summary = result['summary']
        for char_type, count in summary.items():
            total_chars[char_type] += count
    
    print("\n=== 字符类型统计 ===")
    for char_type, count in sorted(total_chars.items(), key=lambda x: x[1], reverse=True):
        print(f"{char_type}: {count:,}")
    
    # 显示前N个最大的PDF
    print(f"\n=== 前{top_n}个最大PDF (按总字符数) ===")
    sorted_results = sorted(results, 
                          key=lambda x: sum(x['summary'].values()), 
                          reverse=True)
    
    for i, result in enumerate(sorted_results[:top_n], 1):
        total_chars = sum(result['summary'].values())
        print(f"{i}. {result['file_path']}")
        print(f"   页数: {result['total_pages']}, 总字符数: {total_chars:,}")

def analyze_high_diversity_ratio(results: List[Dict], chinese_threshold: int = 120, english_threshold: int = 100):
    """分析高多样性页面的占比统计"""
    total_pages = 0
    chinese_pages = 0  # 中文文档页数
    english_pages = 0  # 英文文档页数
    mixed_pages = 0    # 混合文档页数
    
    high_chinese_pages = 0
    high_english_pages = 0
    high_chinese_files = set()
    high_english_files = set()
    
    print(f"\n=== 高多样性页面占比分析 ===")
    print(f"中文字符阈值: {chinese_threshold}, 英文单词阈值: {english_threshold}")
    
    for result in results:
        file_path = result['file_path']
        pages = result['pages']
        
        for page in pages:
            total_pages += 1
            doc_type = page.get('overall_doc_type', 'unknown')
            unique_chinese = page.get('unique_chinese_chars', 0)
            unique_english = page.get('unique_english_words', 0)
            
            # 根据文档类型分类统计页数
            if doc_type == 'chinese':
                chinese_pages += 1
                if unique_chinese > chinese_threshold:
                    high_chinese_pages += 1
                    high_chinese_files.add(file_path)
            elif doc_type == 'english':
                english_pages += 1
                if unique_english > english_threshold:
                    high_english_pages += 1
                    high_english_files.add(file_path)
            else:  # mixed or unknown
                mixed_pages += 1
    
    # 计算比例
    chinese_ratio = high_chinese_pages / chinese_pages if chinese_pages > 0 else 0
    english_ratio = high_english_pages / english_pages if english_pages > 0 else 0
    
    print(f"总页数: {total_pages:,}")
    print(f"  中文文档页数: {chinese_pages:,}")
    print(f"  英文文档页数: {english_pages:,}")
    print(f"  混合/未知文档页数: {mixed_pages:,}")
    
    print(f"\n中文字符数量超过 {chinese_threshold} 的页面:")
    print(f"  页面数: {high_chinese_pages:,}")
    print(f"  占中文文档总页数比例: {chinese_ratio:.4f} ({chinese_ratio*100:.2f}%)")
    print(f"  涉及PDF文件数: {len(high_chinese_files)}")
    
    print(f"\n英文单词数量超过 {english_threshold} 的页面:")
    print(f"  页面数: {high_english_pages:,}")
    print(f"  占英文文档总页数比例: {english_ratio:.4f} ({english_ratio*100:.2f}%)")
    print(f"  涉及PDF文件数: {len(high_english_files)}")
    
    return {
        'total_pages': total_pages,
        'chinese_pages': chinese_pages,
        'english_pages': english_pages,
        'mixed_pages': mixed_pages,
        'high_chinese_pages': high_chinese_pages,
        'high_english_pages': high_english_pages,
        'chinese_ratio': chinese_ratio,
        'english_ratio': english_ratio,
        'chinese_threshold': chinese_threshold,
        'english_threshold': english_threshold,
        'high_chinese_files': len(high_chinese_files),
        'high_english_files': len(high_english_files)
    }

def analyze_saved_results(json_file: str, chinese_threshold: int = 120, english_threshold: int = 100):
    """分析已保存的结果文件"""
    print(f"正在读取分析结果文件: {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"成功读取 {len(results)} 个PDF文件的分析结果")
        
        # 执行高多样性占比分析
        ratio_stats = analyze_high_diversity_ratio(results, chinese_threshold, english_threshold)
        
        return ratio_stats
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式错误 {e}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None

def show_sample_page_analysis(results: List[Dict], pdf_index: int = 0, page_index: int = 0):
    """显示样本页面分析结果"""
    if not results or pdf_index >= len(results):
        print("没有可显示的结果")
        return
    
    result = results[pdf_index]
    if page_index >= len(result['pages']):
        print(f"PDF第{page_index+1}页不存在")
        return
    
    page = result['pages'][page_index]
    doc_type = page.get('overall_doc_type', 'unknown')
    doc_type_label = {
        'chinese': '[中文文献]',
        'english': '[英文文献]',
        'mixed': '[混合文献]'
    }.get(doc_type, '[未知类型]')
    
    print(f"\n=== 样本分析: {result['file_path']} {doc_type_label} ===")
    print(f"第 {page['page_number']} 页字符统计:")
    
    # 首先显示多样性统计
    unique_chinese = page.get('unique_chinese_chars', 0)
    unique_english = page.get('unique_english_words', 0)
    print(f"  unique_chinese_chars: {unique_chinese}")
    print(f"  unique_english_words: {unique_english}")
    
    # 然后显示其他统计
    for char_type, count in sorted(page.items()):
        if char_type not in ['page_number', 'unique_chinese_chars', 'unique_english_words', 'doc_type', 'overall_doc_type']:
            print(f"  {char_type}: {count}")

if __name__ == "__main__":
    import sys
    
    # 分析文件
    input_file = "nas1/extracted_content_001.jsonl"
    output_file = "text_analysis_results.json"
    
    # 检查是否要分析已保存的结果
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        print("分析已保存的结果文件...")
        analyze_saved_results(output_file, chinese_threshold=120, english_threshold=100)
        sys.exit(0)
    
    # 可以通过命令行参数指定线程数，默认为4
    num_threads = 4
    if len(sys.argv) > 1:
        try:
            num_threads = int(sys.argv[1])
            print(f"使用 {num_threads} 个线程")
        except ValueError:
            print("线程数参数无效，使用默认值4")
    
    # 分析全部PDF
    print("开始多线程分析全部PDF...")
    results = analyze_jsonl_file(input_file, output_file, max_records=None, num_threads=num_threads)
    
    # 显示统计信息
    print_statistics(results)
    
    # 分析文本多样性（统一处理中文字符和英文单词）
    if results:
        analyze_text_diversity(results, threshold=50)
    
    # 分析高多样性页面占比
    if results:
        analyze_high_diversity_ratio(results, chinese_threshold=120, english_threshold=100)
    
    # 显示样本分析
    if results:
        show_sample_page_analysis(results, 0, 0)
    
    print(f"\n完成! 详细结果保存在: {output_file}")
    print(f"\n提示: 可以使用 'python analyze_text_types.py analyze' 来分析已保存的结果文件")