#!/usr/bin/env python3
"""生成中文阅读版，并保留冻结实验原件、数值、代码和链接目标。"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/zh-CN"
BASE = Path("experiments/memory_interchange")

# 只用于阅读文档；不修改任何实际数据字段或科学代码。
TERMS = {
    "Memory-state interchange experiment": "记忆状态交换实验",
    "Donor-native output consistency": "供体原生输出一致性复核",
    "Final audit": "最终复核",
    "leave-one-seed-out": "逐一种子留出",
    "PyTorch CUDA wheel": "PyTorch CUDA 安装包",
    "ridge 回归": "岭回归",
    "T CE nat": "T 交叉熵（nat）",
    "V CE nat": "V 交叉熵（nat）",
    "T n": "T 数量",
    "V n": "V 数量",
    "MAE nat": "平均绝对误差（nat）",
    "RMSE nat": "均方根误差（nat）",
    "CE nat": "交叉熵（nat）",
    "test z": "测试标准化分数",
    "Autograd mechanics": "自动微分机制",
    "Towards Best Practices of Activation Patching": "迈向激活替换的最佳实践",
    "Bucketing the Good Apples": "对优质样本进行分组",
    "pre-patch": "替换前",
    "donor-native": "供体原生",
    "cross-context": "跨上下文",
    "same-context": "同上下文",
    "native(pair)": "条件对的原生表现",
    "true_operation_sequence": "真实操作序列",
    "max_abs_probability_difference": "最大绝对概率差",
    "receiver_initial_distinct": "接收方与初始输入候选可区分",
    "receiver_donor_distinct": "接收方与供体候选可区分",
    "initial_donor_distinct": "初始输入与供体候选可区分",
    "all_three_distinct": "三个候选两两可区分",
    "first_only": "只执行第一步",
    "second_only": "只执行第二步",
    "query_type": "查询类型",
    "effective_suffix": "实际输入后缀",
    "memory_dim": "记忆维度",
    "non_N": "非 N 操作数",
    "item7": "项目7查询",
    "position7": "位置7查询",
    "bootstrap": "自助重采样",
    "pilot": "先导模型",
    "native": "原生",
    "patch": "记忆替换",
    "sham": "假替换对照",
    "matched": "内容匹配替换",
    "wrong": "错误内容替换",
    "translated": "线索翻译后",
    "original": "原始",
    "receiver": "接收方",
    "donor": "供体",
    "context": "上下文",
    "delay": "保持间隔",
    "query": "查询",
    "item": "项目",
    "position": "位置",
    "condition": "条件",
    "directions": "方向",
    "different": "不同",
    "same": "相同",
    "cross": "跨上下文",
    "all": "全体",
    "rec": "接收方原生正确子集",
    "both": "双方原生正确子集",
    "three": "三方原生正确子集",
    "mask": "样本掩码",
    "masks": "样本掩码",
    "correct": "原生正确",
    "error": "原生错误",
    "reversed": "反序执行",
    "none": "完全不更新",
    "KEEP": "保持不变",
    "True": "是",
    "False": "否",
    "seed": "种子",
    "best": "最佳",
    "latest": "最新",
    "advance": "晋级",
    "short": "短时序",
    "mid": "中等时序",
    "overall": "总体",
    "mode": "分析口径",
    "step": "步数",
    "update": "更新",
    "loss": "损失",
    "accuracy": "准确率",
    "test": "测试",
    "pooled": "合并统计",
    "fold": "留出折",
    "ridge": "岭回归",
    "oracle": "真值核验器",
    "collate": "批次整理",
    "sampler": "采样器",
    "padding": "填充",
    "scratch": "从头训练",
    "forward": "前向计算",
    "fused": "融合实现",
    "eager": "即时执行",
    "smoke test": "基础运行检查",
    "bug": "错误",
    "gate": "门值",
    "one-hot": "独热",
    "inverse": "取逆",
    "complement": "位置镜像",
    "control": "控制",
    "memory": "记忆",
    "raw": "原始",
    "max": "最大",
    "min": "最小",
    "median": "中位数",
    "mean": "均值",
    "实际 effective_suffix": "实际输入后缀",
    "lexicographic permutation ID": "按字典序排列的置换标识",
    "log-probabilities": "对数概率",
    "manifest": "清单",
    "dataset": "数据集",
    "treatment": "处理",
    "cell": "单元",
    "pair": "配对",
    "flag": "标志",
    "tie": "并列最大",
    "equal_cells": "等权单元",
    "Bernoulli": "伯努利",
    "bias": "偏置",
    "undefined": "未定义",
    "Frobenius": "弗罗贝尼乌斯",
    "smoke": "基础运行检查",
    "COMPLETED": "已完成（COMPLETED）",
    "PARTIAL": "部分完成（PARTIAL）",
    "BLOCKED": "受阻（BLOCKED）",
    "EXPLORATORY": "探索性（EXPLORATORY）",
    "TASK_READY": "任务就绪（TASK_READY）",
}

# 完整句子优先翻译，避免逐词翻译改变结果含义。
ENGLISH_RESULT = (
    "Across the six frozen RNNs and tested continuations, memory-state patches "
    "reproduced donor-native outputs (B1: 99.99988387% of query answers and "
    "99.99837414% of complete 14-query prediction vectors agreed, with mean total "
    "variation 7.8481e-07 and maximum probability difference 0.0067318; B_reserved: "
    "99.99991402% of query answers and 99.99879633% of complete 14-query prediction "
    "vectors agreed, with mean total variation 5.87055e-07 and maximum probability "
    "difference 0.00819505), including donor-native errors; these are descriptive "
    "comparisons on previously opened data."
)
CHINESE_RESULT = (
    "在六个冻结的循环神经网络及所测试的后续输入中，记忆状态替换后的输出复现了供体原生输出"
    "（B1：99.99988387% 的查询答案一致，99.99837414% 的完整 14 查询预测向量一致，"
    "平均总变差为 7.8481e-07，最大概率差为 0.0067318；B_reserved：99.99991402% 的查询答案一致，"
    "99.99879633% 的完整 14 查询预测向量一致，平均总变差为 5.87055e-07，最大概率差为 0.00819505），"
    "其中也包括供体原生输出的错误答案；这些结果是在此前已用于分析的数据上进行的描述性比较。"
)

TERM_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    + "|".join(re.escape(k) for k in sorted(TERMS, key=len, reverse=True))
    + r")(?![A-Za-z0-9_])"
)
# 行内代码、链接目标、网址和文件名均逐字保留。
PROTECTED_RE = re.compile(
    r"(`+)[^`\n]*?\1|\]\([^\n]*?\)|https?://[^\s<>]+|"
    r"\b(?:mean|max|min|sum|exp|log)(?=\s*[\[(])|"
    r"(?:[A-Za-z0-9_.*-]+/)*[A-Za-z0-9_.*-]+\."
    r"(?:csv|json|md|py|pt|npz|npy|yaml|log|txt|png|pdf|svg)(?![A-Za-z])"
)
LINK_RE = re.compile(r"\]\(([^\s()]+)\)")
FORMULA_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_]*(?:\([^=]*\))?\s*=")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", str(BASE)], cwd=ROOT
    ).decode().rstrip("\0").split("\0")
    return sorted(
        Path(p) for p in tracked
        if p.endswith(".md") or Path(p).name.startswith("protocol_")
        or Path(p).name == "english_result.txt"
    )


def translate_text(text, source):
    text = text.replace(ENGLISH_RESULT, CHINESE_RESULT)
    text = text.replace("可供摘要使用的英文结果句", "可供摘要使用的结果句（中文译文）")
    if "donor_consistency" in source.parts or "Final_audit" in source.parts:
        text = re.sub(r"(?<=\|) original (?=\|)", " 未翻译 ", text)
    def translate_piece(piece):
        piece = TERM_RE.sub(lambda m: TERMS[m[0]], piece)
        return re.sub(r"(?<=[\u4e00-\u9fff]) +(?=[\u4e00-\u9fff])", "", piece)

    pieces = []
    pos = 0
    for match in PROTECTED_RE.finditer(text):
        pieces.append(translate_piece(text[pos:match.start()]))
        pieces.append(match[0])
        pos = match.end()
    pieces.append(translate_piece(text[pos:]))
    return "".join(pieces)


def body(source):
    result = []
    fence = None
    headers = None
    for line in (ROOT / source).read_text().splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            if fence is None:
                fence = marker[1][0]
            elif fence == marker[1][0]:
                fence = None
            result.append(line)
        elif fence is not None or FORMULA_RE.match(line):
            result.append(line)
        else:
            if line.startswith("|"):
                cells = line.split("|")
                if headers is None:
                    headers = [cell.strip() for cell in cells]
                else:
                    for i, cell in enumerate(cells):
                        header = headers[i] if i < len(headers) else ""
                        if header in ("上下文", "同/跨"):
                            cells[i] = cell.replace(" same ", " 同上下文 ").replace(" cross ", " 跨上下文 ")
                        elif header == "历史":
                            cells[i] = cell.replace(" same ", " 同历史 ").replace(" different ", " 异历史 ")
                    line = "|".join(cells)
            else:
                headers = None
            result.append(translate_text(line, source))
    return "".join(result)


def rewrite_links(text, source, translated_sources):
    def replace(match):
        url = urlsplit(match[1])
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            return match[0]
        target = (ROOT / source.parent / unquote(url.path)).resolve()
        try:
            relative = target.relative_to(ROOT)
        except ValueError:
            return match[0]
        destination = OUT / relative if relative in translated_sources else target
        link = Path(os.path.relpath(destination, (OUT / source).parent)).as_posix()
        if url.query:
            link += "?" + url.query
        if url.fragment:
            link += "#" + url.fragment
        return "](" + link + ")"
    return LINK_RE.sub(replace, text)


def numeric_tokens(text):
    # 科学计数法整体比较；版本号及标识符内的数字也保留。
    text = LINK_RE.sub("]()", text)
    return Counter(re.findall(r"\d+(?:\.\d+)?(?:[eE][+−-]?\d+)?", text))


def code_blocks(text):
    return re.findall(r"^\s*(`{3,}|~{3,})[^\n]*\n([\s\S]*?)^\s*\1\s*$", text, re.M)


def inline_code(text):
    return Counter(re.findall(r"(`+)([^`\n]*?)\1", text))


def generate(check):
    paths = sources()
    translated_sources = set(paths)
    manifest = {"说明": "中文阅读版；原件、数值和科学分析保持不变。", "文档": []}
    failures = []
    for source in paths:
        original = (ROOT / source).read_bytes()
        translated = body(source)
        source_text = original.decode()
        if numeric_tokens(source_text) != numeric_tokens(translated):
            raise ValueError(f"数值或含数字标识符被改动：{source}")
        if code_blocks(source_text) != code_blocks(translated):
            raise ValueError(f"代码块被改动：{source}")
        if inline_code(source_text) != inline_code(translated):
            raise ValueError(f"行内代码被改动：{source}")
        if [s for s in source_text.splitlines() if FORMULA_RE.match(s)] != [
            s for s in translated.splitlines() if FORMULA_RE.match(s)
        ]:
            raise ValueError(f"公式被改动：{source}")
        translated = rewrite_links(translated, source, translated_sources)
        target = OUT / source
        original_link = Path(os.path.relpath(ROOT / source, target.parent)).as_posix()
        index_link = Path(os.path.relpath(OUT / "README.md", target.parent)).as_posix()
        note = (
            f"> 中文阅读版 · [冻结原件]({original_link}) · [中文导航]({index_link})\n>\n"
            "> 原版本的时间、状态和结论按原记录保留；代码、字段标识、公式和原图沿用原件。\n\n"
        )
        rendered = (note + translated.rstrip("\n") + "\n").encode()
        if check:
            if not target.exists() or target.read_bytes() != rendered:
                failures.append(f"中文版本需更新：{source}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(rendered)
        manifest["文档"].append({
            "原件": source.as_posix(), "原件_SHA256": digest(original),
            "中文版本": target.relative_to(ROOT).as_posix(), "中文版本_SHA256": digest(rendered),
        })
    manifest["生成器_SHA256"] = digest(Path(__file__).read_bytes())
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    manifest_path = OUT / "translation_manifest.json"
    if check:
        if not manifest_path.exists() or manifest_path.read_bytes() != manifest_bytes:
            failures.append("中文文档来源清单需更新")
        for path in [ROOT / "README.md", *OUT.rglob("*.md")]:
            for match in LINK_RE.finditer(path.read_text()):
                url = urlsplit(match[1])
                if url.scheme or url.netloc or not url.path:
                    continue
                if not (path.parent / unquote(url.path)).exists():
                    failures.append(f"链接不存在：{path.relative_to(ROOT)} -> {url.path}")
        if failures:
            raise SystemExit("\n".join(failures))
        print(f"校验通过：{len(paths)} 篇中文文档；原文数值、代码、程序标识与本地链接一致。")
    else:
        manifest_path.write_bytes(manifest_bytes)
        print(f"已生成 {len(paths)} 篇中文文档，原件未修改。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验，不写入文件")
    generate(parser.parse_args().check)
