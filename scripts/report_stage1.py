"""Generate a readable result report from completed frozen test artifacts."""
import json
from echolens.experiments import ROOT


def main():
    reports = ROOT / "docs/reports"
    lines = ["# 阶段 1：数据与基线结果", "", "指标基于 ASAP 官方标签；单次固定种子实验，非新领域泛化或人工新增金标准成绩。", "",
             "## 同一去污染测试集上的比较", "", "两种训练视图均评测同一份 4,937 条去污染测试集。官方训练视图含已知相似测试文本，仅用于污染对照。", "",
             "| 训练视图 | 模型 | 方面 Macro-F1 | 方面 Micro-F1 | 配对 Macro-F1 | 配对 Micro-F1 | 真实提及情感 Macro-F1 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    all_reports = {}
    for version in ("official", "clean"):
        for kind in ("tfidf", "encoder"):
            run = f"{kind}-{version}"
            report = json.loads((reports / f"stage1-{run}-test.json").read_text(encoding="utf-8"))
            all_reports[run] = report
            m = report["results"]["clean"]["metrics"]
            values = [m["aspects"]["macro_f1"], m["aspects"]["micro_f1"], m["pairs"]["macro_f1"], m["pairs"]["micro_f1"], m["sentiment_on_gold_mentions"]["macro_f1"]]
            lines.append(f"| {version} | {kind} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## 官方原始测试视图", "", "保留 4,940 条原始测试样本，以下为官方训练视图的结果；存在已知污染，不能作为无泄漏成绩。", "",
              "| 模型 | 方面 Macro-F1 | 配对 Micro-F1 |", "| --- | ---: | ---: |"]
    for kind in ("tfidf", "encoder"):
        m = all_reports[f"{kind}-official"]["results"]["official"]["metrics"]
        lines.append(f"| {kind} | {m['aspects']['macro_f1']:.4f} | {m['pairs']['micro_f1']:.4f} |")
    lines += ["", "## 去污染训练与测试：逐方面识别", "", "P/R/F1 均按是否提及该方面计算，support 是真实提及评论数。", "",
              "| 方面 | support | TF-IDF P/R/F1 | 编码器 P/R/F1 |", "| --- | ---: | --- | --- |"]
    linear = all_reports["tfidf-clean"]["results"]["clean"]["metrics"]["aspects"]["per_label"]
    encoder = all_reports["encoder-clean"]["results"]["clean"]["metrics"]["aspects"]["per_label"]
    for aspect, l in linear.items():
        e = encoder[aspect]
        render = lambda x: "/".join(f"{x[k]:.4f}" for k in ("precision", "recall", "f1"))
        lines.append(f"| {aspect} | {l['support']} | {render(l)} | {render(e)} |")
    lines += ["", "## 解释与限制", "",
              "- 方面是 18 个二元提及标签；配对是 54 个方面×情感标签。Macro-F1 对所有固定标签等权，缺少预测的长尾类计零。",
              "- 真实提及情感仅用真实标签非 -2 的位置，但预测未提及仍算错。逐情感、逐配对 P/R/F1/support 见相应 JSON。",
              "- 官方与去污染训练结果的差异同时受删样本、开发集选择和训练批次顺序影响，不是重复污染的严格因果估计。",
              "- 编码器截断至 384 tokens；截断数、批次时延、训练时长和显存见 dev/test JSON。TF-IDF 使用全长文本，二者计算路径不同。",
              "- 近重复仅覆盖固定字符规则发现的候选；不能断言没有语义、用户或预训练泄漏。",
              "- 模型文件及逐条预测保留本地 artifacts/stage-01；公开报告不含真实评论原文。",
              "- 测试前的配置和模型哈希见 stage1-test-freeze.json。本阶段不根据测试结果追加调参。", ""]
    (reports / "stage1-baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print("Created docs/reports/stage1-baselines.md")


if __name__ == "__main__":
    main()
