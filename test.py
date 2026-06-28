import os
import math
import sacrebleu
import torch
import transformer
import translate

import re

def detokenize(text):
    """简单还原空格：去除标点符号前面的空格"""
    # 匹配 [.,!?;:] 等标点前面的空格并去掉
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # 匹配左括号、引号等后面的空格并去掉
    text = re.sub(r'([\(\[\'\"\"\u201c])\s+', r'\1', text)
    return text.strip()


def test(model, test_en, test_de, device, src_vocab, tgt_vocab,
         beam_size=5, tokenizer_fn=None):
    """计算 BLEU 分数（使用 Beam Search 解码）"""
    src_sentences = test_en
    ref_sentences = test_de
    predictions = []
    total = len(src_sentences)
    
    for i, src_sentence in enumerate(src_sentences):
        pred = translate.translate(model, src_sentence, device, src_vocab, tgt_vocab,
                                   beam_size=beam_size, tokenizer_fn=tokenizer_fn)
        predictions.append(detokenize(pred))
        
        # 进度显示
        if (i + 1) % 100 == 0:
            print(f"  Translated {i+1}/{total} sentences...")
    
    bleu = sacrebleu.corpus_bleu(predictions, [ref_sentences], lowercase=True)
    print(f"BLEU score: {bleu.score:.4f}")
    print(f"  BLEU-1: {bleu.counts[0] / (bleu.totals[0] + 1e-9) * 100:.2f}")
    print(f"  BLEU-4 precision: {bleu.precisions[3]:.4f}")
    print(f"  Brevity penalty: {bleu.bp:.4f}")
    return bleu.score


def load_checkpoint(MODEL_PATH, device):
    """从 checkpoint 加载模型和词汇表，返回 (model, src_vocab, tgt_vocab)"""
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise RuntimeError(
            "Checkpoint format not recognized. "
            "Please re-train the model with: python transformer.py"
        )
    
    # 提取词汇表
    src_vocab = checkpoint.get('src_vocab')
    tgt_vocab = checkpoint.get('tgt_vocab')
    if src_vocab is None or tgt_vocab is None:
        # 旧格式兼容
        src_vocab_size = checkpoint.get('src_vocab_size', 10000)
        tgt_vocab_size = checkpoint.get('tgt_vocab_size', 10000)
        # 无法找回完整 vocab，报错
        raise RuntimeError(
            "Old checkpoint format (missing vocab). "
            "Please re-train the model with: python transformer.py"
        )
    
    # 构建模型
    model = transformer.Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128, nhead=8,
        num_layers=4, dim_ff=512,
        dropout=0.2
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, src_vocab, tgt_vocab


if __name__ == "__main__":
    # 确定脚本所在目录，确保模型路径一致
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "best_transformer.pt")

    BPE_SRC_PATH = os.path.join(SCRIPT_DIR, "bpe_src.json")
    BPE_TGT_PATH = os.path.join(SCRIPT_DIR, "bpe_tgt.json")

    # 设备选择
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = transformer.download_multi30k(data_dir=r"D:\code\python\udlbook-notebooks\Notebooks\myself\data")
    
    # 从 checkpoint 加载模型和词汇表
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found!")
        print("Please train the model first: python transformer.py")
        exit(1)
    
    model, src_vocab, tgt_vocab = load_checkpoint(MODEL_PATH, device)
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}")
    
    # 尝试加载 BPE 分词器
    use_bpe = os.path.exists(BPE_SRC_PATH) and os.path.exists(BPE_TGT_PATH)
    print("Using BPE tokenizer...")
    src_bpe = transformer.BPETokenizer()
    src_bpe.load(BPE_SRC_PATH)
    tgt_bpe = transformer.BPETokenizer()
    tgt_bpe.load(BPE_TGT_PATH)
    src_tokenize_fn = src_bpe.tokenize
    tgt_tokenize_fn = tgt_bpe.tokenize

    # 测试 Beam Search 不同 beam size
    print("\n" + "="*60)
    print("Testing with Beam Search...")
    print("="*60)
    
    for bs in [5]:
        print(f"\n--- Beam Size = {bs} ---")
        test(model, data['test_en'], data['test_de'], device, src_vocab, tgt_vocab,
             beam_size=bs, tokenizer_fn=src_tokenize_fn)