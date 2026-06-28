import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import transformer


def translate(model, src_sentence, device, src_vocab, tgt_vocab, max_len=100, 
              beam_size=5, tokenizer_fn=None):
    """使用训练好的 Transformer 模型进行 Beam Search 解码翻译
    
    Args:
        model: Transformer 模型
        src_sentence: 源语言句子（字符串）
        device: torch device
        src_vocab: 源语言词汇表
        tgt_vocab: 目标语言词汇表
        max_len: 最大生成长度
        beam_size: beam search 宽度 (默认5，设为1等价于贪心解码)
        tokenizer_fn: 分词函数（若为None则使用默认 word-level tokenize）
    
    Returns:
        翻译后的目标语言句子（字符串）
    """
    if tokenizer_fn is None:
        tokenizer_fn = transformer.tokenize
    
    # 1. 分词 + 编码源句子
    src_ids = transformer.encode(src_sentence, src_vocab, tokenizer_fn)
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)  # (1, src_len)

    # 2. 构建反向词汇表 (id -> word)
    id_to_token = {v: k for k, v in tgt_vocab.items()}

    # 3. 编码器前向传播（只执行一次，所有 beam 共享）
    model.eval()
    with torch.no_grad():
        src_mask, _ = model.generate_mask(src_tensor, src_tensor)
        src_emb = model.src_embed(src_tensor) * torch.tensor(model.d_model ** 0.5, device=device)
        src_emb = model.pos_encoding(src_emb)
        enc_output = src_emb
        for enc_layer in model.encoder:
            enc_output = enc_layer(enc_output, src_mask)  # (1, src_len, d_model)
        
        # 扩展编码器输出以匹配 beam_size（广播复用）
        enc_output = enc_output.expand(beam_size, -1, -1)  # (beam_size, src_len, d_model)
        src_mask = src_mask.expand(beam_size, -1, -1, -1)  # (beam_size, 1, 1, src_len)
        
        # 4. Beam Search 解码
        bos_id = tgt_vocab['<bos>']
        eos_id = tgt_vocab['<eos>']
        pad_id = tgt_vocab['<pad>']
        
        # 每个 beam 项: (sequence_ids, cumulative_log_prob, finished_flag)
        # sequence_ids: list of token ids（包含 <bos>）
        beams = [([bos_id], 0.0, False)]  # 初始 beam
        
        for step in range(max_len):
            new_beams = []
            
            for seq_ids, cum_log_prob, is_finished in beams:
                if is_finished:
                    # 已结束的 beam 直接保留
                    new_beams.append((seq_ids, cum_log_prob, True))
                    continue
                
                # 构建当前步的输入
                tgt_tensor = torch.tensor([seq_ids], dtype=torch.long).to(device)  # (1, cur_len)
                
                # 生成目标掩码
                _, tgt_mask = model.generate_mask(
                    src_tensor,  # 原始 src (1, src_len)
                    tgt_tensor   # (1, cur_len)
                )
                
                # 解码器前向传播
                tgt_emb = model.tgt_embed(tgt_tensor) * torch.tensor(model.d_model ** 0.5, device=device)
                tgt_emb = model.pos_encoding(tgt_emb)
                dec_output = tgt_emb
                for dec_layer in model.decoder:
                    dec_output = dec_layer(
                        dec_output, 
                        enc_output[:1],   # 取第一个 batch（所有 beam 编码器输出相同）
                        enc_output[:1], 
                        tgt_mask, 
                        src_mask[:1]
                    )
                
                # 投影到词汇表，取最后一个位置的 logits
                logits = model.tgt_fc(dec_output)  # (1, cur_len, tgt_vocab_size)
                next_token_logits = logits[0, -1, :]  # (tgt_vocab_size,)
                
                # log softmax 转为对数概率
                log_probs = torch.log_softmax(next_token_logits, dim=-1)
                
                # 取 top-k 候选
                top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_size)
                
                for k in range(beam_size):
                    token_id = top_k_indices[k].item()
                    token_log_prob = top_k_log_probs[k].item()
                    
                    new_seq = seq_ids + [token_id]
                    new_cum_log_prob = cum_log_prob + token_log_prob
                    new_finished = (token_id == eos_id)
                    
                    new_beams.append((new_seq, new_cum_log_prob, new_finished))
            
            # 按累积对数概率排序，保留 top beam_size（带长度惩罚 α=0.6）
            new_beams.sort(key=lambda x: x[1] / (len(x[0]) ** 0.6), reverse=True)
            beams = new_beams[:beam_size]
            
            # 检查是否所有 beam 都已结束
            if all(b[2] for b in beams):
                break
        
        # 5. 选择最佳 beam（按累积概率最高，且已正常结束或最长者）
        valid_beams = [b for b in beams if b[2]]  # 已结束的
        if not valid_beams:
            valid_beams = beams  # 没有已结束的，全部考虑
        
        best_beam = max(valid_beams, key=lambda x: x[1] / (len(x[0]) ** 0.6))
        best_seq = best_beam[0]

    # 6. 解码为目标语言句子
    translated_tokens = []
    for tid in best_seq[1:]:  # 跳过 <bos>
        if tid in (pad_id, bos_id, eos_id):
            continue
        token_str = id_to_token.get(tid, '<unk>')
        translated_tokens.append(token_str)
        

    return merge_subwords(translated_tokens)

# 合并子词
def merge_subwords(tokens):
    merged = []
    current = ""
    for tok in tokens:
        if tok.endswith('</w>'):
            current += tok.replace('</w>', '')
            merged.append(current)
            current = ""
        else:
            current += tok
    if current:
        merged.append(current)
    return " ".join(merged)


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
    if use_bpe:
       src_bpe = transformer.BPETokenizer()
       src_bpe.load(BPE_SRC_PATH)
       tgt_bpe = transformer.BPETokenizer()
       tgt_bpe.load(BPE_TGT_PATH)
       src_tokenize_fn = src_bpe.tokenize
       tgt_tokenize_fn = tgt_bpe.tokenize

    # 翻译交互循环
    print("\n" + "="*60)
    print("  Beam Search Translation (beam_size=5)")
    print("  Type 'q' to quit")
    print("="*60)
    while True:
        sentence = input("\nEnglish sentence: ").strip()
        if sentence.lower() == 'q':
            print("Goodbye!")
            break
        if not sentence:
            continue
        translation = translate(model, sentence, device, src_vocab, tgt_vocab,
                                beam_size=5, tokenizer_fn=src_tokenize_fn)
        print(f"German translation: {translation}")