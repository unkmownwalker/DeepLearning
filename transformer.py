import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import urllib.request
import tarfile
import re
import json
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
import math
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt


#--------------------------------------------------------------数据集下载-----------------------------------------------------------、
def download_multi30k(data_dir="./data"):
    """
    下载/读取 multi30k 数据集。
    仅在 data_dir 目录下查找（包括其子目录），不再向上搜索父目录。
    如果未找到，则下载并解压到 data_dir/multi30k。
    """
    url = "https://ossci-datasets.s3.amazonaws.com/torchbench/data/multi30k.tar.gz"

    # ---------- 1. 搜索已有数据目录（仅在 data_dir 及其子目录中） ----------
    expect_pattern = re.compile(r'^(train|val|test)\.(en|de)$')
    found_extract_dir = None

    # 只搜索给定的 data_dir，不再向上遍历父目录
    search_dirs = [data_dir]

    for base in search_dirs:
        candidate = os.path.join(base, "multi30k")
        if not os.path.isdir(candidate):
            continue
        # 直接在 candidate 下查找 .en/.de 文件
        top_files = [f for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))]
        if any(expect_pattern.match(f) for f in top_files):
            found_extract_dir = candidate
            break
        # 或者在嵌套子目录中查找
        for root, _, files in os.walk(candidate):
            if any(expect_pattern.match(f) for f in files):
                found_extract_dir = root
                break
        if found_extract_dir:
            break

    # ---------- 2. 未找到则下载 ----------
    if found_extract_dir is None:
        os.makedirs(data_dir, exist_ok=True)
        archive_path = os.path.join(data_dir, "multi30k.tar.gz")
        if not os.path.exists(archive_path):
            print(f"Downloading from {url} ...")
            urllib.request.urlretrieve(url, archive_path)
        extract_dir = os.path.join(data_dir, "multi30k")
        if not os.path.exists(extract_dir):
            print(f"Extracting to {extract_dir} ...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
        # 再次搜索（处理嵌套目录情况）
        top_files = [f for f in os.listdir(extract_dir) if os.path.isfile(os.path.join(extract_dir, f))]
        if any(expect_pattern.match(f) for f in top_files):
            found_extract_dir = extract_dir
        else:
            for root, _, files in os.walk(extract_dir):
                if any(expect_pattern.match(f) for f in files):
                    found_extract_dir = root
                    break
        if found_extract_dir is None:
            raise FileNotFoundError(f"Downloaded data but could not locate .en/.de files under {extract_dir}")
    else:
        print(f"Using existing data at {os.path.abspath(found_extract_dir)}")

    # ---------- 3. 读取文件 ----------
    raw_data = {}
    for fname in sorted(os.listdir(found_extract_dir)):
        file_path = os.path.join(found_extract_dir, fname)
        if not os.path.isfile(file_path):
            continue
        m = expect_pattern.match(fname)
        if m:
            split, lang = m.group(1), m.group(2)
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data[f"{split}_{lang}"] = [line.rstrip('\n') for line in f]
    return raw_data

#-------------------------------------------------- BPE 子词分词器 ----------------------------------------------------------

class BPETokenizer:
    """字节对编码 (BPE) 分词器，完全基于 Python 标准库实现，无需外部依赖"""
    
    def __init__(self, num_merges=8000):
        self.num_merges = num_merges
        self.bpe_merges = []          # 按顺序存储合并对 [(token_a, token_b), ...]
        self.bpe_ranks = {}           # {(token_a, token_b): rank} 用于快速查找合并优先级
    
    def _get_pair_stats(self, vocab):
        """统计相邻符号对的频率"""
        pairs = defaultdict(int)
        for word_tokens, freq in vocab.items():
            for i in range(len(word_tokens) - 1):
                pairs[(word_tokens[i], word_tokens[i+1])] += freq
        return pairs
    
    def _apply_merge(self, vocab, pair):
        """将最频繁的符号对合并"""
        bigram_str = pair[0] + pair[1]
        new_vocab = {}
        for word_tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == pair[0] and word_tokens[i+1] == pair[1]:
                    new_tokens.append(bigram_str)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        return new_vocab
    
    def fit(self, sentences):
        """从句子列表中学习 BPE 合并规则"""
        # 初始化：将每个词拆分为字符，末尾添加 </w> 标记词边界
        vocab = defaultdict(int)
        for sent in sentences:
            for word in sent.lower().split():
                char_list = list(word)
                char_list[-1] = char_list[-1] + '</w>'
                chars = tuple(char_list)
                vocab[chars] += 1
        
        print(f"  Initial character vocabulary size: {len(vocab)}")
        
        # 迭代合并最频繁的符号对
        for merge_i in range(self.num_merges):
            pairs = self._get_pair_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.bpe_merges.append(best_pair)
            self.bpe_ranks[best_pair] = merge_i
            vocab = self._apply_merge(vocab, best_pair)
        
        print(f"  Learned {len(self.bpe_merges)} BPE merges (requested {self.num_merges})")
    
    def tokenize(self, text):
        """使用已学习的 BPE 合并规则对文本进行分词"""
        tokens = []
        for word in tokenizes(text):
            char_list = list(word)
            char_list[-1] = char_list[-1] + '</w>'
            chars = char_list  
            # 重复应用合并，直到无法再合并
            while True:
                # 找到最早学到的可合并符号对（即合并优先级最高）
                best_rank = len(self.bpe_merges)  # 默认：无合并
                best_pos = -1
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i+1])
                    if pair in self.bpe_ranks:
                        rank = self.bpe_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pos = i
                if best_pos == -1:
                    break
                # 执行合并
                merged = chars[best_pos] + chars[best_pos + 1]
                chars = chars[:best_pos] + [merged] + chars[best_pos + 2:]
            # 移除词边界标记并收集结果
            result =[]
            for c in chars:
                if c == '</w>':
                    result[-1] = result[-1] + '</w>'
                else:
                    result.append(c)

            if result:
                tokens.extend(result)

        return tokens
    
    def save(self, path):
        """保存 BPE 合并规则到 JSON 文件"""
        data = {
            'num_merges': self.num_merges,
            'bpe_merges': self.bpe_merges,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path):
        """从 JSON 文件加载 BPE 合并规则"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.num_merges = data['num_merges']
        self.bpe_merges = [tuple(pair) for pair in data['bpe_merges']]
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.bpe_merges)}
        print(f"  Loaded {len(self.bpe_merges)} BPE merges from {path}")


#--------------------------------------------------数据预处理-----------------------------------------------------------
def tokenizes(text):
     text = text.lower()
     text = re.sub(r'([.,!?;:()\[\]{}\'"“”])', r' \1 ', text) #正则表达式，匹配字符加空格
     tokens = [tok for tok in text.split() if tok]
     return tokens

#构建词汇表
def build_vocab(sentences, min_freq=2, max_size=10000, tokenizer_fn=BPETokenizer.tokenize):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenizer_fn(sent))
    vocab = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}      #<pad>填充，<unk>未知词，<bos>开始，<eos>结束
    for word ,freq in counter.most_common(max_size - len(vocab)): #counter.most_common(n)返回前n个最常用的元素,降序
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

#编码
def encode(sentence, vocab, tokenizer_fn):
    ids=[]
    tokens= tokenizer_fn(sentence)
    for token in tokens[:]:
        if token not in vocab:
            ids.append(vocab['<unk>'])
        else:
            ids.append(vocab[token])
    return [vocab['<bos>']] + ids + [vocab['<eos>']]

#批量封装数据
class TranslationDataset(Dataset):
    #初始编码
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab,
                 src_tokenizer_fn=BPETokenizer.tokenize, tgt_tokenizer_fn=BPETokenizer.tokenize):
        self.pairs = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_ids = encode(src, src_vocab, src_tokenizer_fn)
            tgt_ids = encode(tgt, tgt_vocab, tgt_tokenizer_fn)
            self.pairs.append((torch.tensor(src_ids), torch.tensor(tgt_ids)))

    #后面两个方法为继承 Dataset 必须实现
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

#batch 填充
def padding(batch, src_vocab, tgt_vocab):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab['<pad>'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab['<pad>'])
    return src_batch, tgt_batch
    
# --------------------------------------------Transformer 模型 ---------------------------------------------
#位置编码
class positional_encoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)   # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数索引列
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数索引列
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer('pe', pe) #注明buffer,不参与梯度计算,无需每次重新生成

    def forward(self, x):
    # x: (batch, seq_len, d_model)
       x = x + self.pe[:, :x.size(1), :]   # 截取前 seq_len 个位置
       return x

#多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
      super().__init__()
      self.n_heads = n_heads
      self.d_model = d_model
      self.d_k = d_model // n_heads
      self.w_q = nn.Linear(d_model, d_model)   # 查询投影矩阵 Ω_q
      self.w_k = nn.Linear(d_model, d_model)   # 键投影矩阵 Ω_k
      self.w_v = nn.Linear(d_model, d_model)   # 值投影矩阵 Ω_v
      self.w_o = nn.Linear(d_model, d_model)   # 输出投影矩阵 W_O
      self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):#这里的q,k,v为传入的，对于单编码，q=k=v=x
        batch_size = q.size(0)
        q = self.w_q(q)   # (batch, seq_len_q, d_model)
        k = self.w_k(k)   # (batch, seq_len_k, d_model)
        v = self.w_v(v)   # (batch, seq_len_v, d_model)
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)   # (batch, n_heads, seq_len_q, d_k)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)   # (batch, n_heads, seq_len_k, d_k)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)   # (batch, n_heads, seq_len_v, d_k)
        scores=torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(self.d_k)   # (batch, n_heads, seq_len_q, seq_len_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention=self.dropout(torch.softmax(scores,dim=-1))
        context=torch.matmul(attention,v)   # (batch, n_heads, seq_len_q, d_k)
        context=context.permute(0,2,1,3).contiguous()   # (batch, seq_len_q, n_heads, d_k),contiguous()使得内存连续，view要求内存连续
        context=context.view(batch_size,-1,self.d_model)   # (batch, seq_len_q, d_model)
        output=self.w_o(context)   # (batch, seq_len_q, d_model)，最后的矩阵将不同注意力混合
        return output

#前馈网络
class FFN(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.2):
        super().__init__()
        self.ffn=nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.ffn(x)
    
#编码器
class Encoder(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.2):
        super().__init__()
        self.self_attn=MultiHeadAttention(d_model,n_heads,dropout=dropout)
        self.ffn = FFN(d_model,d_ff,dropout=dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
#解码器
class Decoder(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.2):
        super().__init__()
        self.self_attn= MultiHeadAttention(d_model,n_heads,dropout=dropout)
        self.cross_attn= MultiHeadAttention(d_model,n_heads,dropout=dropout)
        self.ffn = FFN(d_model,d_ff,dropout=dropout)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,enc_k, enc_v,self_mask,cross_mask=None):
        attn_output = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_output = self.cross_attn(x, enc_k, enc_v, cross_mask)
        x = self.norm2(x + self.dropout(cross_output))
        ff_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

#Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=8, num_layers=4, dim_ff=512, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.num_layers = num_layers
        #词嵌入
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = positional_encoding(d_model)
        #编码器解码器
        self.encoder = nn.ModuleList([Encoder(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([Decoder(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        #输出投影
        self.tgt_fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        #初始化权重
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
           if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # 改成 Xavier
           else:
            nn.init.constant_(p, 0)
        # 额外：对 LayerNorm 的 weight 和 bias 进行专门初始化
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
               nn.init.constant_(module.weight, 1.0)
               nn.init.constant_(module.bias, 0.0)

    def generate_mask(self, src, tgt):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        # 源掩码：忽略填充符
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
        # 目标填充掩码
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
        # 未来掩码（下三角）
        tgt_len = tgt.size(1)
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=src.device)).bool()
        # 合并填充掩码和未来掩码
        tgt_mask = tgt_pad_mask & subsequent_mask.unsqueeze(0)  # (batch, 1, tgt_len, tgt_len)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        """
        src: (batch, src_len)  源语言 token 索引
        tgt: (batch, tgt_len)  目标语言 token 索引（训练时使用 teacher forcing)
        返回: (batch, tgt_len, tgt_vocab_size) 每个位置上的词汇分布
        """
        # 1. 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # 2. 词嵌入 + 缩放（论文中除以 sqrt(d_model) 的逆操作，实际是乘以 sqrt(d_model)）
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)   # (batch, src_len, d_model)
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)   # (batch, tgt_len, d_model)

        # 3. 添加位置编码
        src_emb = self.pos_encoding(src_emb)   # (batch, src_len, d_model)
        tgt_emb = self.pos_encoding(tgt_emb)   # (batch, tgt_len, d_model)

        # 4. 编码器：逐层传递
        enc_output = src_emb
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output, src_mask)   # (batch, src_len, d_model)

        # 5. 解码器：逐层传递，交叉注意力使用编码器输出作为 K 和 V
        dec_output = tgt_emb
        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output, enc_output, tgt_mask,src_mask)   # (batch, tgt_len, d_model)

        # 6. 输出投影到目标词汇表
        output = self.tgt_fc(dec_output)   # (batch, tgt_len, tgt_vocab_size)
        return output

#-----------------------------------------------------------训练部分-----------------------------------------------------------
def one_epoch_train(model,device,loader,optimizer,criterion,scheduler):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing：输入不包含最后一个 token，输出不包含第一个 token
        tgt_input = tgt[:, :-1]   # (batch, tgt_len-1)
        tgt_output = tgt[:, 1:]   # (batch, tgt_len-1)
        optimizer.zero_grad()
        #前向传播
        logits = model(src, tgt_input)  # (batch, tgt_len-1, vocab_size)
        #计算损失
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪梯度，防止梯度爆炸
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model,device,loader,criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            # Teacher forcing：输入不包含最后一个 token，输出不包含第一个 token
            tgt_input = tgt[:, :-1]   # (batch, tgt_len-1)
            tgt_output = tgt[:, 1:]   # (batch, tgt_len-1)
            logits = model(src, tgt_input)  # (batch, tgt_len-1, vocab_size)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(loader)

#主函数
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 数据集下载与划分
    data = download_multi30k(data_dir=r"D:\code\python\udlbook-notebooks\Notebooks\myself\data")
    train_en=data['train_en']
    train_de=data['train_de']
    val_en=data['val_en']
    val_de=data['val_de']
    test_en=data['test_en']
    test_de=data['test_de']
    
    # ------------------ BPE 分词器（支持缓存） ------------------
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BPE_SRC_PATH = os.path.join(SCRIPT_DIR, "bpe_src.json")
    BPE_TGT_PATH = os.path.join(SCRIPT_DIR, "bpe_tgt.json")

    src_bpe = BPETokenizer(num_merges=2000)
    tgt_bpe = BPETokenizer(num_merges=3000)

    if os.path.exists(BPE_SRC_PATH) and os.path.exists(BPE_TGT_PATH):
        print("Loading cached BPE tokenizers...")
        src_bpe.load(BPE_SRC_PATH)
        tgt_bpe.load(BPE_TGT_PATH)
    else:
        print("Training BPE tokenizer for English (source)...")
        src_bpe.fit(train_en)
        src_bpe.save(BPE_SRC_PATH)
    
        print("Training BPE tokenizer for German (target)...")
        tgt_bpe.fit(train_de)
        tgt_bpe.save(BPE_TGT_PATH)

    src_tokenize_fn = src_bpe.tokenize
    tgt_tokenize_fn = tgt_bpe.tokenize

    

    # ---------- 加载预训练模型（如果存在） ----------
    MODEL_PATH = os.path.join(SCRIPT_DIR, "best_transformer.pt")
    initial_val_loss = None
    best_val_loss = float('inf')
    skip_load = False
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'src_vocab' in checkpoint and 'tgt_vocab' in checkpoint:
          src_vocab = checkpoint['src_vocab']
          tgt_vocab = checkpoint['tgt_vocab']
          print("Loaded vocab from checkpoint.")
    else:
        src_vocab = build_vocab(train_en, min_freq=1, max_size=20000, tokenizer_fn=src_tokenize_fn)
        tgt_vocab = build_vocab(train_de, min_freq=1, max_size=20000, tokenizer_fn=tgt_tokenize_fn)
        print("No checkpoint found, starting from scratch.")
        
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    train_dataset = TranslationDataset(train_en, train_de, src_vocab, tgt_vocab, 
                                         src_tokenizer_fn=src_tokenize_fn, tgt_tokenizer_fn=tgt_tokenize_fn)
    val_dataset = TranslationDataset(val_en, val_de, src_vocab, tgt_vocab,
                                       src_tokenizer_fn=src_tokenize_fn, tgt_tokenizer_fn=tgt_tokenize_fn)
    test_dataset = TranslationDataset(test_en, test_de, src_vocab, tgt_vocab,
                                        src_tokenizer_fn=src_tokenize_fn, tgt_tokenizer_fn=tgt_tokenize_fn)
    batch_size = 128
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=partial(padding, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(padding, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    ) 
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(padding, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    )



    #模型初始化
    model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=128,nhead=8,
    num_layers=4,dim_ff=512,
    dropout=0.2).to(device)

    criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_vocab['<pad>'],
    label_smoothing=0.1)

    optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.03)

    n_epochs = 100
    total_steps=len(train_loader) * n_epochs
    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)


    if os.path.exists(MODEL_PATH):
       model.load_state_dict(checkpoint['model_state_dict'])
       #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
       best_val_loss = checkpoint['best_val_loss']
       print("Resumed training from checkpoint.")


    patience = 10          # 早停耐心值
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # 开始训练
    print("Starting training with BPE...")
    for epoch in range(1, n_epochs + 1):
        train_loss = one_epoch_train(model, device, train_loader, optimizer, criterion,scheduler)
        val_loss = evaluate(model, device, val_loader, criterion)
        current_lr = optimizer.param_groups[0]['lr'] 
        print(f"Epoch {epoch}: train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, learning rate: {current_lr:.6f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        #早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, MODEL_PATH)
            print("  -> New best model saved!")
            patience_counter = 0
        else:
             patience_counter += 1
             if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
        
    print("Training finished.")
    test_loss = evaluate(model, device, test_loader, criterion)
    print(f"Test loss: {test_loss:.4f}")


    #绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves (BPE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SCRIPT_DIR, 'loss_curves_bpe1.png'))
    plt.show()