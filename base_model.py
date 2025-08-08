import torch
import torch.nn as nn

# Multi-head attention mechanism class

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out%num_heads==0), "d_out must be a multiple of num_heads"
        self.head_dim = d_out//num_heads
        self.num_heads = num_heads
        self.d_out = d_out
        self.Q_layer = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.K_layer = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.V_layer = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_size, context_size), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        Q = self.Q_layer(x)
        K = self.K_layer(x)
        V = self.V_layer(x)

        Q = Q.view(b, num_tokens, self.num_heads, self.head_dim)
        K = K.view(b, num_tokens, self.num_heads, self.head_dim)
        V = V.view(b, num_tokens, self.num_heads, self.head_dim)
        
        Q.transpose_(1, 2)
        K.transpose_(1, 2)
        V.transpose_(1, 2)
        
        attention_scores = Q@K.transpose(-1, -2)
        
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        attention_scores.masked_fill_(mask_bool, -torch.inf)
        
        attention_weights = torch.softmax(attention_scores/self.head_dim**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = (attention_weights@V).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        context_vec = self.out_proj(context_vec)
        
        return context_vec
        

# Model Configurations

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 1024,
    "emb_dim" : 768,
    "n_heads" : 12,
    "n_layers" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

# Layer Normalization class

class layer_norm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_out = (x - mean)/(torch.sqrt(var) + self.eps)
        return self.scale*norm_out + self.shift
    
# GELU activation function class
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        out = 0.5*x*(1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))*(x + 0.044715*x**3)))
        return out

# Feed Forward class (FF)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
            )
    def forward(self, x):
        return self.layers(x)

# Transformer block class

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = layer_norm(cfg['emb_dim'])
        self.Masked_multi_head_attn = MultiHeadAttention(
            cfg['emb_dim'], cfg['emb_dim'],
            cfg['context_length'], cfg['drop_rate'], cfg['n_heads'], cfg['qkv_bias'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.layer_norm2 = layer_norm(cfg['emb_dim'])
        self.ff = FeedForward(cfg)
           
    def forward(self, x):
        
        shortcut = x
        
        x = self.layer_norm1(x)
        x = self.Masked_multi_head_attn(x)
        x = self.dropout(x) + shortcut
        
        shortcut = x
        
        x = self.layer_norm2(x)
        x = self.ff(x)
        x = self.dropout(x) + shortcut
        
        return x

# Defining model

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.transformer_layer = nn.Sequential(*[Transformer(cfg) for _ in range(cfg['n_layers'])])
        self.final_norm_layer = layer_norm(cfg['emb_dim'])
        self.linear_out_layer = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_emb = self.tok_emb(in_idx)
        positional_emb = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        embeddings = token_emb + positional_emb
        
        embeddings = self.dropout(embeddings)
        
        x = self.transformer_layer(embeddings)
        
        x = self.final_norm_layer(x)
        
        logits = self.linear_out_layer(x)
        
        return logits
    

# Function to check if two tensors rigth and left have same shape and return the right

def assign(left, right):
    if left.shape!=right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                         f"Right: {right.shape}"
                         )
    return torch.nn.Parameter(torch.tensor(right, device=left.device))

# Function to replace the weights of the model with OpenAI's weights

import numpy as np

def load_params_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"]["w"]), 3, axis=-1)
        gpt.transformer_layer[b].Masked_multi_head_attn.Q_layer.weight = assign(
            gpt.transformer_layer[b].Masked_multi_head_attn.Q_layer.weight, q_w.T)
        gpt.transformer_layer[b].Masked_multi_head_attn.K_layer.weight = assign(
            gpt.transformer_layer[b].Masked_multi_head_attn.K_layer.weight, k_w.T)
        gpt.transformer_layer[b].Masked_multi_head_attn.V_layer.weight = assign(
            gpt.transformer_layer[b].Masked_multi_head_attn.V_layer.weight, v_w.T)
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"]["b"]), 3, axis=-1)
        gpt.transformer_layer[b].Masked_multi_head_attn.Q_layer.bias = assign(
            gpt.transformer_layer[b].Masked_multi_head_attn.Q_layer.bias, q_b)
        gpt.transformer_layer[b].Masked_multi_head_attn.K_layer.bias = assign(
            gpt.transformer_layer[b].Masked_multi_head_attn.K_layer.bias, k_b)
        gpt.transformer_layer[b].Masked_multi_head_attn.V_layer.bias = assign(
            gpt.transformer_layer[b].Masked_multi_head_attn.V_layer.bias, v_b)
        
        gpt.transformer_layer[b].Masked_multi_head_attn.out_proj.weight = assign(
            gpt.transformer_layer[b].Masked_multi_head_attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_layer[b].Masked_multi_head_attn.out_proj.bias = assign(
            gpt.transformer_layer[b].Masked_multi_head_attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        gpt.transformer_layer[b].ff.layers[0].weight = assign(
            gpt.transformer_layer[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_layer[b].ff.layers[0].bias = assign(
            gpt.transformer_layer[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_layer[b].ff.layers[2].weight = assign(
            gpt.transformer_layer[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_layer[b].ff.layers[2].bias = assign(
            gpt.transformer_layer[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        gpt.transformer_layer[b].layer_norm1.scale = assign(
            gpt.transformer_layer[b].layer_norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_layer[b].layer_norm1.shift = assign(
            gpt.transformer_layer[b].layer_norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_layer[b].layer_norm2.scale = assign(
            gpt.transformer_layer[b].layer_norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_layer[b].layer_norm2.shift = assign(
            gpt.transformer_layer[b].layer_norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
        
    gpt.final_norm_layer.scale = assign(gpt.final_norm_layer.scale, params["g"])
    gpt.final_norm_layer.shift = assign(gpt.final_norm_layer.shift, params["b"])
    gpt.linear_out_layer.weight = assign(gpt.linear_out_layer.weight, params["wte"])
    
    
    
#####------Functions to take input text and generate model output--------#####

def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    tokens = torch.tensor(encoded).unsqueeze(0)
    return tokens

def tokens_to_text(tokens, tokenizer):
    tokens = tokens.squeeze(0).tolist()
    text = tokenizer.decode(tokens)
    return text

# Function to generate output tokens from input tokens

def generated_tokens(model, idx, max_new_tokens, context_size,
                               temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        input = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(input)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits<min_val,
                                 torch.tensor(float('-inf')).to(logits.device),
                                 logits)
        if temperature > 0.0:
            logits = logits/temperature
            prob_dist = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob_dist, num_samples=1)
        else:
            prob_dist = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(prob_dist, dim=-1, keepdim=True)
        if idx_next==eos_id:
            break
        idx = torch.cat([idx, idx_next], dim=-1)
    return idx


# Function to take input text and predict next words based on the model

def generate(model, tokenizer, input_context, max_new_tokens, context_size,
                               temperature=0.0, top_k=None, eos_id=None):
    model.eval()
    input_tkns = text_to_tokens(input_context, tokenizer)
    output_tkns = generated_tokens(model, input_tkns, max_new_tokens, context_size,
                               temperature, top_k, eos_id)
    output_txt = tokens_to_text(output_tkns, tokenizer)
    print(output_txt)

