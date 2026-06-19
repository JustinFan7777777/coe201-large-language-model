# lab6_transformer.py
# Lab 6 Task 3: Full Encoder-Decoder Transformer [40 points]

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding from 'Attention is All You Need'.

    This is provided for you — use it in the Transformer.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# Import your Task 1 & Task 2 implementations
from lab6_encoder import TransformerEncoderBlock
from lab6_decoder import TransformerDecoderBlock


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Architecture:
        Encoder: src -> Embedding -> scale -> PosEnc -> N x EncoderBlock -> enc_output
        Decoder: tgt -> Embedding -> scale -> PosEnc -> N x DecoderBlock(enc_output) -> dec_output
        Output:  dec_output -> Linear -> logits
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        max_len,
    ):
        super().__init__()
        self.d_model = d_model

        ### TODO: Initialize all layers
        # 1. src_embedding: nn.Embedding(src_vocab_size, d_model)
        # 2. tgt_embedding: nn.Embedding(tgt_vocab_size, d_model)
        # 3. pos_encoding: PositionalEncoding(d_model, max_len)
        # 4. encoder_layers: nn.ModuleList of TransformerEncoderBlock (from Task 1)
        # 5. decoder_layers: nn.ModuleList of TransformerDecoderBlock (from Task 2)
        # 6. fc_out: nn.Linear(d_model, tgt_vocab_size)
        # --- Your code starts here ---
        
        # first, we need embeddings for source and target vocabularies
        # the embedding dimension is d_model
        # which is the same as the model dimension used in the encoder and decoder blocks
        # token_ids will be converted to dense vectors of size d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)

        # second, we need an embedding layer for the target vocabulary
        # this will be used to convert target token ids to dense vectors
        # before feeding into the decoder
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # third, we need positional encoding to
        # add positional information to the token embeddings
        # this is important because the transformer has no inherent notion of sequence order
        # max_len is the maximum sequence length we expect to handle
        # why positional encoding? because the transformer architecture is permutation invariant
        # without positional encoding, the model would treat the input as a bag of tokens
        # and lose all sequential information
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # fourth, we need to create the encoder layers
        # each encoder layer is a TransformerEncoderBlock
        # we need num_encoder_layers of them
        # we can use nn.ModuleList to hold them
        self.encoder_layers = nn.ModuleList(
            # each encoder block takes d_model, nhead, and d_ff as arguments
            # for _ in range(num_encoder_layers) creates the specified number of layers
            [TransformerEncoderBlock(d_model, nhead, d_ff) for _ in range(num_encoder_layers)]
        )

        # fifth, we need to create the decoder layers
        # each decoder layer is a TransformerDecoderBlock
        # we need num_decoder_layers of them
        # we can also use nn.ModuleList to hold them
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderBlock(d_model, nhead, d_ff) for _ in range(num_decoder_layers)]
        )

        # finally, we need a linear layer to project the decoder output to the target vocabulary size
        # this will produce the logits for each token in the target vocabulary
        # the input dimension is d_model (the output of the decoder)
        # the output dimension is tgt_vocab_size (the number of tokens in the target vocabulary)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # --- Your code ends here ---

    def encode(self, src, src_mask=None):
        """
        Encode the source sequence.

        ### TODO: Implement the encoding pass
        Steps:
        1. Apply src_embedding to src
        2. Scale by sqrt(d_model)
        3. Apply pos_encoding
        4. Pass through each encoder layer
        """
        # --- Your code starts here ---

        # token id -> embedding vector
        # shape: (batch, src_len) -> (batch, src_len, d_model)
        src = self.src_embedding(src)

        # scale the embeddings by sqrt(d_model)
        # this is a common practice in transformer models to help with training stability
        src = src * math.sqrt(self.d_model)

        # add positional encoding to the token embeddings
        # this will give the model information about the position of each token in the sequence
        src = self.pos_encoding(src)

        # pass the source through each encoder layer
        # each encoder layer will take the output of the previous layer as input
        # and optionally use the src_mask for attention
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src, src_mask)

        # the final output of the last encoder layer will be the encoder output
        return src
        # --- Your code ends here ---

    def decode(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        """
        Decode the target sequence using encoder output.

        ### TODO: Implement the decoding pass
        Steps:
        1. Apply tgt_embedding to tgt
        2. Scale by sqrt(d_model)
        3. Apply pos_encoding
        4. Pass through each decoder layer (pass enc_output to each!)
        """
        # --- Your code starts here ---

        # first, we need to convert target token ids to embeddings
        # shape: (batch, tgt_len) -> (batch, tgt_len, d_model    
        tgt = self.tgt_embedding(tgt)

        # second, we scale the target embeddings by sqrt(d_model)
        # this is the same scaling we did for the source embeddings in the encoder
        tgt = tgt * math.sqrt(self.d_model)

        # third, we add positional encoding to the target embeddings
        tgt = self.pos_encoding(tgt)

        # fourth, we pass the target through each decoder layer
        # each decoder layer will take the encoder output as an additional input
        # the decoder layers will use the encoder output for cross-attention
        # tgt_mask is the optional mask for the target sequence (causal mask for self-attention)
        # memory_mask is the optional mask for the encoder output when used in cross-attention
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt,
                enc_output,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )

        # the final output of the last decoder layer will be the decoder output
        return tgt
        # --- Your code ends here ---

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        Full forward pass: encode -> decode -> project to vocab.

        ### TODO: Implement the full forward pass
        Steps:
        1. enc_output = self.encode(src, src_mask)
        2. dec_output = self.decode(tgt, enc_output, tgt_mask, memory_mask)
        3. return self.fc_out(dec_output)
        """
        # --- Your code starts here ---

        # first, we encode the source sequence to get the encoder output
        enc_output = self.encode(src, src_mask)    

        # second, we decode the target sequence using the encoder output
        dec_output = self.decode(tgt, enc_output, tgt_mask, memory_mask)

        # finally, we project the decoder output to the target vocabulary size using the linear layer    
        out = self.fc_out(dec_output)

        # the output shape will be (batch, tgt_len, tgt_vocab_size)
        # this will give us the logits for each token in the target vocabulary
        # for each position in the target sequence
        return out
        # --- Your code ends here ---


def main():
    print("Testing Task 3: Full Transformer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 512
    num_layers = 2

    model = Transformer(
        src_vocab_size, tgt_vocab_size, d_model, 8, num_layers, num_layers, 2048, 100
    ).to(device)

    src = torch.randint(0, src_vocab_size, (2, 10)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (2, 8)).to(device)

    # Generate causal mask for decoder
    tgt_len = tgt.size(1)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(device)

    out = model(src, tgt, tgt_mask=tgt_mask)
    print(f"Full Transformer output shape: {out.shape}")
    assert out.shape == (2, 8, tgt_vocab_size), "Output shape mismatch!"
    print("Task 3 test passed!")


if __name__ == "__main__":
    main()
