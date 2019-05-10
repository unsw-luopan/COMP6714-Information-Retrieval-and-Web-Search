import torch
import torch.nn.functional as F
from config import config
import numpy as np

_config = config()


def evaluate(golden_list, predict_list):
    fp = 0
    tp = 0
    fn = 0
    BT = 'B-TAR'
    BH = 'B-HYP'
    IT = 'I-TAR'
    IH = 'I-HYP'
    for i in range(len(golden_list)):
        for j in range(len(golden_list[i])):
            if predict_list[i][j] == BT or predict_list[i][j] == BH:
                if predict_list[i][j] != golden_list[i][j]:
                    fp += 1
            if golden_list[i][j] == BT or golden_list[i][j] == BH:
                if golden_list[i][j] != predict_list[i][j]:
                    fn += 1
                if golden_list[i][j] == predict_list[i][j]:
                    if j == len(golden_list[i]) - 1:
                        tp += 1
                    else:
                        for m in range(j + 1, len(golden_list[i])):
                            if golden_list[i][m] != IT and golden_list[i][m] != IH:
                                if predict_list[i][m] != IT and predict_list[i][m] != IH:
                                    tp += 1
                                else:
                                    fn += 1
                                    fp += 1
                                break
                            if predict_list[i][m] != IH and predict_list[i][m] != IT:
                                if golden_list[i][m] != IH and golden_list[i][m] != IT:
                                    tp += 1
                                else:
                                    fn += 1
                                    fp += 1
                                    break
                            if m == len(predict_list[i]) - 1:
                                if golden_list[i][m] == predict_list[i][m]:
                                    tp += 1
                                else:
                                    fn += 1
                                    fp += 1
    if tp == 0 and fp == 0 and fn == 0:
        F1_score = 1
    elif tp == 0 and (fp > 0 or fn > 0):
        F1_score = 0
    else:
        F1_score = 2 * tp / (2 * tp + fn + fp)
    return F1_score


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + ((1 - forgetgate) * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    batch_size = batch_char_index_matrices.size(0)
    word_len = batch_char_index_matrices.size(1)

    batch_char_index_matrices = batch_char_index_matrices.view(-1, batch_char_index_matrices.size(2))
    batch_word_len_lists = batch_word_len_lists.view(-1)

    input_char_embeds = model.char_embeds(batch_char_index_matrices)

    # Sort the mini-batch wrt word-lengths, to form a pack_padded sequence.
    sorted_batch_word_len_lists, perm_idx = batch_word_len_lists.sort(0, descending=True)

    sorted_char_embeds = input_char_embeds[perm_idx.view(-1)]
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    sorted_batch_word_len_lists = sorted_batch_word_len_lists.view(-1)
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    output_sequence = pack_padded_sequence(sorted_char_embeds,
                                           lengths=sorted_batch_word_len_lists.data.tolist(),
                                           batch_first=True)
    # Feed the pack_padded sequence to the char_LSTM layer.
    output_sequence, state = model.char_lstm(output_sequence)

    hidden1 = state[0].permute(1, 0, 2)
    hidden1 = hidden1[desorted_indices]  # Recover the hidden states corresponding to the sorted index
    hidden1 = hidden1.view(batch_size, word_len, -1)  # Reshape it

    return hidden1
