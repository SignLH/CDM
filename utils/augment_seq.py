import random
import numpy as np
import math

















































def augment_kt_seqs(
    q_seq,
    s_seq,
    r_seq,
    mask_prob,
    crop_prob,
    permute_prob,
    
    negative_prob,
    
    
    q_mask_id,
    s_mask_id,
    seq_len,
    seed=None,
    skill_rel=None,
):
    
    
    rng = random.Random(seed)
    np.random.seed(seed)

    masked_q_seq = []
    masked_s_seq = []
    masked_r_seq = []
    negative_r_seq = []

    if mask_prob > 0:
        for q, s, r in zip(q_seq, s_seq, r_seq):
            prob = rng.random()
            if prob < mask_prob and s != 0:
                prob /= mask_prob
                if prob < 0.8:
                    masked_q_seq.append(q_mask_id)
                    masked_s_seq.append(s_mask_id)
                elif prob < 0.9:
                    masked_q_seq.append(
                        rng.randint(1, q_mask_id - 1)
                    )  
                    masked_s_seq.append(
                        rng.randint(1, s_mask_id - 1)
                    )  
                else:
                    masked_q_seq.append(q)
                    masked_s_seq.append(s)
            else:
                masked_q_seq.append(q)
                masked_s_seq.append(s)
            masked_r_seq.append(r)  

            
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)
    else:
        masked_q_seq = q_seq[:]
        masked_s_seq = s_seq[:]
        masked_r_seq = r_seq[:]

        for r in r_seq:
            
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)



















    true_seq_len = np.sum(np.asarray(q_seq) != 0)
    if permute_prob > 0:
        reorder_seq_len = math.floor(permute_prob * true_seq_len)
        start_idx = (np.asarray(q_seq) != 0).argmax()
        while True:
            start_pos = rng.randint(start_idx, seq_len - reorder_seq_len)
            if start_pos + reorder_seq_len < seq_len:
                break

        
        perm = np.random.permutation(reorder_seq_len)
        masked_q_seq = (
            masked_q_seq[:start_pos]
            + np.asarray(masked_q_seq[start_pos : start_pos + reorder_seq_len])[
                perm
            ].tolist()
            + masked_q_seq[start_pos + reorder_seq_len :]
        )
        masked_s_seq = (
            masked_s_seq[:start_pos]
            + np.asarray(masked_s_seq[start_pos : start_pos + reorder_seq_len])[
                perm
            ].tolist()
            + masked_s_seq[start_pos + reorder_seq_len :]
        )
        masked_r_seq = (
            masked_r_seq[:start_pos]
            + np.asarray(masked_r_seq[start_pos : start_pos + reorder_seq_len])[
                perm
            ].tolist()
            + masked_r_seq[start_pos + reorder_seq_len :]
        )

    
    if 0 < crop_prob < 1:
        crop_seq_len = math.floor(crop_prob * true_seq_len)
        if crop_seq_len == 0:
            crop_seq_len = 1
        start_idx = (np.asarray(q_seq) != 0).argmax()
        while True:
            start_pos = rng.randint(start_idx, seq_len - crop_seq_len)
            if start_pos + crop_seq_len < seq_len:
                break

        masked_q_seq = masked_q_seq[start_pos : start_pos + crop_seq_len]
        masked_s_seq = masked_s_seq[start_pos : start_pos + crop_seq_len]
        masked_r_seq = masked_r_seq[start_pos : start_pos + crop_seq_len]

    pad_len = seq_len - len(masked_q_seq)

    attention_mask = [0] * pad_len + [1] * len(masked_s_seq)
    masked_q_seq = [0] * pad_len + masked_q_seq
    masked_s_seq = [0] * pad_len + masked_s_seq
    masked_r_seq = [-1] * pad_len + masked_r_seq

    return masked_q_seq, masked_s_seq, masked_r_seq, negative_r_seq, attention_mask


def preprocess_qr(questions, responses, seq_len, pad_val=-1):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_responses = []

    for q, r in zip(questions, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])

            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )

    return preprocessed_questions, preprocessed_responses


def preprocess_qsr(questions, skills, responses, seq_len, pad_val=0):
    """
    split the interactions whose length is more than seq_len
    """
    preprocessed_questions = []
    preprocessed_skills = []
    preprocessed_responses = []
    attention_mask = []

    for q, s, r in zip(questions, skills, responses):
        i = 0
        while i + seq_len < len(q):
            preprocessed_questions.append(q[i : i + seq_len])
            preprocessed_skills.append(s[i : i + seq_len])
            preprocessed_responses.append(r[i : i + seq_len])
            attention_mask.append(np.ones(seq_len))
            i += seq_len

        preprocessed_questions.append(
            np.concatenate([q[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_skills.append(
            np.concatenate([s[i:], np.array([pad_val] * (i + seq_len - len(q)))])
        )
        preprocessed_responses.append(
            np.concatenate([r[i:], np.array([-1] * (i + seq_len - len(q)))])
        )
        attention_mask.append(
            np.concatenate(
                [np.ones_like(r[i:]), np.array([0] * (i + seq_len - len(q)))]
            )
        )

    return (
        preprocessed_questions,
        preprocessed_skills,
        preprocessed_responses,
        attention_mask,
    )
