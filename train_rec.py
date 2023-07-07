from copy import deepcopy
from math import ceil
import random

import torch
import numpy as np
from loguru import logger
from torch import nn, optim
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

topk = [1, 5, 10, 20, 50]


def pretrain_evaluate(model, pretrain_dataloader, epoch, results_file_path, content_hit):
    model.eval()
    hit_pt = [[], [], [], [], []]

    # Pre-training Test
    for movie_id, review_meta, review_token, review_mask in tqdm(
            pretrain_dataloader, bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
        scores, target_id = model.pre_forward(review_meta, review_token,
                                              review_mask, movie_id, compute_score=True)
        scores = scores[:, torch.LongTensor(model.movie2ids)]

        target_id = target_id.cpu().numpy()

        for k in range(len(topk)):
            sub_scores = scores.topk(topk[k])[1]
            sub_scores = sub_scores.cpu().numpy()

            for (label, score) in zip(target_id, sub_scores):
                y = model.movie2ids.index(label)
                hit_pt[k].append(np.isin(y, score))

    print('Epoch %d : pre-train test done' % epoch)
    for k in range(len(topk)):
        hit_score = np.mean(hit_pt[k])
        print('[pre-train] hit@%d:\t%.4f' % (topk[k], hit_score))

    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '[PRE TRAINING] Epoch:\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                epoch, 100 * np.mean(hit_pt[0]), 100 * np.mean(hit_pt[1]), 100 * np.mean(hit_pt[2]),
                100 * np.mean(hit_pt[3]), 100 * np.mean(hit_pt[4])))

    if epoch == 0:
        content_hit[0] = 100 * np.mean(hit_pt[0])
        content_hit[1] = 100 * np.mean(hit_pt[2])
        content_hit[2] = 100 * np.mean(hit_pt[4])


def finetuning_evaluate(model, item_rep_model, test_dataloader, item_dataloader, epoch, results_file_path, initial_hit,
                        best_hit, eval_metric, prediction, device_id, item_rep):
    hit_ft = [[], [], [], [], []]
    item_rep, movie_ids = [], []
    # Fine-tuning Test
    item_rep_bert = deepcopy(model.word_encoder)
    if prediction == 0:
        for movie_id, title, title_mask, review, review_mask, num_reviews in tqdm(item_dataloader,
                                                                                  bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            item_rep.extend(
                item_rep_model.forward(movie_id, title, title_mask, review, review_mask, num_reviews, item_rep_bert))
            movie_ids.extend(movie_id.tolist())

    for batch in test_dataloader.get_rec_data(shuffle=False):
        context_entities, context_tokens, target_items, candidate_items, _, _ = batch
        scores = model.forward(context_entities, context_tokens, torch.tensor(item_rep).to(device_id))
        # scores = scores[:, torch.LongTensor(model.movie2ids)]

        target_items = target_items.cpu().numpy()

        for k in range(len(topk)):
            sub_scores = scores.topk(topk[k])[1]
            sub_scores = sub_scores.cpu().numpy()

            for (label, score) in zip(target_items, sub_scores):
                target_idx = label  # model.movie2ids.index(label)
                hit_ft[k].append(np.isin(target_idx, score))

    print('Epoch %d : test done' % (epoch))

    for k in range(len(topk)):
        hit_score = np.mean(hit_ft[k])
        print('hit@%d:\t%.4f' % (topk[k], hit_score))

    with open(results_file_path, 'a', encoding='utf-8') as result_f:
        result_f.write(
            '[FINE TUNING] Epoch:\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % (
                epoch, 100 * np.mean(hit_ft[0]), 100 * np.mean(hit_ft[1]), 100 * np.mean(hit_ft[2]),
                100 * np.mean(hit_ft[3]), 100 * np.mean(hit_ft[4])))

    if epoch == 0:
        initial_hit[0] = 100 * np.mean(hit_ft[0])
        initial_hit[1] = 100 * np.mean(hit_ft[2])
        initial_hit[2] = 100 * np.mean(hit_ft[4])

    if np.mean(hit_ft[0]) > eval_metric[0]:
        eval_metric[0] = np.mean(hit_ft[0])
        for k in range(len(topk)):
            best_hit[k] = np.mean(hit_ft[k])


def negative_sampler(args, target_items, num_items=6923):
    negative_indices = []
    target_items = target_items.tolist()
    for target_item in target_items:
        negative_indice = []
        while len(negative_indice) < args.negative_num:
            negative_idx = random.randint(0, num_items - 1)
            # negative_idx = random.choice(candidate_knowledges)
            if (negative_idx not in negative_indice) and (negative_idx != target_item):
                negative_indice.append(negative_idx)
        negative_indices.append(negative_indice)
    return torch.tensor(negative_indices)


def train_recommender(args, model, item_rep_model, train_dataloader, test_dataloader, item_dataloader, path,
                      results_file_path):
    best_hit = [[], [], [], [], []]
    initial_hit = [[], [], []]
    content_hit = [[], [], []]
    eval_metric = [-1]
    item_rep, movie_ids = [], []

    optimizer = optim.Adam(model.parameters(), lr=args.lr_ft)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)

    # if args.prediction == 0:
    #     for movie_id, title, title_mask, review, review_mask, num_reviews in tqdm(item_dataloader,
    #                                                                               bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
    #         item_rep.extend(
    #             item_rep_model.forward(movie_id, title, title_mask, review, review_mask, num_reviews))
    #
    # item_rep = torch.tensor(item_rep).to(args.device_id)
    for epoch in range(args.epoch_ft):

        # pretrain_evaluate(model, pretrain_dataloader, epoch, results_file_path, content_hit)
        finetuning_evaluate(model, item_rep_model, test_dataloader, item_dataloader, epoch + 1, results_file_path,
                            initial_hit, best_hit, eval_metric, args.prediction, args.device_id, item_rep)

        # TRAIN
        model.train()
        item_rep_model.train()
        total_loss = 0

        logger.info(f'[Recommendation epoch {str(epoch)}]')
        logger.info('[Train]')

        for batch in train_dataloader.get_rec_data(args.batch_size):
            context_entities, context_tokens, target_items, candidate_items, title, review = batch
            batch_title, batch_review = [], []
            # negative_indices = negative_sampler(args, target_items)
            # candidate_items = torch.cat([target_items.unsqueeze(1), negative_indices], dim=1).tolist() # [B , k + 1]
            item_matrix = target_items.repeat(target_items.size()).view(target_items.size()[0], -1)
            tmp = torch.eq(item_matrix, target_items.view(target_items.size()[0], -1)).to(args.device_id)
            isSameItem = tmp.fill_diagonal_(0) * -1e20
            for item in target_items.tolist():
                title, review = [], []
                batch_title.append(train_dataloader.review_data[item]['title'])
                batch_review.append(train_dataloader.review_data[item]['review'])
                # batch_title.append(title)
                # batch_review.append(review)
            batch_title = torch.tensor(batch_title)  # [B, L]
            batch_review = torch.tensor(batch_review)  # [B, L]
            if args.forward_type == 0:
                scores_ft = model.forward(context_entities, context_tokens, item_rep)
                loss = model.criterion(scores_ft, target_items.to(args.device_id))
            elif args.forward_type == 1:
                scores_ft = model.forward_negativeSampling(context_entities, context_tokens, batch_title, batch_review)
                scores_ft = scores_ft + isSameItem
                prob = -torch.log_softmax(scores_ft, dim=1)
                loss = torch.diagonal(prob,
                                      0).mean()  # (-torch.log_softmax(scores_ft, dim=1).select(dim=1, index=0).mean())
            # loss = model.criterion(scores_ft, target_items.to(args.device_id))
            # loss_pt = model.pre_forward(review_meta, review, review_mask, target_items)
            # loss = loss_ft + ((loss_pt) * args.loss_lambda)

            total_loss += loss.data.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print('Loss:\t%.4f\t%f' % (total_loss, scheduler.get_last_lr()[0]))
    torch.save(model.state_dict(), path)  # TIME_MODELNAME 형식
    item_rep_bert = deepcopy(model.word_encoder)
    # pretrain_evaluate(model, pretrain_dataloader, epoch, results_file_path, content_hit)
    finetuning_evaluate(model, item_rep_model, test_dataloader, item_dataloader, epoch + 1, results_file_path,
                        initial_hit, best_hit, eval_metric, args.prediction, args.device_id, item_rep)

    best_result = [100 * best_hit[0], 100 * best_hit[2], 100 * best_hit[4]]

    return best_result
