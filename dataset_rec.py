import random
from collections import defaultdict
from copy import copy, deepcopy

from torch.utils.data import Dataset
import torch
import json
from loguru import logger

from tqdm import tqdm
import os

import numpy as np


class ContentInformation(Dataset):
    def __init__(self, args, data_path, tokenizer, device):
        super(Dataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_samples = dict()
        self.device = device
        self.crsid2id = json.load(open(os.path.join(self.data_path, 'crsid2id.json'), 'r', encoding='utf-8'))
        self.entity2id = json.load(
            open(os.path.join(data_path, 'entity2id.json'), 'r',
                 encoding='utf-8'))  # to convert review meta to entity id
        self.movie2name = json.load(open(os.path.join(data_path, 'movie2name.json'), 'r',
                                         encoding='utf-8'))  # to convert movie crs id toentity id
        self.read_data(args.max_review_len)  # read review text and meta
        self.key_list = list(self.data_samples.keys())  # movie id list

    def read_data(self, max_review_len):
        data = json.load(open(os.path.join(self.data_path, 'content_data.json'), encoding='utf-8'))[0]
        review_phrase = json.load(open(os.path.join(self.data_path, 'reviewPhrase.json'), encoding='utf-8'))

        for sample in tqdm(zip(data, review_phrase), bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}'):
            phrase_list, phrase_mask_list = [], []

            crs_id = str(sample[1]['crs_id'])
            label = self.crsid2id[crs_id]
            phrases = sample[1]['phrases'][:self.args.n_review]
            reviews = sample[0]['review'][:self.args.n_review]
            genre = " ".join(sample[0]['meta']['genre'])
            director = " ".join(sample[0]['meta']['director'])
            writers = " ".join(sample[0]['meta']['writers'])
            stars = " ".join(sample[0]['meta']['stars'])

            # if self.movie2name[crs_id][0] == -1:
            #     continue

            title = self.movie2name[crs_id][1]
            seed_keywords = genre + " " + director + " " + writers + " " + stars
            phrase_num = min(len(phrases), self.args.n_review)
            #     phrases = ['']
            if self.args.source == 0:
                if len(reviews) != 0:
                    sampled_reviews = [review for review in reviews]
                    tokenized_phrases = self.tokenizer(sampled_reviews, max_length=max_review_len,
                                                       padding='max_length',
                                                       truncation=True,
                                                       add_special_tokens=True)
                    tokenized_title = self.tokenizer(title + " " + seed_keywords,
                                                     max_length=max_review_len,
                                                     padding='max_length',
                                                     truncation=True,
                                                     add_special_tokens=True)
                else:
                    sampled_reviews = []
                    tokenized_title = self.tokenizer(title + " " + seed_keywords, max_length=max_review_len,
                                                     padding='max_length',
                                                     truncation=True,
                                                     add_special_tokens=True)

                for i in range(min(len(sampled_reviews), self.args.n_review)):
                    phrase_list.append(tokenized_phrases.input_ids[i])
                    phrase_mask_list.append(tokenized_phrases.attention_mask[i])

                for i in range(self.args.n_review - len(sampled_reviews)):
                    zero_vector = [0] * max_review_len
                    phrase_list.append(zero_vector)
                    phrase_mask_list.append(zero_vector)
                    # phrase_list.append(tokenized_title.input_ids)
                    # phrase_mask_list.append(tokenized_title.attention_mask)

                if crs_id in self.data_samples.keys():
                    print()
                self.data_samples[label] = {
                    "title": tokenized_title.input_ids,
                    "title_mask": tokenized_title.attention_mask,
                    "review": phrase_list,
                    "review_mask": phrase_mask_list,
                    "num_reviews": phrase_num
                }
            elif self.args.source == 1:
                if len(phrases) != 0:
                    sampled_reviews = [phrase for phrase in phrases]
                    tokenized_phrases = self.tokenizer(sampled_reviews, max_length=max_review_len,
                                                       padding='max_length',
                                                       truncation=True,
                                                       add_special_tokens=True)
                    tokenized_title = self.tokenizer(title + " " + seed_keywords,
                                                     max_length=max_review_len,
                                                     padding='max_length',
                                                     truncation=True,
                                                     add_special_tokens=True)
                else:
                    sampled_reviews = []
                    tokenized_title = self.tokenizer(title + " " + seed_keywords, max_length=max_review_len,
                                                     padding='max_length',
                                                     truncation=True,
                                                     add_special_tokens=True)

                for i in range(min(len(sampled_reviews), self.args.n_review)):
                    phrase_list.append(tokenized_phrases.input_ids[i])
                    phrase_mask_list.append(tokenized_phrases.attention_mask[i])

                for i in range(self.args.n_review - len(sampled_reviews)):
                    zero_vector = [0] * max_review_len
                    phrase_list.append(zero_vector)
                    phrase_mask_list.append(zero_vector)
                    # phrase_list.append(tokenized_title.input_ids)
                    # phrase_mask_list.append(tokenized_title.attention_mask)

                if crs_id in self.data_samples.keys():
                    print()
                self.data_samples[label] = {
                    "title": tokenized_title.input_ids,
                    "title_mask": tokenized_title.attention_mask,
                    "review": phrase_list,
                    "review_mask": phrase_mask_list,
                    "num_reviews": phrase_num
                }

        logger.debug('Total number of content samples:\t%d' % len(self.data_samples))

    def __getitem__(self, item):
        idx = self.key_list[item]  # entity id
        title = self.data_samples[idx]['title']
        title_mask = self.data_samples[idx]['title_mask']
        review_token = self.data_samples[idx]['review']
        review_mask = self.data_samples[idx]['review_mask']
        num_reviews = self.data_samples[idx]['num_reviews']

        # review_exist_num = np.count_nonzero(np.sum(np.array(review_mask), axis=1))
        #
        # # randomly sample review
        # if review_exist_num == 0:
        #     review_exist_num = 1
        # review_sample_idx = [random.randint(0, review_exist_num - 1) for _ in range(self.args.n_sample)]

        # review_token = [review_token[k] for k in review_sample_idx]
        # review_mask = [review_mask[k] for k in review_sample_idx]

        idx = torch.tensor(int(idx)).to(self.args.device_id)
        title = torch.LongTensor(title).to(self.args.device_id)  # [L, ]
        title_mask = torch.LongTensor(title_mask).to(self.args.device_id)  # [L, ]
        review_token = torch.LongTensor(review_token).to(self.args.device_id)  # [R, L]
        review_mask = torch.LongTensor(review_mask).to(self.args.device_id)  # [R, L]
        num_review_mask = torch.tensor([1] * num_reviews + [0] * (self.args.n_review - num_reviews)).to(
            self.args.device_id)

        return idx, title, title_mask, review_token, review_mask, num_review_mask

    def __len__(self):
        return len(self.data_samples)


class CRSDatasetRec:
    def __init__(self, args, data_path, tokenizer, kg_information):
        super(CRSDatasetRec, self).__init__()
        self.args = args
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.movie2name = kg_information.movie2name
        self.entity2id = kg_information.entity2id
        self.crsid2id = json.load(open(os.path.join(self.data_path, 'crsid2id.json'), 'r', encoding='utf-8'))
        self._load_data()

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.data_path, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.data_path, 'train_data.json')}]")
        with open(os.path.join(self.data_path, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.data_path, 'valid_data.json')}]")
        with open(os.path.join(self.data_path, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.data_path, 'test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_data(self):
        train_data_raw, valid_data_raw, test_data_raw = self._load_raw_data()  # load raw train, valid, test data

        self.train_data = self._raw_data_process(train_data_raw)  # training sample 생성
        logger.debug("[Finish train data process]")
        self.test_data = self._raw_data_process(test_data_raw)
        logger.debug("[Finish test data process]")
        self.valid_data = self._raw_data_process(valid_data_raw)
        logger.debug("[Finish valid data process]")

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._merge_conv_data(conversation["dialog"]) for
                           conversation in tqdm(raw_data,
                                                bar_format=' {percentage:3.0f} % | {bar:23} {r_bar}')]  # 연속해서 나온 대화들 하나로 합침 (예) S1, S2, R1 --> S1 + S2, R1
        augmented_conv_dicts = []
        # augment_dataset = []
        # for conv_dict in tqdm(augmented_convs):
        #     if conv_dict['role'] == 'Recommender':
        #         for idx, movie in enumerate(conv_dict['items']):
        #             augment_conv_dict = deepcopy(conv_dict["text"])
        #             augment_conv_dict['item'] = movie
        #             # augment_conv_dict['review'] = conv_dict['review'][idx]
        #             # augment_conv_dict['review_mask'] = conv_dict['review_mask'][idx]
        #             augment_dataset.append(augment_conv_dict)

        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))  # conversation length 만큼 training sample 생성
        return augmented_conv_dicts

    def _merge_conv_data(self, dialog):
        augmented_convs = []
        last_role = None

        for utt in dialog:
            movie_ids = []
            # BERT_tokenzier 에 입력하기 위해 @IDX 를 해당 movie의 name으로 replace
            for idx, word in enumerate(utt['text']):
                if word[0] == '@' and word[1:].isnumeric():
                    utt['text'][idx] = '%s' % (self.movie2name[word[1:]][1])
                    if word[1:] in self.crsid2id.keys():
                        movie_ids.append(self.crsid2id[word[1:]])

            text = ' '.join(utt['text'])
            # movie_ids = [self.entity2id[movie] for movie in utt['movies'] if
            #              movie in self.entity2id]  # utterance movie(entity2id) 마다 entity2id 저장
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if
                          entity in self.entity2id]  # utterance entity(entity2id) 마다 entity2id 저장

            if utt["role"] == last_role:
                augmented_convs[-1]["text"] += ' ' + text
                augmented_convs[-1]["movie"] += movie_ids
                augmented_convs[-1]["entity"] += entity_ids
            else:

                if utt["role"] == 'Recommender':
                    role_name = 'System'
                else:
                    role_name = 'User'

                augmented_convs.append({
                    "role": utt["role"],
                    "text": f'{role_name}: {text}',  # role + text
                    "entity": entity_ids,
                    "movie": movie_ids,
                })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_items = [], [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies = conv["text"], conv["entity"], conv["movie"]
            text_tokens = text_tokens + self.sep_token
            text_token_ids = self.tokenizer(text_tokens, add_special_tokens=False).input_ids
            plot_meta, plot, plot_mask, review, review_mask = [], [], [], [], []
            if len(context_tokens) > 0:
                # for movie in movies:
                #     review.append(self.content_dataset.data_samples[movie]['review'])
                #     review_mask.append(self.content_dataset.data_samples[movie]['review_mask'])

                conv_dict = {
                    "role": conv['role'],
                    "context_tokens": copy(context_tokens),
                    "response": text_token_ids,  # text_tokens,
                    "context_entities": copy(context_entities),
                    "context_items": copy(context_items),
                    "items": movies

                }
                augmented_conv_dicts.append(conv_dict)
            context_tokens.append(text_token_ids)
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)

        return augmented_conv_dicts