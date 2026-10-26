import argparse
from collections import defaultdict
import copy
import datasets as hf_datasets
import einops
import os
import os.path as osp
import json
import re
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import Utils
from Utils import twrite
import UtilsBase

def get_system_prompt():
    prompt_base = f"Your task is rewriting headlines to be perceived with a new target sentiment by changing tone, adding stylistic flair, or adding related made-up details that support the target sentiment but do not contradict the original. Otherwise the rewritten headline closely resembles the original and includes the same factual information. You output only the text of the rewritten headline. Never state target sentiment directly."
    return prompt_base

def get_data_from_file(fname):
    """Returns the list of headlines within the dataset given by JSON file [fname]."""
    with open(fname, "r+") as f:
        headline2sentiment = json.load(f)
    return [dict(headline=h, sentiment=s) for h,s in headline2sentiment.items() if len([hh for hh in h.split() if len(hh)>4]) >= 8 and len(h.split()) <= 24]
    
class CollateFn:
    """Class implementing a collate function for DataLoader.

    Args:
    model_tokenizer     -- tokenizer for the policy model
    finbert_tokenizer    -- tokenizer for the reward model
    """
    def __init__(self, *,  model_tokenizer):
        self.model_tokenizer = model_tokenizer

    def __call__(self, batch):
        """Returns a collated version of [batch] for use in DataLoader. Any keys whose
        values are tensor-valued are collated with the appropriate tokenizer's
        DataCollatorWithPadding. This will turn a BS-length list of
        possibly-different-length tensors containing input_ids and attention_mask for
        prompts, headlines, and sentiments into properly padded BSxT tensors.

        Args:
        batch       -- list of examples, each of which is a dict with the keys
                        - prompt_input_ids
                        - prompt_attention_mask
                        as well as possibly other keys (eg. the original headline, sentiment, and prompt strings)
        """
        headlines = [hs["headline"] for hs in batch]
        sentiments = [hs["sentiment"] for hs in batch]
        prompts = [hs["prompt"] for hs in batch]

        batch_prompt = self.model_tokenizer(prompts, truncation=True, padding=True, return_tensors="pt")
        batch_prompt = {f"prompt_{k}": v for k,v in batch_prompt.items()}
        return batch_prompt | dict(sentiment=sentiments, headline=headlines, prompt=prompts) 

def get_data(*, data_path, model_tokenizer, args, write_data=False, train=True):
    """Returns training and validation DataLoaders given [args].
    
    Args:
    model_tokenizer     -- tokenizer for the policy model
    train               -- whether the returned data is for training or validation/testing
    args                -- Namespace with arguments
    """
    def write_data_to_file(data, fname):
        """Writes [data] to JSON file [fname]."""
        headline2sentiment = {d["headline"]: d["sentiment"] for d in data}
        with open(fname, "w+") as f:
            json.dump(headline2sentiment, f, indent=4)
        twrite(f"Wrote {len(data)} examples to {fname}")

    def dataset_to_sampler_loader(data, *, batch_size, shuffle=False, drop_last=False,
        seed=None, num_workers=8, persistent_workers=True,
        pin_memory=False, multiprocessing_context="fork", collate_fn=None):
        """Returns a (sampler, loader) tuple for [data]."""
        sampler = DistributedSampler(data, num_replicas=1, rank=0, seed=seed, shuffle=shuffle)

        persistent_workers = persistent_workers if num_workers > 0 else False
        return sampler, DataLoader(data, batch_size=batch_size,
            sampler=sampler, collate_fn=collate_fn, drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context if num_workers > 0 else None)

    def make_prompt(headline, sentiment):
        s = f"Target: {sentiment}. Headline: {headline}"
        # s = f"({sentiment}, {headline})"
        messages = [dict(role="system", content=get_system_prompt())]
        messages += [dict(role="user", content=s)]
        return model_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Load a dataset to build off of
    sentiments = ["negative", "positive"]
    if osp.exists(data_path):
        #  headlines_sentiments = get_data_from_file(fname=args.data_tr)
        headlines_sentiments = get_data_from_file(fname=data_path)
        headlines = [hs["headline"] for hs in headlines_sentiments]
    elif data_path == "financial_phrasebank75":
        headlines_sentiments = get_data_from_file(fname="financial_phrasebank75_train.json")
        headlines_sentiments = [hs for hs in headlines_sentiments if not any([c.isdigit() for c in hs["headline"]])]  # Remove any headlines with digits
        sentiment2headlines = defaultdict(list)
        for hs in headlines_sentiments:
            sentiment2headlines[hs["sentiment"]].append(hs)
        
        # Extract more normal-length headlines only, and make it class-balanced
        sentiment2headlines = {s: sorted(hs, key=lambda x: len(x["headline"])) for s,hs in sentiment2headlines.items()}
        headlines = [hs[idx]["headline"] for idx in range(min([len(hs) for hs in sentiment2headlines.values()])) for s,hs in sentiment2headlines.items()]
    else:
        raise NotImplementedError()
    
    # The actual dataset we care about is a list of prompts, where each prompt is a
    # (headline, sentiment, prompt) triple. The LM we're training needs just the
    # prompt, while the reward model needs to know the headline and sentiment as well.
    data = [dict(headline=h, sentiment=s) for h in headlines for s in sentiments]
    data = data[:args.data_max_ex] if args.data_max_ex else data
    data = [d | dict(prompt=make_prompt(d["headline"], d["sentiment"])) for d in data]    
    data = hf_datasets.Dataset.from_list(data)

    # twrite(f"Loaded dataset with {len(data)} examples from {data_path}")

    
    # data_size_minus_2x_max_ex = len(data) - (2 * args.data_max_ex)
    # splits = data.train_test_split(test_size=data_size_minus_2x_max_ex/len(data), seed=args.seed)
    # data_tr_val, data_te = splits["train"], splits["test"]
    # splits = data_tr_val.train_test_split(test_size=0.5, seed=args.seed)
    # data_tr, data_val = splits["train"], splits["test"]

    # data_tr = data_tr[:args.data_max_ex]
    # data_val = data_val[:args.data_max_ex]
    
    # data_tr, data_val = splits["train"], splits["test"]

    # data_trainval = []
    # for data_val_ex,data_tr_ex in zip(data_val, data_tr):
    #     data_trainval.append(data_tr_ex)
    #     data_trainval.append(data_val_ex)
    # data_trainval = hf_datasets.Dataset.from_list(data_trainval)


    # assert len(data_tr) >= args.bs, f"Dataset too small ({len(data_tr)} examples) for batch size {args.bs}!"
    # # data_tr, data_val = data, data

    # if write_data or True:
    #     write_data_to_file(data_tr, f"headline2sentiment_ex{len(data_tr)}_s{args.seed}_train.json")
    #     write_data_to_file(data_val, f"headline2sentiment_ex{len(data_val)}_s{args.seed}_val.json")
    #     write_data_to_file(data_te, f"headline2sentiment_ex{len(data_te)}_s{args.seed}_test.json")
    #     write_data_to_file(data_trainval, f"headline2sentiment_ex{len(data_trainval)}_s{args.seed}_trainval.json")

    # assert 0
        

    # Create the collate funtion. Essentially, it just needs to pad the input_ids and
    # attention_mask for headlines, sentiments, and prompts individually to have the
    # same sequence length within a batch.
    
    collate_fn = CollateFn(model_tokenizer=model_tokenizer)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"          # Apparently safe, though we should probably not be forking stuff...
    dataloader_kwargs = dict(num_workers=args.num_workers,  # Number of subprocesses to use for data loading. This is the first thing to tune for performance
        pin_memory=True,                                    # Probably slightly faster, consider turning off if training is unstable
        persistent_workers=True,                            # Keep workers alive between epochs. Slightly faster, but might be less stable
        multiprocessing_context=args.hw_mp_ctx,             # 'fork' is generally fastest but can be unstable especially when run for longer
        collate_fn=collate_fn,
        drop_last=train,
        shuffle=train)
    
    sampler, loader = dataset_to_sampler_loader(data, batch_size=args.bs,
        seed=args.seed, **dataloader_kwargs)
    return sampler, loader

def get_word_trigrams(s):
    """Returns the set of lowercased word trigrams in string [s]."""
    words = re.findall(r"\w+", s.lower())
    return set(zip(words, words[1:], words[2:]))

def is_valid_headline(s):
    """Returns whether [s] ends a sentence, ignoring trailing whitespace, quotes, and
    parentheses. Completions cut off by the token limit don't.
    """
    return 1 if s.rstrip(" \n\t'\"`)").endswith((".", "!", "?")) else 0.3

class SequenceRewardModel(nn.Module):
    """Neural network used to compute sequence-level rewards."""
    def __init__(self, args=None):
        super(SequenceRewardModel, self).__init__() 
        self.args = copy.deepcopy(args)
        self.device = Utils.get_device(self.args)

        # Setup FinBERT for sentiment analysis
        self.sentiment2idx = {"neutral": 0, "positive": 1, "negative": 2} # Sentiment-to-label map used by FinBERT
        self.finbert_tokenizer_kwargs = dict(padding=True, truncation=True, padding_side="right", return_tensors="pt")
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain",
            use_fast=True, model_max_length=512, **self.finbert_tokenizer_kwargs)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone",
            device_map=None if torch.cuda.is_available() else "cpu",)
        self.sentiment_model.eval()
        for param in self.sentiment_model.parameters():
            param.requires_grad = False

        # Setup the content embedding model
        self.content_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_fast=True, model_max_length=512)
        self.content_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.content_model.eval()
        for param in self.content_model.parameters():
            param.requires_grad = False
        embedded_system_prompt = self.content_model.encode(get_system_prompt(),
            convert_to_tensor=True, normalize_embeddings=True).to(self.device)
        embedded_system_prompt = einops.rearrange(embedded_system_prompt, "d -> 1 d")
        bad_strings = ["New headline", "Target sentiment", "Target positive",
            "Target negative", "Target neutral",]
        embedded_system_prompt2 = self.content_model.encode(bad_strings,
            convert_to_tensor=True, normalize_embeddings=True).to(self.device)
        self.embedded_system_prompt = torch.cat([embedded_system_prompt,
            embedded_system_prompt2], dim=0)

        # Setup the NER model used to find entities in headlines. Headlines come from a
        # fixed dataset, so we cache their entities to run NER on each only once.
        self.ner = pipeline("token-classification", model="dslim/distilbert-NER",
            aggregation_strategy="first", device=self.device)
        self.headline2entities = {}
        

    @torch.no_grad()
    def get_headline_entities(self, headlines):
        """Returns a list giving the set of lowercased entity words in each of
        [headlines], running NER only on headlines not seen before.
        """
        new_headlines = list({h for h in headlines if not h in self.headline2entities})
        if len(new_headlines) > 0:
            for h,entities in zip(new_headlines, self.ner(new_headlines, batch_size=32)):
                self.headline2entities[h] = {w for e in entities if e["score"] > 0.5
                    for w in re.findall(r"\w+", h[e["start"]:e["end"]].lower())}
        return [self.headline2entities[h] for h in headlines]

    @torch.no_grad()
    def forward_sentiment_logits(self, *, input_ids, attention_mask, **kwargs):
        """Returns the BSx3 tensor of sentiment logits for [input_ids] processed with
        [attention_mask].

        Args:
        input_ids       -- BSxT tensor of token IDs
        attention_mask  -- BSxT tensor giving the attention mask
        """
        outputs = self.sentiment_model(input_ids=input_ids, attention_mask=attention_mask,
            return_dict=True)
        return outputs.logits
        
    @torch.no_grad()
    def forward(self, *, completions, sentiments, headlines):
        """Returns a (BS C)-length vector of sequence-level rewards for the completions
        responses [completions], given the target [sentiments] and original [headlines].

        BS is the batch size, while C is the number of completions candidates for each
        of the BS prompts. Alternatively, if we've already 'expanded' the headlines
        to be of length (BS C), then C=1 effectively and this will work anyways.

        Args:
        completions   -- (BS C)-length list of decoded candidate headline strings. Each
                        group of [C]  consecutive entries correspond to the [C]
                        completions for a single prompt in the batch (and thus have
                        the same target sentiment and original headline).
        sentiments  -- BS- or (BS C)-length list of target sentiments
        headlines   -- BS- or (BS C)-length list of original headlines
        """
        assert len(sentiments) == len(headlines), "Sentiments and headlines must have the same batch size!"
        bs_c = len(completions)
        bs = len(sentiments)
        c = bs_c // bs
        # assert c == 1 or (c % self.args.cbs == 0), f"Got c={c} cbs={self.args.cbs} bs_c={bs_c} bs={bs}"

        # Indices of completions with target sentiments.
        # S = ["negative", "neutral", "positive"]
        # S += [s.upper() for s in S]


        # Get embeddings for [completions] and [headlines] from the content model. These
        # are used to compute their similarity.
        completions_embeds = self.content_model.encode(completions, convert_to_tensor=True, normalize_embeddings=True)
        completions_embeds = einops.rearrange(completions_embeds, "(bs c) d -> bs c d", bs=bs, c=c)
        headline_embeds = self.content_model.encode(headlines, convert_to_tensor=True, normalize_embeddings=True)
        content_sims = einops.einsum(completions_embeds, headline_embeds, "bs c d, bs d -> bs c") # Dot-product over last d-dimension
        content_sims = einops.rearrange(content_sims, "bs c -> (bs c)")
        content_reward = torch.exp(content_sims) - 0.9

        # Compute the similarity between [completions] and the system prompt to penalize
        bad_content_sims = einops.einsum(completions_embeds, self.embedded_system_prompt, "bs c d, b d -> bs c b")
        bad_content_sims = einops.reduce(bad_content_sims, "bs c b -> bs c", "sum")
        bad_content_sims = einops.rearrange(bad_content_sims, "bs c -> (bs c)")
        bad_content_reward = torch.exp(bad_content_sims) - 0.9

        # Get the sentiment predictions for [completions]
        answers_sentiment_ids = self.finbert_tokenizer(completions, **self.finbert_tokenizer_kwargs).to(self.device)
        sentiment_logits = self.forward_sentiment_logits(**answers_sentiment_ids)


        # Repeat-interleave [sentiments] to have shape (BS C) to extract the target
        # sentiments for each element of [completions]
        sentiment_idxs = torch.tensor([self.sentiment2idx[s] for s in sentiments], device=self.device)
        sentiment_idxs = einops.repeat(sentiment_idxs, "bs -> (bs c)", c=c)
        sentiment_pred = nn.functional.softmax(sentiment_logits, dim=-1)
        sentiment_pred = torch.gather(sentiment_pred, 1, einops.rearrange(sentiment_idxs, "bsc -> bsc 1"))
        sentiment_pred = einops.rearrange(sentiment_pred, "bsc 1 -> bsc")
        sentiment_reward = torch.exp(sentiment_pred) - 0.9

        # For each index in sentiment_idxs, check whether the corresponding completion
        # contains the target sentiment at all. We need a binary (BS C)-length mask
        # for this.
        sentiment_ids = torch.tensor([1527, 483, 591], device=self.device)  # neutral, positive, negative
        contains_pos = (answers_sentiment_ids.input_ids.int() == 483)
        contains_neg = (answers_sentiment_ids.input_ids.int() == 591)
        contains_neu = (answers_sentiment_ids.input_ids.int() == 1527)
        contains_sentiment_id = (contains_pos | contains_neg | contains_neu).sum(dim=-1)
        num_tokens_in_completion = answers_sentiment_ids.attention_mask.sum(dim=-1)
        num_tokens_in_completion = torch.clamp(torch.log(num_tokens_in_completion.float()), min=1)
        contains_sentiment_id = contains_sentiment_id.float() / num_tokens_in_completion

        # Ensure responses remain diverse by penalizing completions that copy phrases
        # from the headline. [trigram_overlap] is the fraction of each completion's
        # word trigrams that also appear in its headline. Reordering words breaks
        # trigrams, so scrambled headlines aren't penalized, but inserting a word
        # into a copied headline only breaks a few of them.
        headline_trigrams = [get_word_trigrams(h) for h in headlines]
        trigram_overlap = []
        for idx,completion in enumerate(completions):
            completion_trigrams = get_word_trigrams(completion)
            shared = completion_trigrams & headline_trigrams[idx // c]
            trigram_overlap.append(len(shared) / max(len(completion_trigrams), 1))
        trigram_overlap = torch.tensor(trigram_overlap, device=self.device)

        # Reward completions for keeping the headline's entities (people, companies,
        # places, etc.), which carry most of its content regardless of sentiment.
        # [entity_recall] is the fraction of the headline's entity words that appear
        # in the completion, or one if the headline has none.
        headline_entities = self.get_headline_entities(headlines)
        entity_recall = []
        for idx,completion in enumerate(completions):
            entities = headline_entities[idx // c]
            kept = entities & set(re.findall(r"\w+", completion.lower()))
            entity_recall.append(len(kept) / len(entities) if len(entities) > 0 else 1)
        entity_recall = torch.tensor(entity_recall, device=self.device)

        # Penalize completions that don't end a sentence, which happens when they hit
        # the --grpo_max_new_tokens limit and are cut off.
        is_valid = torch.tensor([is_valid_headline(c) for c in completions],
            device=self.device).float()

        # Per-sequence reward is product of content and sentiment rewards, scaled down
        # by the trigram overlap with the headline and the entity recall, and scaled
        # by 0.3 if the completion isn't a valid headline, minus the penalty for bad content
        # similar to the system prompt wherever it's positive
        total_reward = content_reward * sentiment_reward * (1 - trigram_overlap) * entity_recall * is_valid - torch.max(torch.zeros_like(bad_content_reward), bad_content_reward) - contains_sentiment_id
        return total_reward

