import types
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer, LogitsProcessor, LogitsProcessorList
from functools import partial
from undecorate import unwrap
from types import MethodType
from utils import *
from ling_disc import DebertaReplacedTokenizer
from const import *
from lingconv_t5 import LingConvT5ForConditionalGeneration
from dataclasses import dataclass
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Dict, Any



def vae_sample(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(args.input_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                )
        self.decoder = nn.Sequential(
                nn.Linear(args.latent_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.input_dim),
                )
        self.fc_mu = nn.Linear(args.hidden_dim, args.latent_dim)
        self.fc_var = nn.Linear(args.hidden_dim, args.latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        x = vae_sample(mu, logvar)
        o = self.decoder(x)
        return o, (mu, logvar)

class LingGenerator(nn.Module):
    def __init__(self, args, hidden_dim=1000):
        super().__init__()

        self.gen = T5EncoderModel.from_pretrained('google/flan-t5-small')
        self.hidden_size = self.gen.config.d_model
        self.ling_embed = nn.Linear(args.lng_dim, self.hidden_size)
        # self.gen = nn.Sequential(
        #         nn.Linear(args.lng_dim, 2*hidden_dim),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(2*hidden_dim),
        #         nn.Linear(2*hidden_dim, 2*hidden_dim),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(2*hidden_dim),
        #         nn.Linear(2*hidden_dim, hidden_dim),
        #         nn.ReLU(),
        #         )

        self.gen_type = args.linggen_type
        self.gen_input = args.linggen_input
        if self.gen_type == 'vae':
            self.gen_mu = nn.Linear(hidden_dim, args.lng_dim)
            self.gen_logvar = nn.Linear(hidden_dim, args.lng_dim)
        elif self.gen_type == 'det':
            self.projection = nn.Linear(self.hidden_size, args.lng_dim)

    def forward(self, batch):
        inputs_embeds = self.gen.shared(batch['sentence1_input_ids'])
        inputs_att_mask = batch['sentence1_attention_mask']
        bs = inputs_embeds.shape[0]

        if self.gen_input == 's+l':
            sentence1_ling = self.ling_embed(batch['sentence1_ling'])
            sentence1_ling = sentence1_ling.view(bs, 1, -1)
            inputs_embeds = inputs_embeds + sentence1_ling

        gen = self.gen(inputs_embeds=inputs_embeds,
                attention_mask=inputs_att_mask).last_hidden_state.mean(1)
        # gen = self.gen(batch['sentence1_ling'])

        cache = {}
        if self.gen_type == 'vae':
            mu = self.gen_mu(gen)
            logvar = self.gen_logvar(gen)
            output = vae_sample(mu, logvar)
            cache['linggen_mu'] = mu
            cache['linggen_logvar'] = logvar
        elif self.gen_type == 'det':
            output = self.projection(gen)

        return output, cache


class LingDisc(nn.Module):
    def __init__(self,
                 model_name,
                 disc_type,
                 disc_ckpt,
                 lng_dim=40,
                 quant_nbins=1,
                 disc_lng_dim=None,
                 lng_ids=None,
                 **kwargs):
        super().__init__()
        if disc_type == 't5':
            self.encoder = T5EncoderModel.from_pretrained(model_name)
            hidden_dim = self.encoder.config.d_model
            self.dropout = nn.Dropout(0.2)
            self.lng_dim = disc_lng_dim if disc_lng_dim else lng_dim
            self.quant = quant_nbins > 1
            self.quant = False
            if self.quant:
                self.ling_classifier = nn.Linear(hidden_dim, self.lng_dim * quant_nbins)
            else:
                self.ling_classifier = nn.Linear(hidden_dim, self.lng_dim)
            lng_ids = torch.tensor(lng_ids) if lng_ids is not None else None
            # from const import used_indices
            # lng_ids = torch.tensor(used_indices)
            self.register_buffer('lng_ids', lng_ids)
        elif disc_type == 'deberta':
            self.encoder= DebertaReplacedTokenizer.from_pretrained(
                    pretrained_model_name_or_path=disc_ckpt,
                    tok_model_name = model_name,
                    problem_type='regression', num_labels=40)
            self.quant = False

        self.disc_type = disc_type

    def forward(self, **batch):
        if not 'attention_mask' in batch:
            if 'input_ids' in batch:
                att_mask = torch.ones_like(batch['input_ids'])
            else:
                att_mask = torch.ones_like(batch['logits'])[:,:,0]
        else:
            att_mask = batch['attention_mask']
        if 'input_ids' in batch:
            enc_output = self.encoder(input_ids=batch['input_ids'],
                    attention_mask=att_mask)
        elif 'logits' in batch:
            logits = batch['logits']
            scores = F.softmax(logits, dim = -1)
            onehot = F.one_hot(logits.argmax(-1), num_classes=logits.shape[2]).float().to(logits.device)
            onehot_ = scores - scores.detach() + onehot

            embed_layer = self.encoder.get_input_embeddings()
            if isinstance(embed_layer, nn.Sequential):
                for i, module in enumerate(embed_layer):
                    if i == 0:
                        embeds = torch.matmul(onehot_, module.weight)
                    else:
                        embeds = module(embeds)
            else:
                embeds =  onehot_ @ embed_layer.weight
                embeds = torch.matmul(onehot_, embed_layer.weight)

            enc_output = self.encoder(inputs_embeds=embeds,
                    attention_mask=att_mask)
        if self.disc_type == 't5':
            sent_emb = self.dropout(enc_output.last_hidden_state.mean(1))
            bs = sent_emb.shape[0]
            output = self.ling_classifier(sent_emb)
            if self.quant:
                output = output.reshape(bs, -1, self.lng_dim)
            if self.lng_ids is not None:
                output = torch.index_select(output, 1, self.lng_ids)
        elif self.disc_type == 'deberta':
            output = enc_output.logits
        return output

class SemEmb(T5EncoderModel):
    def __init__(self, config, sep_token_id):
        super().__init__(config)
        self.sep_token_id = sep_token_id
        hidden_dim = self.config.d_model
        self.projection = nn.Sequential(nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1))

    def compare_sem(self, **batch):
        bs = batch['sentence1_attention_mask'].shape[0]
        ones = torch.ones((bs, 1), device=batch['sentence1_attention_mask'].device)
        sep = torch.ones((bs, 1), dtype=torch.long,
                device=batch['sentence1_attention_mask'].device) * self.sep_token_id
        att_mask = torch.cat([batch['sentence1_attention_mask'], ones, batch['sentence2_attention_mask']], dim=1)
        if 'logits' in batch:
            input_ids = torch.cat([batch['sentence1_input_ids'], sep], dim=1)
            embeds1 = self.shared(input_ids)

            logits = batch['logits']
            scores = F.softmax(logits, dim = -1)
            onehot = F.one_hot(logits.argmax(-1), num_classes=logits.shape[2]).float().to(logits.device)
            onehot_ = scores - scores.detach() + onehot

            embeds2 =  onehot_ @ self.shared.weight
            embeds1_2 = torch.cat([embeds1, embeds2], dim=1)
            hidden_units = super().forward(inputs_embeds=embeds1_2,
                    attention_mask=att_mask).last_hidden_state.mean(1)
        elif 'sentence2_input_ids' in batch:
            input_ids = torch.cat([batch['sentence1_input_ids'], sep, batch['sentence2_input_ids']], dim=1)
            hidden_units = super().forward(input_ids=input_ids,
                    attention_mask=att_mask).last_hidden_state.mean(1)
        probs = self.projection(hidden_units)
        return probs

def prepare_inputs_for_generation(
        combine_method,
        ling2_only,
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        sentence1_ling=None,
        sentence2_ling=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        input_ids = input_ids.clone()
        decoder_inputs_embeds = self.shared(input_ids)

        if combine_method == 'layer_injection':
            # For layer injection, we'll pass the ling embeddings separately
            ling_embed = sentence2_ling if ling2_only else (sentence1_ling + sentence2_ling)
        elif combine_method == 'decoder_add_first':
            sentence2_ling = torch.cat([sentence2_ling,
                torch.repeat_interleave(torch.zeros_like(sentence2_ling), input_ids.shape[1] - 1, dim=1)], dim = 1)
        elif combine_method == 'decoder_concat':
            if ling2_only:
                decoder_inputs_embeds = torch.cat([sentence2_ling, decoder_inputs_embeds], dim=1)
            else:
                decoder_inputs_embeds = torch.cat([sentence1_ling, sentence2_ling, decoder_inputs_embeds], dim=1)

        is_first_step = past_key_values is None or len(past_key_values) == 0
        if combine_method == 'decoder_add' or (is_first_step and combine_method == 'decoder_add_first'):
            if ling2_only:
                decoder_inputs_embeds = decoder_inputs_embeds + sentence2_ling
            else:
                decoder_inputs_embeds = decoder_inputs_embeds + sentence1_ling + sentence2_ling

        return {
            "decoder_inputs_embeds": decoder_inputs_embeds,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "ling_embed": ling_embed if combine_method == 'layer_injection' else None,
        }

class LogitsAdd(LogitsProcessor):
    def __init__(self, sentence2_ling):
        super().__init__()
        self.sentence2_ling = sentence2_ling

    def __call__(self, input_ids, scores):
        return scores + self.sentence2_ling

class EncoderDecoderVAE(LingConvT5ForConditionalGeneration):
    def __init__(self, config, args, pad_token_id, sepeos_token_id, vocab_size = 32128):
        if args.combine_method == 'layer_injection':
            if args.injection_layer < 0 or args.injection_layer >= config.num_decoder_layers:
                raise ValueError(f"Invalid injection layer: {args.injection_layer}. Must be between 0 and {config.num_decoder_layers - 1}.")
            config.ling_injection_layer = args.injection_layer
            config.ling_injection_type = args.injection_type  # 'first' or 'all'
            
        super().__init__(config)
        
        self.prepare_inputs_for_generation = types.MethodType(
                partial(prepare_inputs_for_generation, args.combine_method, args.ling2_only),
                self)
        self.args = args
        self.pad_token_id = pad_token_id
        self.eos_token_id = sepeos_token_id
        hidden_dim = self.config.d_model if not 'logits' in args.combine_method else vocab_size
        if args.combine_method == 'fusion1':
            self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim + 2 * args.lng_dim, hidden_dim),
                    )
        elif args.combine_method == 'fusion2':
            self.fusion = nn.Sequential(
                    nn.Linear(hidden_dim + 2 * args.lng_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    )
        elif 'concat' in args.combine_method or 'add' in args.combine_method or 'layer_injection' in args.combine_method:
            if args.ling_embed_type == 'two-layer':
                self.ling_embed = nn.Sequential(
                        nn.Linear(args.lng_dim, args.lng_dim),
                        nn.ReLU(),
                        nn.Linear(args.lng_dim, hidden_dim),
                        )
            else:
                self.ling_embed = nn.Linear(args.lng_dim, hidden_dim)
            self.ling_dropout = nn.Dropout(args.ling_dropout)
        self.ling_embed.apply(self._init_weights)

        if args.ling_vae:
            self.ling_mu = nn.Linear(hidden_dim, hidden_dim)
            self.ling_logvar = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(self.ling_embed.weight)
            nn.init.xavier_uniform_(self.ling_mu.weight)
            nn.init.xavier_uniform_(self.ling_logvar.weight)


        generate_with_grad = unwrap(super().generate)
        self.generate_with_grad = MethodType(generate_with_grad, self)
        self.generate_original = super().generate

    def _init_weights(self, module):
        std = self.args.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def _normalize_batch_dict(batch):
        normalized = dict(batch)
        sentence1_input_ids = normalized.pop("sentence1_input_ids", None)
        sentence1_attention_mask = normalized.pop("sentence1_attention_mask", None)
        sentence2_input_ids = normalized.pop("sentence2_input_ids", None)
        sentence2_attention_mask = normalized.pop("sentence2_attention_mask", None)

        if sentence1_input_ids is not None and "input_ids" not in normalized:
            normalized["input_ids"] = sentence1_input_ids
        if sentence1_attention_mask is not None and "attention_mask" not in normalized:
            normalized["attention_mask"] = sentence1_attention_mask
        if sentence2_input_ids is not None and "labels" not in normalized:
            normalized["labels"] = sentence2_input_ids
        if sentence2_attention_mask is not None and "decoder_attention_mask" not in normalized:
            normalized["decoder_attention_mask"] = sentence2_attention_mask
        return normalized

    def get_fusion_layer(self):
        if 'fusion' in self.args.combine_method:
            return self.fusion
        elif 'concat' in self.args.combine_method or 'add' in self.args.combine_method:
            return self.ling_embed
        else:
            return None

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _process_ling_embeddings(self, sentence1_ling, sentence2_ling, 
                               sentence1_ling_embed, sentence2_ling_embed, bs):
        """Helper method to process linguistic embeddings"""
        cache = {}
        
        # Process sentence1 embedding
        if sentence1_ling_embed is not None:
            sentence1_ling = sentence1_ling_embed
        elif sentence1_ling is not None:
            sentence1_ling = self.ling_embed(self.ling_dropout(sentence1_ling))
        else:
            sentence1_ling = None
            
        # Process sentence2 embedding
        if sentence2_ling_embed is not None:
            sentence2_ling = sentence2_ling_embed
        elif sentence2_ling is not None:
            sentence2_ling = self.ling_embed(self.ling_dropout(sentence2_ling))
        else:
            sentence2_ling = None

        # Apply VAE if configured
        if self.args.ling_vae and sentence1_ling is not None and sentence2_ling is not None:
            sentence1_ling = F.leaky_relu(sentence1_ling)
            sent1_mu, sent1_logvar = self.ling_mu(sentence1_ling), self.ling_logvar(sentence1_ling)
            sentence1_ling = self.sample(sent1_mu, sent1_logvar)

            sentence2_ling = F.leaky_relu(sentence2_ling)
            sent2_mu, sent2_logvar = self.ling_mu(sentence2_ling), self.ling_logvar(sentence2_ling)
            sentence2_ling = self.sample(sent2_mu, sent2_logvar)
            
            cache.update({
                'sent1_mu': sent1_mu, 'sent1_logvar': sent1_logvar,
                'sent2_mu': sent2_mu, 'sent2_logvar': sent2_logvar,
                'sentence1_ling': sentence1_ling, 'sentence2_ling': sentence2_ling
            })
        else:
            if sentence2_ling is not None:
                cache['sentence2_ling'] = sentence2_ling
            if sentence1_ling is not None:
                cache['sentence1_ling'] = sentence1_ling

        # Reshape embeddings
        if sentence1_ling is not None:
            sentence1_ling = sentence1_ling.view(bs, 1, -1)
        if sentence2_ling is not None:
            sentence2_ling = sentence2_ling.view(bs, 1, -1)
            
        return sentence1_ling, sentence2_ling, cache

    def encode(self, 
               input_ids=None,
               attention_mask=None,
               sentence1_ling=None,
               sentence2_ling=None,
               sentence1_ling_embed=None,
               sentence2_ling_embed=None,
               inputs_embeds=None,
               ):
        if inputs_embeds is None:
            inputs_embeds = self.shared(input_ids)
        inputs_att_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        bs = inputs_embeds.shape[0]
        
        if self.args.combine_method in ('input_concat', 'input_add'):
            sentence1_ling, sentence2_ling, cache = self._process_ling_embeddings(
                sentence1_ling, sentence2_ling,
                sentence1_ling_embed, sentence2_ling_embed, bs
            )
            
            if self.args.combine_method == 'input_concat':
                if self.args.ling2_only:
                    inputs_embeds = torch.cat([inputs_embeds, sentence2_ling], dim=1)
                    inputs_att_mask = torch.cat([inputs_att_mask,
                        torch.ones((bs, 1)).to(inputs_embeds.device)], dim=1)
                else:
                    inputs_embeds = torch.cat([inputs_embeds, sentence1_ling, sentence2_ling], dim=1)
                    inputs_att_mask = torch.cat([inputs_att_mask,
                        torch.ones((bs, 2)).to(inputs_embeds.device)], dim=1)
            elif self.args.combine_method == 'input_add':
                if self.args.ling2_only:
                    inputs_embeds = inputs_embeds + sentence2_ling
                else:
                    inputs_embeds = inputs_embeds + sentence1_ling + sentence2_ling
        else:
            cache = {}

        return self.encoder(inputs_embeds=inputs_embeds,
                attention_mask=inputs_att_mask), inputs_att_mask, cache

    def decode(self,
              sentence2_input_ids=None,
              sentence1_ling=None,
              sentence2_ling=None,
              encoder_outputs=None,
              encoder_attention_mask=None,
              decoder_inputs_embeds=None,
              decoder_attention_mask=None,
              generate=False,
              sentence1_ling_embed=None,
              sentence2_ling_embed=None,
              ling_embed=None,
              generate_with_grad=False,
              **kwargs
              ):
        bs = encoder_outputs[0].shape[0]
        cache = {}
        
        if decoder_inputs_embeds is None:
            if self.args.combine_method in ('embed_concat', 'decoder_concat', 'decoder_add', 
                                        'logits_add', 'decoder_add_first', 'layer_injection'):
                sentence1_ling, sentence2_ling, cache = self._process_ling_embeddings(
                    sentence1_ling, sentence2_ling,
                    sentence1_ling_embed, sentence2_ling_embed, bs
                )
                
                if (self.args.combine_method == 'decoder_add_first' or 
                    (self.args.combine_method == 'layer_injection' and 
                    self.args.injection_type == 'first')) and not generate:
                    sentence2_ling = torch.cat([sentence2_ling,
                        torch.repeat_interleave(torch.zeros_like(sentence2_ling), 
                        sentence2_input_ids.shape[1] - 1, dim=1)], dim = 1)
            else:
                sentence1_ling, sentence2_ling = None, None
        
        if generate:
            if self.args.combine_method == 'logits_add':
                logits_processor = LogitsProcessorList([LogitsAdd(sentence2_ling.view(bs, -1))])
            else:
                logits_processor = LogitsProcessorList()

            generate_fn = self.generate_with_grad if generate_with_grad else self.generate_original
            dec_output = generate_fn(
                    attention_mask=encoder_attention_mask,
                    encoder_outputs=encoder_outputs,
                    sentence1_ling=sentence1_ling,
                    sentence2_ling=sentence2_ling,
                    logits_processor = logits_processor,
                    # renormalize_logits=True,
                    # do_sample=True,
                    # top_p=0.8,
                    eos_token_id=self.eos_token_id,
                    # min_new_tokens=3,
                    # repetition_penalty=1.2,
                    max_length=self.args.max_length,
                    )
            # scores = torch.stack(dec_output.scores, 1)
            # cache.update({'scores': scores})
            return dec_output, cache

        if sentence2_input_ids is not None:
            labels = sentence2_input_ids.clone()
            labels[labels == self.pad_token_id] = -100
        else:
            labels = None

        if decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(sentence2_input_ids)
            decoder_inputs_embeds = self.shared(decoder_input_ids)

            if self.args.combine_method == 'decoder_concat':
                if self.args.ling2_only:
                    decoder_inputs_embeds = torch.cat([sentence2_ling, decoder_inputs_embeds], dim=1)
                    decoder_attention_mask = torch.cat([torch.ones((bs, 1)).to(decoder_inputs_embeds.device), decoder_attention_mask], dim=1)
                    labels = torch.cat([torch.ones((bs, 1), dtype=torch.int64).to(decoder_inputs_embeds.device) * self.pad_token_id,
                        labels], dim=1)
                else:
                    decoder_inputs_embeds = torch.cat([sentence1_ling, sentence2_ling, decoder_inputs_embeds], dim=1)
                    decoder_attention_mask = torch.cat([torch.ones((bs, 2)).to(decoder_inputs_embeds.device), decoder_attention_mask], dim=1)
                    labels = torch.cat([torch.ones((bs, 2), dtype=torch.int64).to(decoder_inputs_embeds.device) * self.pad_token_id,
                        labels], dim=1)
            elif self.args.combine_method == 'decoder_add' or self.args.combine_method == 'decoder_add_first' :
                if self.args.ling2_only:
                    decoder_inputs_embeds = decoder_inputs_embeds + self.args.combine_weight * sentence2_ling
                else:
                    decoder_inputs_embeds = decoder_inputs_embeds + sentence1_ling + sentence2_ling

        if ling_embed is None:
            ling_embed = sentence2_ling

        dec_output = super().forward(
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                labels=labels,
                ling_embed=ling_embed,
                **kwargs
                )
        if self.args.combine_method == 'logits_add':
            dec_output.logits = dec_output.logits + self.args.combine_weight * sentence2_ling
            vocab_size = dec_output.logits.size(-1)
            dec_output.loss = F.cross_entropy(dec_output.logits.view(-1, vocab_size), labels.view(-1))
        return dec_output, cache

    def generate(self, *args, **kwargs):
        return self.forward(*args, **kwargs, generate=True)


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                decoder_attention_mask=None,
                decoder_inputs_embeds=None,
                sentence1_ling=None,
                sentence2_ling=None,
                sentence1_ling_embed=None,
                sentence2_ling_embed=None,
                inputs_embeds=None,
                generate=False,
                encoder_outputs=None,
                encoder_attention_mask=None,
                ling_embed=None,
                generate_with_grad=False,
                **kwargs):
        kwargs = self._normalize_batch_dict(kwargs)
        if input_ids is None:
            input_ids = kwargs.pop("input_ids", None)
        if attention_mask is None:
            attention_mask = kwargs.pop("attention_mask", None)
        if labels is None:
            labels = kwargs.pop("labels", None)
        if decoder_attention_mask is None:
            decoder_attention_mask = kwargs.pop("decoder_attention_mask", None)
        if sentence1_ling is None:
            sentence1_ling = kwargs.pop("sentence1_ling", None)
        if sentence2_ling is None:
            sentence2_ling = kwargs.pop("sentence2_ling", None)
        if inputs_embeds is None:
            inputs_embeds = kwargs.pop("inputs_embeds", None)

        cache = {}
        if encoder_outputs is None:
            encoder_outputs, encoder_attention_mask, cache = self.encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence1_ling=sentence1_ling,
                sentence2_ling=sentence2_ling,
                sentence1_ling_embed=sentence1_ling_embed,
                sentence2_ling_embed=sentence2_ling_embed,
                inputs_embeds=inputs_embeds
            )
        
        dec_output, cache2 = self.decode(
            sentence2_input_ids=labels,
            sentence1_ling=sentence1_ling,
            sentence2_ling=sentence2_ling,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            generate=generate,
            sentence1_ling_embed=sentence1_ling_embed,
            sentence2_ling_embed=sentence2_ling_embed,
            ling_embed=ling_embed,
            generate_with_grad=generate_with_grad,
            **kwargs
        )
        
        cache.update(cache2)
        if generate:
            return dec_output
        else:
            return MySeq2SeqLMOutput(
                loss=dec_output.loss,
                logits=dec_output.logits,
                past_key_values=dec_output.past_key_values,
                decoder_hidden_states=dec_output.decoder_hidden_states,
                decoder_attentions=dec_output.decoder_attentions,
                cross_attentions=dec_output.cross_attentions,
                encoder_last_hidden_state=encoder_outputs[0],
                encoder_hidden_states=getattr(encoder_outputs, 'hidden_states', None),
                encoder_attentions=getattr(encoder_outputs, 'attentions', None),
                cache=cache
                )

    def infer_with_cache(self, batch):
        batch = self._normalize_batch_dict(batch)
        encoder_outputs, encoder_attention_mask, cache = self.encode(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            sentence1_ling=batch.get("sentence1_ling"),
            sentence2_ling=batch.get("sentence2_ling"),
            sentence1_ling_embed=batch.get("sentence1_ling_embed"),
            sentence2_ling_embed=batch.get("sentence2_ling_embed"),
            inputs_embeds=batch.get("inputs_embeds"),
        )
        dec_output, cache2 = self.decode(
            sentence2_input_ids=batch.get("labels"),
            sentence1_ling=batch.get("sentence1_ling"),
            sentence2_ling=batch.get("sentence2_ling"),
            encoder_outputs=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask,
            decoder_inputs_embeds=batch.get("decoder_inputs_embeds"),
            decoder_attention_mask=batch.get("decoder_attention_mask"),
            generate=True,
            sentence1_ling_embed=batch.get("sentence1_ling_embed"),
            sentence2_ling_embed=batch.get("sentence2_ling_embed"),
            ling_embed=batch.get("ling_embed"),
            generate_with_grad=batch.get("generate_with_grad", False),
        )
        cache.update(cache2)
        return dec_output, cache

    def infer(self, batch):
        dec_output, _ = self.infer_with_cache(batch)
        return dec_output

    def infer_with_feedback_BP(self, ling_disc, sem_emb, batch, tokenizer):
        from torch.autograd import grad
        interpolations = []
        def line_search():
            best_val = None
            best_loss = None
            eta = 1e3
            sem_prob = 1
            patience = 4
            while patience > 0:
                param_ = param - eta * grads
                with torch.no_grad():
                    new_loss, pred = get_loss(param_)
                max_len = pred.shape[1]
                lens = torch.where(pred == self.eos_token_id, 1, 0).argmax(-1) + 1
                batch.update({
                    'sentence2_input_ids': pred,
                    'sentence2_attention_mask': sequence_mask(lens, max_len = max_len)
                    })
                sem_prob = torch.sigmoid(sem_emb.compare_sem(**batch)).item()
                # if sem_prob <= 0.1:
                #     patience -= 1
                if new_loss < loss and sem_prob >= 0.90 and lens.item() > 1:
                    return param_
                eta *= 2.25
                patience -= 1
            return False

        def get_loss(param):
            if self.args.feedback_param == 'l':
                batch.update({'sentence2_ling_embed': param})
            elif self.args.feedback_param == 's':
                batch.update({'inputs_embeds': param})

            if self.args.feedback_param == 'logits':
                logits = param
                pred = param.argmax(-1)
            else:
                pred, cache = self.infer_with_cache(batch)
                logits = cache['scores']
            out = ling_disc(logits = logits)
            probs = F.softmax(out, 1)
            if ling_disc.quant:
                loss = F.cross_entropy(out, batch['sentence2_discr'])
            else:
                loss = F.mse_loss(out, batch['sentence2_ling'])
            return loss, pred

        if self.args.feedback_param == 'l':
            ling2_embed = self.ling_embed(batch['sentence2_ling'])
            param = torch.nn.Parameter(ling2_embed, requires_grad = True)
        elif self.args.feedback_param == 's':
            inputs_embeds = self.shared(batch['sentence1_input_ids'])
            param = torch.nn.Parameter(inputs_embeds, requires_grad = True)
        elif self.args.feedback_param == 'logits':
            logits = self.infer_with_cache(batch)[1]['scores']
            param = torch.nn.Parameter(logits, requires_grad = True)
        target_np = batch['sentence2_ling'][0].cpu().numpy()
        while True:
            loss, pred = get_loss(param)
            pred_text = tokenizer.batch_decode(pred.cpu().numpy(),
                    skip_special_tokens=True)[0]
            interpolations.append(pred_text)
            if loss < 1:
                break
            self.zero_grad()
            grads = grad(loss, param)[0]
            param = line_search()
            if param is False:
                break
        return pred, [pred_text, interpolations]

def set_grad(module, state):
    if module is not None:
        for p in module.parameters():
            p.requires_grad = state

def set_grad_except(model, name, state):
    for n, p in model.named_parameters():
        if not name in n:
            p.requires_grad = state

class SemEmbPipeline():
    def __init__(self,
            ckpt = "./checkpoints/ling_conversion_sem_emb_best.pt"):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.model = SemEmb(T5EncoderModel.from_pretrained('google/flan-t5-base'), self.tokenizer.get_vocab()['</s>'])
        state = torch.load(ckpt)
        self.model.load_state_dict(state['model'], strict=False)
        self.model.eval()
        self.model.cuda()

    def __call__(self, sentence1, sentence2):
        sentence1 = self.tokenizer(sentence1, return_attention_mask = True, return_tensors = 'pt')
        sentence2 = self.tokenizer(sentence2, return_attention_mask = True, return_tensors = 'pt')
        sem_logit = self.model(
                sentence1_input_ids = sentence1.input_ids.cuda(),
                sentence1_attention_mask = sentence1.attention_mask.cuda(),
                sentence2_input_ids = sentence2.input_ids.cuda(),
                sentence2_attention_mask = sentence2.attention_mask.cuda(),
                )
        sem_prob = torch.sigmoid(sem_logit).item()
        return sem_prob

class LingDiscPipeline():
    def __init__(self,
                 model_name="google/flan-t5-base",
                 disc_type='deberta',
                 disc_ckpt=None,
                 ):
        if disc_ckpt is None:
            raise ValueError("disc_ckpt is required for LingDiscPipeline.")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = LingDisc(model_name, disc_type, disc_ckpt)
        self.model.eval()
        self.model.cuda()

    def __call__(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors = 'pt')
        with torch.no_grad():
            ling_pred = self.model(input_ids=inputs.input_ids.cuda())
        return ling_pred

def get_model(args, tokenizer, device):
    if args.pretrain_disc or args.disc_loss or args.disc_ckpt:
        ling_disc = LingDisc(args.model_name, args.disc_type, args.disc_ckpt).to(device)
    else:
        ling_disc = None

    if args.ckpt:
        model = EncoderDecoderVAE.from_pretrained(args.ckpt, args, tokenizer.pad_token_id, tokenizer.eos_token_id).to(device)
    else:
        model = EncoderDecoderVAE.from_pretrained(args.model_name, args, tokenizer.pad_token_id, tokenizer.eos_token_id).to(device)

    if args.sem_loss or args.sem_ckpt:
        if args.sem_loss_type == 'shared':
            sem_emb = model.encoder
        elif args.sem_loss_type == 'dedicated':
            sem_emb = SemEmb.from_pretrained(args.sem_model_path, tokenizer.eos_token_id).to(device)
        else:
            raise NotImplementedError('Semantic loss type')
    else:
        sem_emb = None

    return model, ling_disc, sem_emb

@dataclass
class MySeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Extends Seq2SeqLMOutput to include a cache dictionary for additional model outputs.
    
    Args:
        cache (`Dict[str, Any]`):
            Dictionary containing additional model outputs like linguistic features,
            VAE parameters, scores, etc.
    """
    cache: Optional[Dict[str, Any]] = None
