from torch import nn
from transformers import DebertaV2ForSequenceClassification, AutoModel


class DebertaReplacedTokenizer(DebertaV2ForSequenceClassification):
    def __init__(self, config, **kwargs):
        tok_model_name = kwargs.pop('tok_model_name')
        if 'num_labels' in kwargs:
            config.num_labels = kwargs.pop('num_labels')
        super().__init__(config, **kwargs)

        tok_model = AutoModel.from_pretrained(tok_model_name)
        new_emb = nn.Sequential(
                tok_model.get_input_embeddings(),
                nn.Linear(tok_model.config.hidden_size\
                        if 'opt' not in tok_model_name else tok_model.config.word_embed_proj_dim,
                    self.config.hidden_size)
                )
        self.set_input_embeddings(new_emb)
