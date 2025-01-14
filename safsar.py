import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoImageProcessor, AutoModelForVideoClassification, BertTokenizer, BertModel
import numpy as np
from torchvision import datasets, transforms

class SAFSAR(nn.Module):
    def __init__(self, config):
        super(SAFSAR, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(config["processor_name"])
        self.model = AutoModelForVideoClassification.from_pretrained(config["model_name"], output_hidden_states=True)
        self.way = config["way"]
        self.shot = config["shot"]
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]
        self.query_per_class_test = config["query_per_class_test"]
        self.train_unique_classes = config["train_unique_classes"]

        # Freeze patch_embeddings
        for param in self.model.videomae.embeddings.parameters():
            param.requires_grad = False

        self.mm_fusion_module = self._build_transformer(config["hidden_size"],
                                                        config["num_layers_mm"],
                                                        config["num_heads"],
                                                        config["intermediate_size"])  # We do not use batch here
        self.task_specific_learning_module = self._build_transformer(config["hidden_size"],
                                                                     config["num_layers_task"],
                                                                     config["num_heads"],
                                                                     config["intermediate_size"],
                                                                     batch_first=True)  # We pass batch as first dimension
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        # Change transform to custom ones if we are using SAFSAR
        self.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        self.ss_features = []
        self.ss_labels = []

        self.global_classification_layer = nn.Linear(config["hidden_size"], config["n_train_classes"])
        self.model.cuda()
        self.tensor_transform = transforms.ToTensor()

    def custom_transform(self, x):
        return [x for x in self.processor(x)["pixel_values"][0]]

    def set_ss(self, support_set):
        # Keep 5-way 1-shot for now
        support_set = {k: v[0] for k, v in list(support_set.items())[:5]}
        processed = []
        for class_name, class_imgs in support_set.items():
            imgs = [self.tensor_transform(v) for v in self.custom_transform(class_imgs)]
            imgs = torch.stack(imgs)
            processed.append(imgs)
        processed = torch.stack(processed)
        self.ss_features = self.get_multimodal_features(processed, support_set.keys())
        self.ss_labels = list(support_set.keys())

    # Override methods to avoid using l2 loss during evaluation
    def set_train(self):
        self.use_l2_loss = True

    def set_eval(self):
        # During training, l2 loss is useless
        self.use_l2_loss = False

    def _build_transformer(self, hidden_size, num_layers, num_heads, intermediate_size, batch_first=False):
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, 
                                                dim_feedforward=intermediate_size, batch_first=batch_first)
        return TransformerEncoder(encoder_layer, num_layers=num_layers)

    def get_textual_embeddings(self, classes_names):
        # Get features of class names with BERT
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model.cuda()
        bert_model.eval()
        class_name_embeddings = []
        for class_name in classes_names:
            plain_class_name = class_name.replace('_', ' ')
            inputs = tokenizer(plain_class_name, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = bert_model(**inputs)
            class_name_embeddings.append(outputs.last_hidden_state.squeeze(0))
        bert_model = None
        tokenizer = None
        return class_name_embeddings

    def get_multimodal_features(self, support_set, support_textual_labels):
        # Generate support set prototypes
        support_set = support_set.reshape(self.way*self.shot, self.seq_len, 224, 3, 224)
        support_set = support_set.permute(0, 1, 3, 4, 2)
        if self.seq_len == 8:
            inputs = {"pixel_values": support_set.repeat_interleave(2, dim=1).cuda()}
        self.model.cuda()
        self.mm_fusion_module.cuda()
        outputs = self.model(**inputs)
        support_features = outputs.hidden_states[-1].mean(dim=1)
        support_features = self.model.fc_norm(support_features).reshape(self.way, self.shot, -1).mean(dim=1)
        # add textual features
        textual_embeddings = self.get_textual_embeddings(support_textual_labels)
        # textual_embeddings = [self.class_name_embeddings[x] for x in batch_class_list[support_labels].long()]
        raw_support_mm_features = [torch.cat((v.unsqueeze(0), t)) for v, t in zip(support_features, textual_embeddings)]
        support_mm_features = [self.mm_fusion_module(emb)[0] for emb in raw_support_mm_features]
        support_mm_features = torch.stack(support_mm_features)
        return support_mm_features

    def forward(self, target_set):

        support_mm_features = self.ss_features
        # target_set = target_set

        # Generate query prototypes
        target_set = target_set.reshape(-1, self.seq_len, 224, 3, 224)
        target_set = target_set.permute(0, 1, 3, 4, 2)
        if self.seq_len == 8:
            inputs = {"pixel_values": target_set.repeat_interleave(2, dim=1).cuda()}
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = self.model(**inputs)
        query_features = outputs.hidden_states[-1].mean(dim=1)
        query_features = self.model.fc_norm(query_features)

        # Repeat embeddings for each query
        support_mm_features = support_mm_features.unsqueeze(0).repeat(1, 1, 1)
        combined_features = torch.cat((query_features.unsqueeze(1), support_mm_features), dim=1)
        
        combined_features = self.task_specific_learning_module(combined_features)
        query_features_aug, support_mm_features_aug = combined_features.split([1, self.way], dim=1)

        similarity_matrix = self.cosine_similarity(
            query_features_aug.expand(-1, support_mm_features_aug.size(1), -1),
            support_mm_features_aug
        )

        return {"similarity_matrix": similarity_matrix}
