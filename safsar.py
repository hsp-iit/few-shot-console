import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoImageProcessor, AutoModelForVideoClassification, BertTokenizer, BertModel
import numpy as np
from utils import BinaryClassificationModelSAFSAR
from torchvision import transforms

class SAFSAR(nn.Module):
    def __init__(self, config, disc=None, gc=None, dp=None):
        super(SAFSAR, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained(config["processor_name"])
        self.model = AutoModelForVideoClassification.from_pretrained(config["mm_model_name"], output_hidden_states=True)
        if dp:
            self.model = torch.nn.DataParallel(self.model)
        self.way = config["way"]
        self.shot = config["shot"]
        self.seq_len = config["seq_len"]
        self.query_per_class = config["query_per_class"]

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

        # self.use_textual_embedding = config["use_textual_embedding"]
        # self.class_name_embeddings = self.get_textual_embeddings(config["classes_names"])
        self.alpha = config["alpha"]

        self.discriminator = BinaryClassificationModelSAFSAR(768).cuda()
        self.tensor_transform = transforms.ToTensor()
        self.disc = True

    def custom_transform(self, x):
        return [x for x in self.processor(x)["pixel_values"][0]]

    def get_multimodal_features(self, support_set, support_textual_labels):
        # Generate support set prototypes
        # support_set = support_set.reshape(self.way*self.shot, self.seq_len, 224, 3, 224)
        support_set = support_set.permute(0, 1, 3, 4, 2)
        if self.seq_len == 8:
            inputs = {"pixel_values": support_set.repeat_interleave(2, dim=1).cuda()}
        outputs = self.model(**inputs)
        support_features = outputs.hidden_states[-1].mean(dim=1)
        support_features = self.model.fc_norm(support_features)  # .reshape(self.way, self.shot, -1).mean(dim=1)
        support_features_mean = []
        ss_labels_np = np.array(support_textual_labels)
        for c in np.unique(ss_labels_np):
            support_features_mean.append(support_features[ss_labels_np == c].mean(dim=0))
        support_features_mean = torch.stack(support_features_mean)
        # add textual features
        textual_embeddings = self.get_textual_embeddings(np.unique(ss_labels_np))
        # textual_embeddings = [self.class_name_embeddings[x] for x in batch_class_list[support_labels].long()]
        raw_support_mm_features = [torch.cat((v.unsqueeze(0), t)) for v, t in zip(support_features_mean, textual_embeddings)]
        support_mm_features = [self.mm_fusion_module(emb)[0] for emb in raw_support_mm_features]
        support_mm_features = torch.stack(support_mm_features)
        return support_mm_features

    def set_ss(self, support_set):
        # Keep 5-way 1-shot for now
        processed = []
        classes = []
        for class_name, class_examples in support_set.items():
            for class_imgs in class_examples:
                imgs = [self.tensor_transform(v) for v in self.custom_transform(class_imgs)]
                imgs = torch.stack(imgs)
                processed.append(imgs)
                classes.append(class_name)
        processed = torch.stack(processed)
        self.ss_labels = classes
        self.ss_features = self.get_multimodal_features(processed, classes)

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

    def forward(self, target_set):

        support_mm_features = self.ss_features
        # target_set = target_set

        # Generate query prototypes
        if len(target_set.shape) == 4:  # n_q, seq_len, 224, 3, 224
            n_queries = int(target_set.shape[0] / self.seq_len)
        elif len(target_set.shape) == 5:
            n_queries = target_set.shape[0]
        target_set = target_set.reshape(n_queries, self.seq_len, 224, 3, 224)
        target_set = target_set.permute(0, 1, 3, 4, 2)
        if self.seq_len == 8:
            inputs = {"pixel_values": target_set.repeat_interleave(2, dim=1).cuda()}
        inputs['pixel_values'] = inputs['pixel_values'].cuda()
        outputs = self.model(**inputs)
        query_features = outputs.hidden_states[-1].mean(dim=1)
        query_features = self.model.fc_norm(query_features)

        # Repeat embeddings for each query
        support_mm_features = support_mm_features.unsqueeze(0).repeat(n_queries, 1, 1)
        combined_features = torch.cat((query_features.unsqueeze(1), support_mm_features), dim=1)
        combined_features = self.task_specific_learning_module(combined_features)
        query_features_aug, support_mm_features_aug = combined_features.split([1, len(set(self.ss_labels))], dim=1)

        similarity_matrix = self.cosine_similarity(
            query_features_aug.expand(-1, support_mm_features_aug.size(1), -1),
            support_mm_features_aug
        )

        # If discriminator, use it
        if self.disc:
            all_prototypes_differences = query_features_aug.expand(-1, support_mm_features_aug.size(1), -1) - support_mm_features_aug
            predictions = torch.argmax(similarity_matrix, dim=-1)
            best_diffs = all_prototypes_differences[torch.arange(len(all_prototypes_differences)), predictions]
            disc_prob = self.discriminator(best_diffs)
        else:
            disc_prob = None

        return {"similarity_matrix": similarity_matrix,
                "disc_prob": disc_prob}

    def get_debug_data(self):
        return self.debug_data

    def visual_debug(self, similarity_matrix=None, support_global_logits=None, query_global_logits=None, videodataset=None, support_labels=None, target_labels=None, batch_class_list=None, support_set=None, target_set=None, disc_prob=None, unknown_labels=None):
        import cv2
        import imageio
        import os

        # Save support set gif
        support_set = support_set.reshape(-1, self.seq_len, 224, 3, 224)
        support_set = support_set[torch.argsort(support_labels)]
        support_set = support_set.reshape(self.way, self.shot, self.seq_len, 224, 3, 224).permute(0, 1, 2, 5, 3, 4)
        concatenated_frames = []
        support_classes = [videodataset.class_folders[int(batch_class_list[i])] for i in range(self.way)]

        for i in range(self.way):
            for j in range(self.shot):
                frames = []
                for k in support_set[i, j]:

                    frame_rgb = k.cpu().numpy()
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    frames.append(frame_rgb)
                concatenated_frames.append(frames)

        concatenated_frames = np.stack([np.stack(x) for x in concatenated_frames])
        concatenated_frames = concatenated_frames.reshape(self.way, self.shot, self.seq_len, 224, 224, 3)
        concatenated_frames = np.concatenate(concatenated_frames, axis=2)
        concatenated_frames = np.concatenate(concatenated_frames, axis=2)
        os.makedirs(f'visual_debug/{self.debug_samples_counter}', exist_ok=True)
        imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/ss.gif', concatenated_frames, duration=250, loop=0)
        with open(f'visual_debug/{self.debug_samples_counter}/ss.txt', 'w') as f:
            for item in support_classes:
                f.write("%s\n" % item)

        # Check predictions
        good_closed = similarity_matrix.argmax(dim=-1) == target_labels
        if disc_prob is None:
            accept_score = similarity_matrix.max(dim=-1).values
        else:
            accept_score = disc_prob.detach().cpu().numpy()
        # good_open = accept_score == (target_labels != -1)
        true_open = target_labels != -1
        pred_open = accept_score > 0.5
        

        # Save queries gif
        target_set = target_set.reshape(self.way, self.query_per_class, self.seq_len, 224, 3, 224).permute(0, 1, 2, 5, 3, 4)
        # query_labels = torch.argsort(support_labels)[target_labels]
        # query_labels[target_labels == -1] = -1
        unknown_counter = 0
        query_labels = []
        for i in range(self.way):
            for j in range(self.query_per_class):
                if target_labels[i*self.query_per_class+j] == -1:
                    query_label = videodataset.class_folders[int(unknown_labels[i*self.query_per_class+j])]
                    unknown_counter += 1
                else:
                    query_label = videodataset.class_folders[int(batch_class_list[target_labels[i*self.query_per_class+j]])]
                query_labels.append(query_label)
                concatenated_frame = []
                for k in target_set[i, j]:
                    frame_rgb = k.cpu().numpy()
                    frame_rgb = ((frame_rgb - frame_rgb.min()) / (frame_rgb.max() - frame_rgb.min()) * 255).astype(np.uint8)
                    concatenated_frame.append(frame_rgb)
                # determine it TP TN FP FN
                cur = i*self.query_per_class+j
                res = ""
                if true_open[cur] and pred_open[cur] and good_closed[cur]:
                    res = "TP"
                elif not true_open[cur] and not pred_open[cur]:
                    res = "TN"
                elif true_open[cur] and not pred_open[cur]:
                    res = "FN"
                elif not true_open[cur] and pred_open[cur]:
                    res = "FP"
                imageio.mimsave(f'visual_debug/{self.debug_samples_counter}/{res}_{accept_score[cur]}_{query_label}.gif', concatenated_frame, duration=250, loop=0)

        # # Save results
        # true_targets = torch.argsort(support_labels)[target_labels]
        # true_targets[target_labels == -1] = -1
        # similarity_matrix = similarity_matrix.detach().cpu().numpy()
        # with open(f'visual_debug/{self.debug_samples_counter}/similarity_matrix.txt', 'w') as f:
        #     for item in support_classes:
        #         f.write("%s " % item)
        #     f.write("\n")
        #     lazy_counter = 0
        #     for item in similarity_matrix:
        #         f.write(f"%s\t{true_targets[lazy_counter]} {query_labels[lazy_counter]}\n" % item)
        #         lazy_counter += 1
        self.debug_samples_counter += 1
