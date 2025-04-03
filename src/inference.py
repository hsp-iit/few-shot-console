from src.safsar import SAFSAR
from collections import OrderedDict
from PIL import Image
import torch

def load_model(config):
    # Load model
    model = SAFSAR(config)
    model.cuda()
    model.eval()
    model  # .half()
    torch.set_grad_enabled(False)
    state_dict = torch.load(config["checkpoint_path"])
    single_gpu_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("model.module", "model")
        if "global_classification" in name:
            continue
        single_gpu_state_dict[name] = v
    model.load_state_dict(single_gpu_state_dict, strict=False)

    return model

def inference(model, last_n_frames, os_loss):
    log = None
    if len(model.ss_labels) > 0:
        
        query_data = [Image.fromarray(x) for x in last_n_frames]
        query_data = [model.tensor_transform(v) for v in model.custom_transform(query_data)]
        query_data = torch.stack(query_data)  # .half()

        # Inference
        res = model(query_data)
        os_score = None
        if os_loss == "discriminator":
            os_score = res["disc_prob"].item()
            sim_mat = res["similarity_matrix"].squeeze(0)  # Remove batch dimension
        elif os_loss == "softmax":
            sim_mat = res["similarity_matrix"].squeeze(0)  # Remove batch dimension
            os_score = (sim_mat.max().item()+1)/2
            sim_mat = torch.nn.functional.softmax(sim_mat, dim=-1)
        sim_mat = sim_mat.detach().cpu().numpy()
        action_res = {action: sim_mat[i] for i, action in enumerate(list(dict.fromkeys(model.ss_labels)))}

        # visualize frame and activation together
        activations = res["activations"]
        res = {"actions": action_res, "log": log, "os_score": os_score,
               "activations": activations}
    else:
        res = {"actions": {}, "log": log, "os_score": 0, "activations": None}
    return res

async def async_inference(model, last_n_frames, os_loss):
    return inference(model, last_n_frames, os_loss)