# Export PYTHONPATH to pwd
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(f"{os.getcwd()}/strm")
from collections import OrderedDict
from demo.gui import HumanConsole
import torch
import cv2
import imageio.v2 as imageio
import time
from pathlib import Path
import numpy as np
import shutil
from strm.model import CNN_STRM
import asyncio
from demo.demo_utils import SSException


async def do_inference(model, ss, last_n_frames, precomputed_context_features):
    log = None
    if len(ss) > 0:
        # Prepare data
        ss_data = np.concatenate([ss[action] for action in ss.keys()], axis=0)
        ss_data = ss_data.reshape((-1, 224, 224, 3))
        ss_data = torch.FloatTensor(ss_data).cuda().permute(0, 3, 1, 2)
        query_data = np.stack(last_n_frames)
        query_data = torch.FloatTensor(query_data).cuda().permute(0, 3, 1, 2)
        context_labels = []
        for i, k in enumerate(ss.keys()):
            context_labels += [i] * len(ss[k])
        context_labels = torch.LongTensor(context_labels).cuda()

        # Inference
        with torch.no_grad():
            # normalize
            ss_data = ss_data/255.0
            query_data = query_data/255.0
            res, precomputed_context_features = model(ss_data, context_labels, query_data, 
                                                        precomputed_context_features=precomputed_context_features)
        target_logits = res['logits'] + 0.1*res['logits_post_pat']
        target_logits = torch.softmax(target_logits, dim=-1)
        # print(target_logits)  # TODO REMOVE DEBUG
        # print(res['open_set_logits'])
        action_res = {action: target_logits[0][0][i].item() for i, action in enumerate(list(ss.keys()))}

        # visualize frame and activation together
        activations = model.activations
        if len(activations) == 2:  # first time it computes also support set features
            _, target_activations = model.activations
        else:
            target_activations = model.activations[0]
        res = {"actions": action_res, "log": log, "is_true": res['open_set_logits'].cpu().numpy().squeeze().item(), 
               "target_activations": target_activations}
    else:
        res = {"actions": {}, "log": log, "is_true": 0., "target_activations": None}
    return res, precomputed_context_features

def load_ss(ss_path, n_frames):
    # Load ss
    ss = {}
    ss_gifs = {}
    for action in ss_path.iterdir():
        ss[action.name] = []
        ss_gifs[action.name] = []
        for example in (ss_path / action.name).iterdir():
            example_imgs = []
            for i in range(n_frames):
                example_imgs.append(imageio.imread(ss_path / action.name / example.name / f"{i}.jpg"))
            ss[action.name].append(example_imgs)
            ss_gifs[action.name].append(ss_path / action.name / example.name / f"{action.name}.gif")
    return ss, ss_gifs


async def main_loop():

    # Connect to Camera
    W=640
    H=480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    processing = False

    # Loop args
    n_frames = 8
    window_seconds_length = 2
    frame_time = 1.0 / (n_frames/window_seconds_length)
    ss_name = "base"
    ss_path = Path("demo") / "ss" / ss_name
    model_checkpoint = "strm_70k_os_nturgbd.pt"
    # model_checkpoint = "strm_300k_os_ssv2.pt"

    # Main loop
    while True:
        try:
            # Load support set
            ss, ss_gifs = load_ss(ss_path, n_frames)

            # Load model
            class Args:
                method = "resnet50"
                trans_linear_in_dim = 2048
                trans_linear_out_dim = 1152
                temp_set = [2]
                seq_len = 8
                trans_dropout = 0.1
                way = len(ss)
            state_dict = torch.load(f"checkpoints/{model_checkpoint}")["model_state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('.module', '')  # remove `.module`
                new_state_dict[name] = v
            model = CNN_STRM(Args())
            model.load_state_dict(new_state_dict)
            model = model.cuda()
            model = model.eval()
            model.register_activation_hook()
            model.new_dist_loss_post_pat = [n.cuda(0) for n in model.new_dist_loss_post_pat]

            # Create GUI
            gui = HumanConsole(values=ss.keys(), gifs_paths=ss_gifs)

            # Loop
            log = None
            precomputed_context_features = None
            last_n_frames = []
            res = None
            start_time = time.time()

            while True:

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Camera not available")
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # opencv is bgr, move it to rgb

                # Temporal window
                if time.time() - start_time > frame_time: # Add frame to last_n_frames only if it has passed at least frame_time
                    if len(last_n_frames) == n_frames:
                        last_n_frames = last_n_frames[-(n_frames-1):]
                    last_n_frames.append(frame)
                    start_time = time.time()
                if len(last_n_frames) < n_frames:  # Skip computation if we don't have enough frames
                    continue
                

                # Async inference
                if not processing:  # No processing in progress
                    inference_task = asyncio.create_task(do_inference(model, ss, last_n_frames, precomputed_context_features))
                    processing = True
                if processing and inference_task.done():  # Processing finished
                    res, precomputed_context_features = await inference_task
                    processing = False
                if processing and not inference_task.done():  # Processing in progress
                    await asyncio.sleep(0.01)  # Add a small delay to reduce the frequency of checks
                if res is None:  # Special case for first iteration
                    res, precomputed_context_features = await inference_task

                cmd = gui.loop(res)

                # Learn an action
                if cmd is not None:
                    if cmd[0] == "ADDACTION":  # ADD ACTION ######################
                        # Continue displaying camera frames for 3 seconds
                        start_time = time.time()
                        while time.time() - start_time < 3:
                            ret, frame = cap.read()
                            frame = cv2.resize(frame, (224, 224))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # TODO TEST
                            cv2.imshow('Camera', frame)
                            cv2.waitKey(1)
                        action_frames = []
                        for i in range(n_frames):
                            start_time = time.time()
                            ret, frame = cap.read()
                            frame = cv2.resize(frame, (224, 224))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # TODO TEST
                            cv2.imshow('Camera', frame)
                            cv2.waitKey(1)
                            action_frames.append(frame)
                            # Show progress in gui
                            res["log"] = f"{int(((i+1)/(n_frames+1))*100)}%"
                            res["is_true"] = 0
                            gui.loop(res)
                            # Fps
                            time.sleep(max(0, frame_time - (time.time() - start_time)))
                        # Save support set images and gif
                        action_name = cmd[1]
                        os.makedirs(ss_path / action_name, exist_ok=True)
                        example_id = [int(dir.name) for dir in Path(ss_path / action_name).iterdir() if dir.is_dir() and dir.name.isdigit()]
                        example_id = 0 if len(example_id) == 0 else max(example_id)+1
                        example_id = str(example_id)
                        os.makedirs(ss_path / action_name / example_id, exist_ok=True)
                        imageio.mimsave(ss_path / action_name / example_id / f"{action_name}.gif", action_frames)
                        for i, frame in enumerate(action_frames):
                            imageio.imsave(ss_path / action_name / example_id / f"{i}.jpg", frame)
                        log = "Done!"
                        raise SSException()

                    if cmd[0] == "DELETEACTION":  # DELETE ACTION ######################
                        action = cmd[1]
                        id_to_remove = cmd[2]
                        if id_to_remove == "all":
                            shutil.rmtree(ss_path / action)
                            log = f"Removed action {action}"
                        else:
                            shutil.rmtree(ss_path / action / id_to_remove)
                            log = f"Removed example {id_to_remove} from action {action}"
                        raise SSException()

                    if cmd[0] == "LOADSS":  # LOAD SS ######################
                        ss_name = cmd[1]
                        raise SSException

                    if cmd[0] == "SAVESS":  # SAVE SS ######################
                        shutil.copytree(ss_path, Path("demo") / "ss" / cmd[1])
                        log = f"Support set saved in {cmd[1]}"

                # Add model activation on the frame
                if res is not None and res["target_activations"] is not None:
                    target_activations = res["target_activations"]
                    model.activations = []
                    target_activations = target_activations.reshape(-1, 8, 2048, 7, 7)
                    target_activations = target_activations.norm(dim=2)
                    target_activations = target_activations[0, -1, ...]  # First (only) query, last frame  (7x7)
                    act = target_activations
                    act = act.cpu().numpy()
                    act = (act - act.min()) / (act.max() - act.min())
                    act = cv2.resize(act, (frame.shape[1], frame.shape[0]))
                    heatmap = cv2.applyColorMap(np.uint8(255 * act), cv2.COLORMAP_JET)
                    frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
                    action = list(res["actions"].keys())[list(res["actions"].values()).index(max(list(res["actions"].values())))]
                    cv2.putText(frame, action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        except SSException as e:
            gui.close()
            _ = await inference_task
            processing = False
            res = None
            ss_path = Path("demo") / "ss" / ss_name
            continue


if __name__ == "__main__":
    asyncio.run(main_loop())
