from collections import OrderedDict
from demo.gui import HumanConsole
import torch
import cv2
import imageio.v2 as imageio
import time
import os
from pathlib import Path
import numpy as np
import shutil
import sys
sys.path.append("./strm")
from strm.model import CNN_STRM


n_frames = 8
window_seconds_length = 2
frame_time = 1.0 / (n_frames/window_seconds_length)
initial_ss = "base"

class SSException(Exception):
    pass


if __name__ == "__main__":

    # Connect to Camera
    W=640
    H=480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        try:
            # Load ss
            ss = {}
            ss_gifs = {}
            base_path = Path("demo") / "ss" / initial_ss
            for action in base_path.iterdir():
                ss[action.name] = []
                ss_gifs[action.name] = []
                for example in (base_path / action.name).iterdir():
                    example_imgs = []
                    for i in range(n_frames):
                        example_imgs.append(imageio.imread(base_path / action.name / example.name / f"{i}.jpg"))
                    ss[action.name].append(example_imgs)
                    ss_gifs[action.name].append(base_path / action.name / example.name / f"{action.name}.gif")

            class Args:
                method = "resnet50"
                trans_linear_in_dim = 2048
                trans_linear_out_dim = 1152
                temp_set = [2]
                seq_len = 8
                trans_dropout = 0.1
                way = len(ss)

            # Load model
            state_dict = torch.load("checkpoints/checkpoint75000.pt")["model_state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('.module', '')  # remove `.module`
                new_state_dict[name] = v
            model = CNN_STRM(Args())
            model.load_state_dict(new_state_dict)
            model = model.cuda()
            model.new_dist_loss_post_pat = [n.cuda(0) for n in model.new_dist_loss_post_pat]

            # Create GUI
            gui = HumanConsole(values=ss.keys(), gifs_paths=ss_gifs)

            # Loop
            log = None
            last_n_frames = []

            while True:

                start_time = time.time()

                ret, frame = cap.read()
                cv2.imshow('Camera', frame)
                frame = cv2.resize(frame, (224, 224))

                # Temporal window
                last_n_frames = last_n_frames[-(n_frames-1):]
                last_n_frames.append(frame)
                if len(last_n_frames) < n_frames:
                    continue

                # Inference
                if len(ss) > 0:
                    # Prepare data
                    ss_data = np.concatenate([ss[action] for action in ss.keys()], axis=0)
                    ss_data = ss_data.reshape((-1, 224, 224, 3))
                    ss_data = torch.FloatTensor(ss_data).cuda().permute(0, 3, 2, 1)
                    query_data = np.stack(last_n_frames)
                    query_data = torch.FloatTensor(query_data).cuda().permute(0, 3, 2, 1)
                    context_labels = []
                    for i, k in enumerate(ss.keys()):
                        context_labels += [i] * len(ss[k])
                    context_labels = torch.LongTensor(context_labels).cuda()

                    # Inference
                    res = model(ss_data, context_labels, query_data)
                    target_logits = res['logits'] + 0.1*res['logits_post_pat']
                    target_logits = torch.softmax(target_logits, dim=-1)

                    action_res = {action: target_logits[0][0][i].item() for i, action in enumerate(list(ss.keys()))}
                    res = {"actions": action_res, "log": log}
                else:
                    res = {"actions": {}, "log": log}
                cmd = gui.loop(res)

                # Learn an action
                if cmd is not None:
                    if cmd[0] == "ADDACTION":  # ADD ACTION ######################
                        log = "Learning action..."
                        action_frames = []
                        for i in range(n_frames):
                            start_time = time.time()
                            ret, frame = cap.read()
                            cv2.imshow('Camera', frame)
                            frame = cv2.resize(frame, (224, 224))
                            action_frames.append(frame)
                            # Show progress in gui
                            res["log"] = (i+1)/(n_frames+1)
                            gui.loop(res)
                            # Fps
                            time.sleep(max(0, frame_time - (time.time() - start_time)))
                        # Save support set images and gif
                        action_name = cmd[1]
                        os.makedirs(base_path / action_name, exist_ok=True)
                        example_id = [int(dir.name) for dir in Path(base_path / action_name).iterdir() if dir.is_dir() and dir.name.isdigit()]
                        example_id = 0 if len(example_id) == 0 else max(example_id)+1
                        example_id = str(example_id)
                        os.makedirs(base_path / action_name / example_id, exist_ok=True)
                        imageio.mimsave(base_path / action_name / example_id / f"{action_name}.gif", action_frames)
                        for i, frame in enumerate(action_frames):
                            imageio.imsave(base_path / action_name / example_id / f"{i}.jpg", frame)
                        log = "Done!"
                        gui.window.close()
                        raise SSException()

                    if cmd[0] == "DELETEACTION":  # DELETE ACTION ######################
                        action = cmd[1]
                        id_to_remove = cmd[2]
                        if id_to_remove == "all":
                            shutil.rmtree(base_path / action)
                            log = f"Removed action {action}"
                        else:
                            shutil.rmtree(base_path / action / id_to_remove)
                            log = f"Removed example {id_to_remove} from action {action}"
                        gui.window.close()
                        raise SSException()

                    if cmd[0] == "LOADSS":  # LOAD SS ######################
                        initial_ss = cmd[1]
                        gui.window.close()
                        raise SSException

                    if cmd[0] == "SAVESS":  # SAVE SS ######################
                        shutil.copytree(base_path, Path("demo") / "ss" / cmd[1])
                        log = f"Support set saved in {cmd[1]}"


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Wait fps
                time.sleep(max(0, frame_time - (time.time() - start_time)))

        except SSException:
            continue