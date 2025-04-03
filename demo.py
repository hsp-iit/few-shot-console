# Export PYTHONPATH to pwd
import os
from src.gui import HumanConsole
import torch
import cv2
import imageio.v2 as imageio
import time
from pathlib import Path
import numpy as np
import shutil
import asyncio
from src.gui import SSException
import json
from transformers import AutoImageProcessor

from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from src.utils import FakeCap, load_ss
from src.inference import load_model, async_inference
import tkinter as tk

# Change transform to custom ones if we are using SAFSAR
processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

def custom_transform(x):
    return [x for x in processor(x)["pixel_values"][0]]

async def main_loop():

    # Load configs
    with open("config.json", 'r') as f:
        config = json.load(f)

    if config["video_path"] == "":
        # Connect to Camera
        W=640
        H=480
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.read()[0]:
                cameras.append(i)
                cap.release()
        print(f"Available cameras: {cameras}")
        if len(cameras) == 0:
            raise Exception("No cameras available")
        cap = cv2.VideoCapture(cameras[-1])
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = FakeCap(config["video_path"])
    processing = False

    # Get screen dimensions
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera', int((screen_width // 2)*0.9), screen_height // 2)
    cv2.moveWindow('Camera', 0, int((screen_height // 2)*0.8))

    # Loop args
    frame_time = 1.0 / (config["seq_len"]/config["window_seconds"])
    ss_name = config["initial_ss"]
    ss_path = Path("src") / "ss" / ss_name

    # Create log dir
    if config["save_log"]:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = Path(".") / "logs" / timestamp
        os.makedirs(log_path, exist_ok=True)

    # Load model
    model = load_model(config)

    # Main loop
    outputs = []
    while True:
        try:
            # Load support set and model
            ss, ss_gifs = load_ss(ss_path, config["seq_len"])
            model.set_ss(ss)

            # Create GUI
            gui = HumanConsole(values=list(ss.keys()), gifs_paths=ss_gifs)

            # Loop
            last_n_frames = []
            res = None
            start_time = time.time()
            counter = 0

            while True:

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Camera not available")
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # opencv is bgr, move it to rgb

                # Temporal window
                if len(last_n_frames) < config["seq_len"]:  # Skip computation if we don't have enough frames
                    last_n_frames.append(frame)
                    continue
                if time.time() - start_time > frame_time: # Add frame to last_n_frames only if it has passed at least frame_time
                    if len(last_n_frames) == config["seq_len"]:
                        last_n_frames = last_n_frames[-(config["seq_len"]-1):]
                    last_n_frames.append(frame)
                    start_time = time.time()

                # Async inference
                if not processing:  # No processing in progress
                    inference_task = asyncio.create_task(async_inference(model, last_n_frames, config["os_loss"]))
                    processing = True
                if processing and inference_task.done():  # Processing finished
                    res = await inference_task
                    processing = False
                if processing and not inference_task.done():  # Processing in progress
                    # pass
                    await asyncio.sleep(0.01)  # Add a small delay to reduce the frequency of checks
                if res is None:  # Special case for first iteration
                    res = await inference_task

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
                            gui.loop({"log": f"Get ready in {3 - int(time.time() - start_time)}"})
                        action_frames = []
                        for i in range(config["seq_len"]):
                            start_time = time.time()
                            ret, frame = cap.read()
                            frame = cv2.resize(frame, (224, 224))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # TODO TEST
                            cv2.imshow('Camera', frame)
                            cv2.waitKey(1)
                            action_frames.append(frame)
                            # Show progress in gui
                            res["log"] = f"{int(((i+1)/(config['seq_len']+1))*100)}%"
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
                if res is not None and res["activations"] is not None and config["show_activations"]:
                    act = res["activations"]
                    act = act.cpu().numpy()
                    act = (act - act.min()) / (act.max() - act.min())
                    act = cv2.resize(act, (frame.shape[1], frame.shape[0]))
                    heatmap = cv2.applyColorMap(np.uint8(255 * act), cv2.COLORMAP_JET)
                    frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

                # save raw frame to log_path
                if config["save_log"]:
                    cv2.imwrite(log_path / f"{counter}.jpg", frame)

                # If opencv window has opposite color, then the image is correct
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640*2, 480*2))
                cv2.putText(frame, gui.current_action, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.imshow('Camera', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                outputs.append(frame)
                if config["save_output_video"]:
                    if cap.is_over():
                        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280, 960))
                        for frame in outputs:
                            out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        out.release()
                        exit()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if config["video_path"] is not None:
                    time.sleep(1/40)

                counter += 1


        except SSException as e:
            gui.close()
            _ = await inference_task
            processing = False
            res = None
            ss_path = Path("src") / "ss" / ss_name
            continue


if __name__ == "__main__":
    asyncio.run(main_loop())
