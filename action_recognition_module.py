#!/usr/bin/env python

import yarp
import numpy as np
import logging
import json
import asyncio
from pathlib import Path
import cv2
import time
from src.inference import load_model, inference, async_inference
from src.utils import load_ss
from src.gui import HumanConsole
import sys
import os
import shutil
import imageio.v2 as imageio

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class ActionRecognitionModule(yarp.RFModule):
    def configure(self, rf):
        # Read config
        with open("config.json", "r") as stream:
            self.config = json.load(stream)
        
        # Make sure that image dimensions are set in the config
        try:
            self.image_height = self.config["image_height"]
            self.image_width = self.config["image_width"]
        except KeyError as e:
            logging.error(f"Missing key in config: {e}")
            return False
        
        # Load main parameters
        self.seq_len = self.config["seq_len"]
        self.frame_time = 1.0 / (self.seq_len/self.config["window_seconds"])
        self.last_n_frames = []
        self.start_time = time.time()
        logging.debug("Loaded config")

        # Load support set
        ss_name = self.config["initial_ss"]
        ss_path = Path("src") / "ss" / ss_name
        
        if not ss_path.exists():
            logging.error(f"Support set path does not exist: {ss_path}")
            return False
            
        self.ss, self.ss_gifs = load_ss(ss_path, self.seq_len)

        # Create input port
        self.input_port = yarp.BufferedPortImageRgb()
        self.input_port.open("/action_recognition/image:i")
        logging.debug(f"Port opened: {self.input_port.getName()}")

        # Create output port
        self.output_port = yarp.BufferedPortImageRgb()
        self.output_port.open("/action_recognition/image:o")
        logging.debug(f"Port opened: {self.output_port.getName()}")

        # Create output port to send the predicted action
        self.output_action_port = yarp.Port()
        self.output_action_port.open("/action_recognition/action:o")
        logging.debug(f"Port opened: {self.output_action_port.getName()}")

        # Setup input buffer for reading images
        self.input_buffer_array = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.input_buffer_image = yarp.ImageRgb()
        self.input_buffer_image.resize(self.image_width, self.image_height)
        self.input_buffer_image.setExternal(self.input_buffer_array.data, self.image_width, self.image_height)
        logging.debug(f"Image buffer created with shape: {self.input_buffer_array.shape}")

        # Setup output buffer for writing processed images
        self.output_buffer_array = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.output_buffer_image = yarp.ImageRgb()
        self.output_buffer_image.resize(self.image_width, self.image_height)
        self.output_buffer_image.setExternal(self.output_buffer_array.data, self.image_width, self.image_height)
        logging.debug(f"Output buffer created with shape: {self.output_buffer_array.shape}")

        # Load the model
        self.model = load_model(self.config)
        self.model.set_ss(self.ss)

        # Create the gui
        self.gui = HumanConsole(values=list(self.ss.keys()), gifs_paths=self.ss_gifs)
        self.current_task = None
        self.last_result = None

        return True


    def processResult(self, frame) -> None:
        
        if frame is None:
            return

        # Get the best action
        current_best_action = self.gui.current_action    
        
        # Resize the frame to the original res to display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # opencv is bgr, move it to rgb
        frame = cv2.resize(frame, (self.image_width, self.image_height))

        # Add best action to the frame     
        if current_best_action is not None:
            cv2.putText(frame, str(current_best_action), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3, cv2.LINE_AA)

        # Forward the image to the output port
        self.output_buffer_array[:, :] = frame
        output_image = self.output_port.prepare()
        output_image.copy(self.output_buffer_image)
        self.output_port.write()

        # Send the action to the output action port
        if current_best_action is not None:
            action_str = current_best_action
        else:
            action_str = ""

        action_msg = yarp.Bottle()
        action_msg.addString(action_str)
        self.output_action_port.write(action_msg)


    async def async_updateModule(self):
        # Check if there's an image available on the input port
        input_image = self.input_port.read(False)
        frame = None
        result = None

        if input_image is not None:

            # Copy the received image to the input buffer
            self.input_buffer_image.copy(input_image)
            frame = np.copy(self.input_buffer_array)
            logging.debug(f"Image received with shape: {frame.shape}")

            # Convert the image to a format suitable for the model
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # opencv is bgr, move it to rgb

            # Save a temporal window
            if len(self.last_n_frames) < self.seq_len:  # Skip computation if we don't have enough frames
                self.last_n_frames.append(frame)
                return True
                
            if time.time() - self.start_time > self.frame_time:  # Add frame to last_n_frames only if it has passed at least frame_time
                if len(self.last_n_frames) == self.seq_len:
                    self.last_n_frames = self.last_n_frames[-(self.seq_len-1):]
                self.last_n_frames.append(frame)
                self.start_time = time.time()

            logging.debug(f"Saved recent frames: {len(self.last_n_frames)}")

            # Async inference

            if not self.current_task: # We start that no task is found
                self.current_task = asyncio.create_task(async_inference(self.model, self.last_n_frames, self.config["os_loss"]))
            elif self.current_task.done():  # Processing finished
                result = await self.current_task
                self.current_task = None
            elif not self.current_task.done():  # No processing in progress
                await asyncio.sleep(0.03)  # Add a small delay to reduce the frequency of checks
            elif result is None:  # Special case for first iteration
                result = await self.current_task
            logging.debug(f"Inference result: {result}")
        else:
            logging.debug("No image received from input port")

        if result is not None:
            self.last_result = result
            cmd = self.gui.loop(self.last_result)
            self.processResult(frame)
            self.gui_server(cmd)

        return True
    
    def updateModule(self):
        return asyncio.run(self.async_updateModule())
    
    def gui_server(self,cmd):
        if cmd is not None:
            if cmd[0] == "ADDACTION":  # ADD ACTION ######################
                # Continue displaying camera frames for 3 seconds
                start_time = time.time()
                while time.time() - start_time < 3:
                    self.gui.loop({"log": f"Get ready in {3 - int(time.time() - start_time)}"})                
                action_frames = []
                for i in range(self.config["seq_len"]):
                    start_time = time.time()

                    # Read image from yarp port
                    input_image = self.input_port.read(False)
                    if input_image is None:
                        logging.warning("No image received from input port")
                        continue
                    # Resize the image to the model input size
                    self.input_buffer_image.copy(input_image)
                    frame = np.copy(self.input_buffer_array)
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # TODO TEST
                    action_frames.append(frame)

                    # Show progress in gui
                    self.last_result["log"] = f"{int(((i+1)/(self.config['seq_len']+1))*100)}%"
                    self.last_result["is_true"] = 0
                    self.gui.loop(self.last_result)
                    
                    # Fps
                    time.sleep(max(0, self.frame_time - (time.time() - start_time)))
                
                # Save support set images and gif
                action_name = cmd[1]
                os.makedirs(self.ss_path / action_name, exist_ok=True)
                example_id = [int(dir.name) for dir in Path(self.ss_path / action_name).iterdir() if dir.is_dir() and dir.name.isdigit()]
                example_id = 0 if len(example_id) == 0 else max(example_id)+1
                example_id = str(example_id)
                os.makedirs(self.ss_path / action_name / example_id, exist_ok=True)
                imageio.mimsave(self.ss_path / action_name / example_id / f"{action_name}.gif", action_frames)
                for i, frame in enumerate(action_frames):
                    imageio.imsave(self.ss_path / action_name / example_id / f"{i}.jpg", frame)
                self.refresh_gui()

            if cmd[0] == "DELETEACTION":  # DELETE ACTION ######################
                action = cmd[1]
                id_to_remove = cmd[2]
                if id_to_remove == "all":
                    shutil.rmtree(self.ss_path / action)
                    logging.debug(f"Removed action {action}")
                else:
                    shutil.rmtree(self.ss_path / action / id_to_remove)
                    logging.debug(f"Removed example {id_to_remove} from action {action}")
                self.refresh_gui()

            if cmd[0] == "LOADSS":  # LOAD SS ######################
                ss_name = cmd[1]
                self.ss_path = Path("src") / "ss" / ss_name
                self.refresh_gui()

            if cmd[0] == "SAVESS":  # SAVE SS ######################
                shutil.copytree(self.ss_path, Path("demo") / "ss" / cmd[1])
                logging.debug(f"Support set saved in {cmd[1]}")

    def getPeriod(self):
        # Return the period of the module in seconds
        return 0.03  # 30ms, adjust based on requirements

    def interruptModule(self):
        # Handle module interruption
        logging.info("Action recognition module interrupted")
        self.input_port.interrupt()
        self.output_port.interrupt()
        return True

    def close(self):
        # Close ports and cleanup resources
        logging.info("Closing action recognition module")
        self.input_port.close()
        self.output_port.close()
        return True

    def refresh_gui(self):
        self.gui.close()
        self.last_result = None
        self.ss, self.ss_gifs = load_ss(self.ss_path, self.seq_len)
        self.model.set_ss(self.ss)
        self.gui = HumanConsole(values=list(self.ss.keys()), gifs_paths=self.ss_gifs)
        logging.info("GUI refreshed")


if __name__ == "__main__":
    # Initialize YARP network
    yarp.Network.init()

    # Create and configure the module
    module = ActionRecognitionModule()
    rf = yarp.ResourceFinder()
    rf.configure(sys.argv)  # Empty arguments
    
    logging.info("Starting action recognition module")
    
    # Run the module
    try:
        if not module.configure(rf):
            logging.error("Failed to configure the module")
        else:
            module.runModule()
    except KeyboardInterrupt:
        logging.info("Stopping action recognition module.")
    except Exception as e:
        logging.error(f"Error in action recognition module: {e}")
    finally:
        module.close()
        logging.info("Action recognition module closed.")

    # Clean up the YARP network
    yarp.Network.fini()
