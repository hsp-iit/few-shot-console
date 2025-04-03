#!/usr/bin/env python

import yarp
import numpy as np
import logging
import json
from pathlib import Path
import cv2
import time
from src.inference import load_model, inference
from src.utils import load_ss
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self.os_threshold = self.config.get("threshold", 0.9)
        self.fs_threshold = self.config.get("fs_threshold", 0.33)
        logging.debug("Loaded config")

        # Load support set
        ss_name = self.config["initial_ss"]
        ss_path = Path("src") / "ss" / ss_name
        
        if not ss_path.exists():
            logging.error(f"Support set path does not exist: {ss_path}")
            return False
            
        ss, _ = load_ss(ss_path, self.seq_len)

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
        self.model.set_ss(ss)
        
        return True

    def updateModule(self):
        # Check if there's an image available on the input port
        input_image = self.input_port.read(False)
        if input_image is None:
            return True  # Continue running the module
            
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

        # Perform action recognition on the image
        result = inference(self.model, self.last_n_frames, self.config["os_loss"])
        logging.debug(f"Inference result: {result}")

        actions = result["actions"]
        is_true = result["os_score"]

        if actions is not None:
            best_action = max(zip(actions.values(), actions.keys()))[1]

        if actions[best_action] > self.fs_threshold and is_true > self.os_threshold:
            current_best_action = best_action
        else:
            current_best_action = None
            
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

        return True

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
