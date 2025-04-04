import PySimpleGUI as sg
import time

sg.theme('DarkTeal9')

class SSException(Exception):
    pass

class HumanConsole:

    def __init__(self, spawn_location=None, values=None, gifs_paths=None):
        self.acquisition_time = 2
        self.values = values
        self.gifs_paths = gifs_paths
        self.last_thr = None
        self.log = ""
        self.window_size = 8
        self.lay_actions = [[
                         sg.ProgressBar(1, orientation='h', size=(20, 20), key=f"FS-{key}"),
                         sg.ProgressBar(1, orientation='h', size=(20, 20), key=f"OS-{key}"),
                         sg.Text(key, key=f"ACTION-{key}")] 
                         for key in self.values]
        self.lay_thrs = [[sg.Slider(range=(0, 100), size=(20, 20), orientation='h', key='FS-THR', default_value=33) , sg.Slider(range=(0, 100), size=(20, 20), orientation='h', key='OS-THR', default_value=90)]]
        self.lay_commands = [[sg.Button("Remove", key="DELETE", size=(6, 1)),
                              sg.Combo(list(self.values), size=(20,1), enable_events=False, key='DELETEACTION', readonly=True),
                              sg.Combo(["all", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=(3,1), enable_events=False, key='DELETEID', readonly=True)],
                             [sg.Button("Add", key="ADD", size=(6, 1)),
                              sg.Combo(list(self.values), size=(20,1), enable_events=False, key='ADDACTION')],
                             [sg.Text("", key="log")]]
        self.lay_io = [[sg.FolderBrowse("Load", initial_folder="./src/ss"), sg.In(size=(25,1), key='LOAD', enable_events=True), ],
                       [sg.FolderBrowse("Save", initial_folder="./src/ss"), sg.In(size=(25,1), key='SAVE', enable_events=True), ]]
        
        # Load gifs
        self.ss_gifs_keys = {}
        if gifs_paths is not None:
            self.lay_support = [[sg.Text("Support Set"), sg.HorizontalSeparator()]]
            for action in gifs_paths.keys():
                self.lay_support.append([sg.Text(action)])
                # example_gifs = [sg.Text(action)]
                example_gifs = [sg.VerticalSeparator()]
                for example in gifs_paths[action]:
                    example_gifs.append(sg.Text(example.parent.name, key="log"))
                    example_gifs.append(sg.Image(str(example), key=f"SUPPORT_SET_{action}_{example}", expand_x=False, expand_y=True))
                    example_gifs.append(sg.VerticalSeparator())
                    self.ss_gifs_keys[f"SUPPORT_SET_{action}_{example}"] = example
                self.lay_support.append(example_gifs)
                self.lay_support.append([sg.HorizontalSeparator()])
        else:
            self.lay_support = [[]]

        self.lay_left = [[sg.Text("Scores"), sg.HorizontalSeparator()],
                         [sg.Text('Few Shot', size=(20, 1))],
                         [sg.Column(self.lay_actions)],
                         [sg.Text("Thresholds"), sg.HorizontalSeparator()],
                         [sg.Column(self.lay_thrs)],
                         [sg.Text("SS Modifiers"), sg.HorizontalSeparator()],
                         [sg.Column(self.lay_commands)],
                         [sg.Text("SS I/O"), sg.HorizontalSeparator()],
                         [sg.Column(self.lay_io)]]
        self.lay_right = [[sg.Column(self.lay_support, scrollable=True, expand_x=True, expand_y=True, size=(300, 300))]]
        self.lay_final = [[sg.Column(self.lay_left, expand_x=True, expand_y=True),
                          sg.VerticalSeparator(),
                          sg.Column(self.lay_right, expand_x=True, expand_y=True)]]
        if spawn_location is not None:
            self.window = sg.Window('Few-Shot Console', self.lay_final, location=spawn_location, resizable=True, finalize=True)
        else:
            self.window = sg.Window('Few-Shot Console', self.lay_final, resizable=True, finalize=True)
        # Start the window maximized but with decorations (title bar and taskbar visible)
        self.window.maximize()

        self.current_action = None
        self.last_action_time = time.time()

    def set_current_action(self, action):
        if action is None:
            self.current_action = ""
            return
        if action == self.current_action:
            return
        # if we detect an action, we don't want to detect another actions for 2 seconds
        if time.time() - self.last_action_time < self.acquisition_time:
            return
        self.last_action_time = time.time()
        self.current_action = action

    def loop(self, data):

        # EXIT IF NECESSARY
        event, val = self.window.read(timeout=10)
        if event == sg.WIN_CLOSED:
            exit()

        # ACTIONS
        if "actions" in data:
            actions = data["actions"]
            is_true = data["os_score"]
            if actions is not None:
                # RESTART IF SS HAS CHANGED
                if self.values != list(actions.keys()):
                    raise SSException("Support set has changed!")
                # UPDATE SCORES
                if len(self.values) > 0:
                    best_action = max(zip(actions.values(), actions.keys()))[1]
                    for key in actions:
                        self.window[f"FS-{key}"].update(actions[key])
                        if key == best_action:
                            self.window[f"OS-{key}"].update(is_true)
                            if actions[best_action] > val['FS-THR']/100 and is_true > val['OS-THR']/100:
                                self.window[f"ACTION-{best_action}"].update(text_color="red")
                                self.set_current_action(best_action)
                            else:
                                self.window[f"ACTION-{best_action}"].update(text_color="white")
                                self.set_current_action(None)
                        else:
                            self.window[f"OS-{key}"].update(0.)
                            self.window[f"ACTION-{key}"].update(text_color="white")

        # LOG
        if data["log"] is not None:
            self.window["log"].update(data["log"])

        # REMOVE ACTION
        if "DELETE" in event:
            action = val["DELETEACTION"]
            id_to_remove = val["DELETEID"]
            if len(id_to_remove) == 0:
                self.window["log"].update("Please select examples")
            if len(action) == 0:
                self.window["log"].update("Please select the action")
            else:
                return ["DELETEACTION", action, id_to_remove]

        # ADD ACTION
        if "ADD" in event:
            for key in ["LOAD", "DELETE", "DELETEACTION", "ADDACTION", "DELETEID", "FS-THR", "ADD"]:
                self.window[key].update(disabled=True)
            return ["ADDACTION", val["ADDACTION"]]

        # LOAD
        if "LOAD" in event:
            self.log = "Loading support set..."
            return ["LOADSS", val["LOAD"]]

        # SAVE
        if "SAVE" in event:
            self.window["log"].update("Saving support set...")
            return ["SAVESS", val["SAVE"]]

        if event == "__TIMEOUT__":
            if val['FS-THR'] != self.last_thr:
                self.last_thr = val['FS-THR']
        
        # UPDATE SUPPORT SET
        if self.lay_support != [[]]:
            for k, v in self.ss_gifs_keys.items():
                self.window[k].UpdateAnimation(v, time_between_frames=100)

    def close(self):
        self.window.close()
