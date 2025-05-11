#!/usr/bin/env python3
import builtins
builtins.input = lambda *args, **kwargs: "n"      

import os
import sys
import json
import time
import random
import torch
import webbrowser
import importlib.util
import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from PyQt5 import QtWidgets, QtCore, QtGui
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState
from core_ppo_policy import PPOPolicy

# Default-Pfad zur example.py im selben Ordner wie dieses Script
DEFAULT_EXAMPLE = os.path.join(os.path.dirname(__file__), "example.py")
# Buy-me-a-coffee-Link
COFFEE_URL = "https://buymeacoffee.com/simplenik"
# Config für gespeicherte Checkpoint-Ordner
CONFIG_PATH = os.path.expanduser("~/.simple_nik_evaluator_config.json")


class ZeroReward(RewardFunction):
    def reset(self, s): pass
    def get_reward(self, *args): return 0.0


class EloRating:
    def __init__(self, k=32, initial=1200):
        self.k, self.initial = k, initial
        self.ratings = {}

    def rate(self, a, b, wins_a, wins_b):
        Ra = self.ratings.get(a, self.initial)
        Rb = self.ratings.get(b, self.initial)
        total = wins_a + wins_b
        Sa = wins_a / total if total > 0 else 0.5
        Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
        Ra_new = Ra + self.k * (Sa - Ea)
        Rb_new = Rb + self.k * ((1 - Sa) - (1 - Ea))
        self.ratings[a], self.ratings[b] = Ra_new, Rb_new
        return Ra_new, Rb_new


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def infer_env(example_path):
    mod = load_module(example_path, "user_example")
    # finde erste function, die mit "build" beginnt
    for fn in dir(mod):
        if fn.lower().startswith("build"):
            factory = getattr(mod, fn)
            break
    else:
        raise RuntimeError("Keine build_…()-Funktion in example.py gefunden")

    env = factory()
    # observation space
    obs_sp = env.observation_space
    if isinstance(obs_sp, gym.spaces.Box):
        state_dim = int(np.prod(obs_sp.shape))
    else:
        raise RuntimeError(f"Unsupported obs space {obs_sp}")
    # action space
    act_sp = env.action_space
    if isinstance(act_sp, Discrete):
        action_dim = act_sp.n
    elif isinstance(act_sp, MultiDiscrete):
        action_dim = int(np.prod(act_sp.nvec))
    else:
        raise RuntimeError(f"Unsupported action space {act_sp}")
    env.close()
    return factory, state_dim, action_dim, None, None


class EvaluatorGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SimpleNik Elo Rating System")
        self.layout = QtWidgets.QVBoxLayout(self)

        # Obere Buttons: HowTo, Save Folder, Coffee
        top = QtWidgets.QHBoxLayout()
        for label, callback in (
            ("How to Use", self.show_help),
            ("Save Folder", self.save_config),
            ("☕ Buy Me a Coffee", lambda: webbrowser.open(COFFEE_URL))
        ):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(callback)
            top.addWidget(btn)
        top.addStretch()
        self.layout.addLayout(top)

        # Checkpoint-Ordner auswählen
        h1 = QtWidgets.QHBoxLayout()
        self.folder_label = QtWidgets.QLabel("No checkpoint folder")
        h1.addWidget(self.folder_label)
        btn_ck = QtWidgets.QPushButton("Choose Checkpoint Folder")
        btn_ck.clicked.connect(self.choose_folder)
        h1.addWidget(btn_ck)
        self.layout.addLayout(h1)

        # Random sampling Checkbox + Count
        self.chk_random = QtWidgets.QCheckBox("Randomly sample older CPs vs latest")
        self.spin_random = QtWidgets.QSpinBox()
        self.spin_random.setRange(1, 100)
        self.spin_random.setValue(4)
        self.spin_random.setEnabled(False)
        self.chk_random.stateChanged.connect(
            lambda s: self.spin_random.setEnabled(s == QtCore.Qt.Checked)
        )
        hr = QtWidgets.QHBoxLayout()
        hr.addWidget(self.chk_random)
        hr.addWidget(QtWidgets.QLabel("Count:"))
        hr.addWidget(self.spin_random)
        self.layout.addLayout(hr)

        # Manual select: Liste für 2 Checks
        self.layout.addWidget(QtWidgets.QLabel("Select 2 checkpoints:"))
        self.list_ck = QtWidgets.QListWidget()
        self.list_ck.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.layout.addWidget(self.list_ck)

        # Matches per pair
        hm = QtWidgets.QHBoxLayout()
        hm.addWidget(QtWidgets.QLabel("Matches per pair:"))
        self.spin_matches = QtWidgets.QSpinBox()
        self.spin_matches.setRange(1, 1000)
        self.spin_matches.setValue(20)
        hm.addWidget(self.spin_matches)
        self.layout.addLayout(hm)

        # Team size selector
        ht = QtWidgets.QHBoxLayout()
        ht.addWidget(QtWidgets.QLabel("Team size:"))
        self.spin_team = QtWidgets.QSpinBox()
        self.spin_team.setRange(1, 4)
        self.spin_team.setValue(1)
        ht.addWidget(self.spin_team)
        self.layout.addLayout(ht)

        # Layer sizes
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(QtWidgets.QLabel("Policy Layers:"))
        self.edit_policy_layers = QtWidgets.QLineEdit("2048,1024,1024,1024")
        hl.addWidget(self.edit_policy_layers)
        hl.addSpacing(20)
        hl.addWidget(QtWidgets.QLabel("Critic Layers:"))
        self.edit_critic_layers = QtWidgets.QLineEdit("2048,1024,1024,1024")
        hl.addWidget(self.edit_critic_layers)
        self.layout.addLayout(hl)

        # Run Evaluation Button
        self.btn_run = QtWidgets.QPushButton("Run Evaluation")
        self.btn_run.clicked.connect(self.run_evaluation)
        self.layout.addWidget(self.btn_run)

        # Progress Bar + ETA
        self.progress = QtWidgets.QProgressBar()
        self.layout.addWidget(self.progress)
        self.eta_label = QtWidgets.QLabel("")
        self.layout.addWidget(self.eta_label)

        # Log-Fenster
        self.txt = QtWidgets.QPlainTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setMaximumHeight(150)
        self.layout.addWidget(self.txt)

        # Elo-Rating Tracker
        self.elo = EloRating()

        # Dark Theme anwenden
        self.apply_dark_theme()

        # Default example.py laden
        self.example_path = DEFAULT_EXAMPLE
        self.factory_fn, self.state_dim, self.action_dim, _, _ = infer_env(self.example_path)

        # Config laden (Checkpoint-Ordner)
        self.load_config()

    def apply_dark_theme(self):
        app = QtWidgets.QApplication.instance()
        app.setStyle("Fusion")
        p = QtGui.QPalette()
        dark = QtGui.QColor(45, 45, 45)
        disabled = QtGui.QColor(127, 127, 127)
        text = QtGui.QColor(220, 220, 220)
        p.setColor(QtGui.QPalette.Window, dark)
        p.setColor(QtGui.QPalette.WindowText, text)
        p.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        p.setColor(QtGui.QPalette.AlternateBase, dark)
        p.setColor(QtGui.QPalette.ToolTipBase, text)
        p.setColor(QtGui.QPalette.ToolTipText, text)
        p.setColor(QtGui.QPalette.Text, text)
        p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled)
        p.setColor(QtGui.QPalette.Button, dark)
        p.setColor(QtGui.QPalette.ButtonText, text)
        p.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled)
        p.setColor(QtGui.QPalette.Highlight, QtGui.QColor(38, 79, 120))
        p.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.white)
        app.setPalette(p)

    def load_config(self):
        if os.path.isfile(CONFIG_PATH):
            try:
                cfg = json.load(open(CONFIG_PATH))
                ck = cfg.get("checkpoint_folder", "")
                if os.path.isdir(ck):
                    self.ck_dir = ck
                    self.folder_label.setText(ck)
                    self.refresh_list()
            except:
                pass

    def save_config(self):
        data = {}
        if hasattr(self, "ck_dir"):
            data["checkpoint_folder"] = self.ck_dir
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f)
        QtWidgets.QMessageBox.information(self, "Saved", "Configuration saved.")

    def show_help(self):
        QtWidgets.QMessageBox.information(
            self,
            "How to Use",
            "1) Place example.py next to this exe\n"
            "2) Select your checkpoints folder\n"
            "3) Choose random or manual (2)\n"
            "4) Set matches, team size and optional layers\n"
            "5) Run evaluation\n"
            "6) Save config and support via coffee link!",
        )

    def choose_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Checkpoint Folder")
        if not path:
            return
        self.ck_dir = path
        self.folder_label.setText(path)
        self.refresh_list()

    def refresh_list(self):
        self.list_ck.clear()
        for d in sorted(os.listdir(self.ck_dir), key=lambda x: int(x)):
            full = os.path.join(self.ck_dir, d, "PPO_POLICY.pt")
            if os.path.isfile(full):
                self.list_ck.addItem(d)

    def log(self, msg):
        self.txt.appendPlainText(msg)
        self.txt.moveCursor(QtGui.QTextCursor.End)

    def run_evaluation(self):
        self.txt.clear()
        self.progress.setValue(0)
        self.eta_label.setText("")
        if not hasattr(self, "ck_dir") or not hasattr(self, "factory_fn"):
            self.log("Bitte Beispiel (example.py) und Checkpoint-Ordner prüfen")
            return

        # dims neu inferieren nach ausgewählter Team size
        try:
            tmp_env = self.factory_fn(team_size=self.spin_team.value())
        except TypeError:
            tmp_env = self.factory_fn(8, self.spin_team.value())
        obs_sp = tmp_env.observation_space
        if isinstance(obs_sp, gym.spaces.Box):
            self.state_dim = int(np.prod(obs_sp.shape))
        else:
            raise RuntimeError(f"Unsupported obs space {obs_sp}")
        act_sp = tmp_env.action_space
        if isinstance(act_sp, Discrete):
            self.action_dim = act_sp.n
        elif isinstance(act_sp, MultiDiscrete):
            self.action_dim = int(np.prod(act_sp.nvec))
        else:
            raise RuntimeError(f"Unsupported action space {act_sp}")
        tmp_env.close()

        # Checkpoint-Paare bilden
        all_dirs = sorted(
            [d for d in os.listdir(self.ck_dir) if os.path.isdir(os.path.join(self.ck_dir, d))],
            key=lambda x: int(x),
        )
        if self.chk_random.isChecked():
            newest = all_dirs[-1]
            olds = all_dirs[:-1]
            sampled = random.sample(olds, min(self.spin_random.value(), len(olds)))
            pairs = [(old, newest) for old in sampled]
        else:
            sel = [it.text() for it in self.list_ck.selectedItems()]
            if len(sel) != 2:
                self.log("Bitte genau 2 Checkpoints auswählen")
                return
            pairs = [(sel[0], sel[1])]

        imps = []
        for old, new in pairs:
            self.log(f"\n>>> Evaluating {old} vs {new}")
            path_old = os.path.join(self.ck_dir, old, "PPO_POLICY.pt")
            path_new = os.path.join(self.ck_dir, new, "PPO_POLICY.pt")
            old_pol = self.load_policy(path_old)
            new_pol = self.load_policy(path_new)
            tot_old, tot_new = self.eval_pair(old, new, old_pol, new_pol, self.spin_matches.value())
            avg_old = tot_old / self.spin_matches.value()
            avg_new = tot_new / self.spin_matches.value()
            if avg_old == 0:
                imp = 100.0 if avg_new > 0 else 0.0
            else:
                imp = (avg_new - avg_old) / avg_old * 100.0
            imps.append(imp)
            self.log(f" → Improvement: {imp:.1f}%")

        overall = sum(imps) / len(imps) if imps else 0.0
        self.log(f"\n=== Overall average improvement: {overall:.1f}% ===")

    def load_policy(self, path, device="cpu"):
        pol_layers = tuple(int(x) for x in self.edit_policy_layers.text().split(",") if x.strip())
        crit_layers = tuple(int(x) for x in self.edit_critic_layers.text().split(",") if x.strip())
        pol = PPOPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            policy_layer_sizes=pol_layers,
            critic_layer_sizes=crit_layers,
            device=device
        )
        sd = torch.load(path, map_location=device)
        actor_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        pol.actor.net.load_state_dict(actor_sd, strict=True)
        pol.eval()
        return pol

    def eval_pair(self, old, new, old_pol, new_pol, n):
        tot_old = tot_new = 0
        start = time.time()
        for i in range(1, n + 1):
            try:
                env = self.factory_fn(team_size=self.spin_team.value())
            except TypeError:
                env = self.factory_fn(8, self.spin_team.value())
            env._reward_fn = ZeroReward()
            obs = env.reset()
            done = False
            while not done:
                a0 = old_pol.get_action(obs[0])
                a1 = new_pol.get_action(obs[1])
                obs, _, done, _ = env.step([a0, a1])
            gs: GameState = env._prev_state
            tot_old += gs.blue_score
            tot_new += gs.orange_score
            self.log(f"Match {i}/{n} {old} → old={gs.blue_score}, new={gs.orange_score}")
            pct = int(i / n * 100)
            self.progress.setValue(pct)
            elapsed = time.time() - start
            rem = (elapsed / i) * (n - i) if i else 0
            m, s = divmod(int(rem), 60)
            self.eta_label.setText(f"ETA: {m:02d}:{s:02d}")
            QtWidgets.QApplication.processEvents()
        return tot_old, tot_new


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = EvaluatorGUI()
    w.resize(1000, 800)
    w.show()
    sys.exit(app.exec_())
