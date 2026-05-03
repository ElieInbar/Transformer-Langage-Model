import tkinter as tk
from tkinter import messagebox
import json
import os
import subprocess

CONFIG_FILE = "config.json"

class AppConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tableau de bord - Transformer LM")
        self.root.geometry("450x650")

        # Load existing config or setup defaults
        self.config = self.load_config()

        # Vars
        self.mode_var = tk.StringVar(value=self.config.get("mode", "train"))
        
        # We determine language from data_path if possible
        lang = "en"
        if "he" in self.config.get("data_path", ""):
            lang = "he"
        self.lang_var = tk.StringVar(value=lang)

        self.batches_var = tk.IntVar(value=self.config.get("num_batches", 50000))
        self.temp_var = tk.DoubleVar(value=self.config.get("temperature", 1.0))
        self.topk_var = tk.IntVar(value=self.config.get("topK", 5))

        # Variables for Interpretability Lab
        self.analysis_text_var = tk.StringVar(value="Hello world!")
        self.analysis_layer_var = tk.IntVar(value=0)
        self.analysis_head_var = tk.IntVar(value=2)

        self.create_widgets()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "mode": "train",
            "data_path": "../data/en/",
            "save_model_path": "model_en.pt",
            "load_model_path": "model_en.pt",
            "save_tokenizer_path": "tokenizer_en.json",
            "load_tokenizer_path": "tokenizer_en.json",
            "num_batches": 50000,
            "seq_len": 128,
            "batch_size": 64,
            "n_layers": 6,
            "n_heads": 6,
            "embed_size": 192,
            "learning_rate": 0.0005,
            "gradient_clipping": 1.0,
            "temperature": 1.0,
            "topK": 5
        }

    def create_widgets(self):
        # --- Mode ---
        tk.Label(self.root, text="Sélecteur de Mode", font=("Arial", 12, "bold")).pack(pady=5)
        mode_frame = tk.Frame(self.root)
        mode_frame.pack()
        tk.Radiobutton(mode_frame, text="Apprentissage (Train)", variable=self.mode_var, value="train").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Génération (Eval)", variable=self.mode_var, value="eval").pack(side=tk.LEFT)

        # --- Langue ---
        tk.Label(self.root, text="Sélecteur de Langue", font=("Arial", 12, "bold")).pack(pady=5)
        lang_frame = tk.Frame(self.root)
        lang_frame.pack()
        tk.Radiobutton(lang_frame, text="Anglais (en)", variable=self.lang_var, value="en").pack(side=tk.LEFT)
        tk.Radiobutton(lang_frame, text="Hébreu (he)", variable=self.lang_var, value="he").pack(side=tk.LEFT)

        # --- Configuration Apprentissage ---
        tk.Label(self.root, text="Configuration d'Apprentissage", font=("Arial", 12, "bold")).pack(pady=5)
        train_frame = tk.Frame(self.root)
        train_frame.pack()
        tk.Label(train_frame, text="Nombre de lots :").grid(row=0, column=0, padx=5)
        tk.Entry(train_frame, textvariable=self.batches_var, width=10).grid(row=0, column=1, padx=5)

        # --- Configuration Échantillonnage ---
        tk.Label(self.root, text="Créativité & Échantillonnage", font=("Arial", 12, "bold")).pack(pady=10)
        samp_frame = tk.Frame(self.root)
        samp_frame.pack()
        
        tk.Label(samp_frame, text="Température :").grid(row=0, column=0, padx=5, pady=2)
        tk.Entry(samp_frame, textvariable=self.temp_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        tk.Label(samp_frame, text="Top-K :").grid(row=1, column=0, padx=5, pady=2)
        tk.Entry(samp_frame, textvariable=self.topk_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # --- Boutons ---
        tk.Frame(self.root, height=10).pack() # espacement
        
        btn_save = tk.Button(self.root, text="1. Sauvegarder Config", command=self.save_config, bg="lightblue")
        btn_save.pack(pady=2, fill=tk.X, padx=50)

        btn_run = tk.Button(self.root, text="🚀 2. Lancer (main.py)", command=self.run_main, bg="lightgreen", font=("Arial", 12, "bold"))
        btn_run.pack(pady=2, fill=tk.X, padx=50)
        
        # --- Laboratoire d'Interprétabilité ---
        tk.Frame(self.root, height=1, bg="black").pack(fill=tk.X, pady=10, padx=20)
        tk.Label(self.root, text="🔬 Laboratoire d'Interprétabilité (Part 5)", font=("Arial", 12, "bold", "italic")).pack(pady=5)
        
        lab_frame = tk.Frame(self.root)
        lab_frame.pack()
        
        tk.Label(lab_frame, text="Texte :").grid(row=0, column=0, padx=5, pady=2)
        tk.Entry(lab_frame, textvariable=self.analysis_text_var, width=15).grid(row=0, column=1, padx=5, pady=2, columnspan=3)

        tk.Label(lab_frame, text="Couche :").grid(row=1, column=0, padx=5, pady=2)
        tk.Entry(lab_frame, textvariable=self.analysis_layer_var, width=5).grid(row=1, column=1, padx=5, pady=2)
        tk.Label(lab_frame, text="Tête :").grid(row=1, column=2, padx=5, pady=2)
        tk.Entry(lab_frame, textvariable=self.analysis_head_var, width=5).grid(row=1, column=3, padx=5, pady=2)

        btn_analyze = tk.Button(self.root, text="👁️ Générer la Carte de Chaleur", command=self.run_analysis, bg="plum")
        btn_analyze.pack(pady=10, fill=tk.X, padx=50)

    def save_config(self):
        mode = self.mode_var.get()
        lang = self.lang_var.get()
        
        try:
            batches = self.batches_var.get()
            temp = self.temp_var.get()
            topk = self.topk_var.get()
        except ValueError:
            messagebox.showwarning("Attention", "Valeurs numériques invalides.")
            return False

        # update config based on language selection smartly
        self.config["mode"] = mode
        self.config["data_path"] = f"../data/{lang}/"
        self.config["save_model_path"] = f"model_{lang}.pt"
        self.config["load_model_path"] = f"model_{lang}.pt"
        self.config["save_tokenizer_path"] = f"tokenizer_{lang}.json"
        self.config["load_tokenizer_path"] = f"tokenizer_{lang}.json"
        self.config["num_batches"] = batches
        self.config["temperature"] = temp
        self.config["topK"] = topk

        # write to disk
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
            return True
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de sauvegarde: {e}")
            return False

    def run_main(self):
        if self.save_config():
            print("\n" + "="*50)
            print("🚀 Tableau de bord: Démarrage de main.py...")
            print("="*50 + "\n")
            os.system("python3 main.py") # we use os.system to see output in terminal
            
    def run_analysis(self):
        from tkinter import messagebox
        import analyze_attention
        
        # S'assurer que le dernier modèle ciblé est bien sélectionné
        self.save_config()
        
        texte = self.analysis_text_var.get()
        try:
            couche = self.analysis_layer_var.get()
            tete = self.analysis_head_var.get()
        except ValueError:
            messagebox.showerror("Erreur", "La couche et la tête doivent être des nombres entiers.")
            return

        if not texte.strip():
            messagebox.showerror("Erreur", "Veuillez entrer un texte à analyser.")
            return

        try:
            print("Chargement de l'environnement d'analyse...")
            config, tokenizer, model = analyze_attention.load_environment()
            
            n_layers = len(model.layers)
            n_heads = model.layers[0].attention.n_heads
            
            if couche < 0 or couche >= n_layers:
                messagebox.showerror("Erreur", f"Couche invalide. Le modèle possède des couches de 0 à {n_layers-1}.")
                return
            if tete < 0 or tete >= n_heads:
                messagebox.showerror("Erreur", f"Tête invalide. Le modèle possède des têtes de 0 à {n_heads-1}.")
                return
                
            tokens, matrices = analyze_attention.analyze_text(texte, model, tokenizer)
            
            # Afficher matplotlib 
            analyze_attention.plot_heatmap(layer=couche, head=tete, text_str=texte, tokens=tokens, organized_matrices=matrices, tokenizer=tokenizer)
            
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            

if __name__ == "__main__":
    root = tk.Tk()
    app = AppConfigGUI(root)
    root.mainloop()
