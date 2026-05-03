import json
import torch
import data
from transformer import TransformerLM
import attention
import matplotlib.pyplot as plt
import seaborn as sns

def load_environment():
    with open("config.json", "r") as f:
        config = json.load(f)
    print("1. Chargement du Tokenizer...")
    try:
        tokenizer = data.CharTokenizer.load(config["load_tokenizer_path"])
    except FileNotFoundError:
        raise Exception(f"Erreur: Tokenizer introuvable ({config['load_tokenizer_path']}). Entraînez le modèle d'abord.")
        
    print("2. Chargement du Modèle (peut prendre quelques secondes)...")
    try:
        model = TransformerLM.load(config["load_model_path"])
    except FileNotFoundError:
        raise Exception(f"Erreur: Modèle introuvable ({config['load_model_path']}). Entraînez le modèle d'abord.")
        
    model.eval()
    return config, tokenizer, model

def analyze_text(text: str, model: TransformerLM, tokenizer: data.CharTokenizer):
    # Activer l'espion d'attention
    attention.ATTENTION_CACHE['active'] = True
    attention.ATTENTION_CACHE['matrices'] = []
    
    tokens = tokenizer.tokenize(text)
    # L'entrée doit être dimensionnée [Batch, Sequence]
    tensor = torch.tensor([tokens], dtype=torch.int32)
    
    print(f"\nPassage du texte: '{text}' (Longueur: {len(tokens)} caractères)")
    with torch.no_grad():
        _ = model(tensor)
        
    matrices = attention.ATTENTION_CACHE['matrices']
    
    n_layers = len(model.layers)
    n_heads = model.layers[0].attention.n_heads
    print(f"-> Architecture: {n_layers} Couches x {n_heads} Têtes = {n_layers * n_heads} matrices")
    print(f"-> {len(matrices)} matrices d'attention capturées avec succès !")
    
    # Réorganiser les matrices sous forme de grille organisée matrices[Couche][Tête]
    organized_matrices = []
    idx = 0
    for l in range(n_layers):
        layer_list = []
        for h in range(n_heads):
            # La forme initiale est (Batch, N, N). On prend la première ligne de batch
            mat = matrices[idx][0].numpy()
            layer_list.append(mat)
            idx += 1
        organized_matrices.append(layer_list)
        
    # Désactiver l'espion pour ne pas polluer plus tard
    attention.ATTENTION_CACHE['active'] = False
    return tokens, organized_matrices

def plot_heatmap(layer, head, text_str, tokens, organized_matrices, tokenizer):
    matrix = organized_matrices[layer][head]
    
    # Obtenir un beau format visuel pour les lettres
    chars = []
    for t in tokens:
        char = tokenizer.detokenize([t])
        if char == " ":
            char = "[_]" # Visualiser explicitement l'espace
        chars.append(char)
    
    plt.figure(figsize=(10, 8))
    # seaborn génère une cartographie précise
    sns.heatmap(matrix, xticklabels=chars, yticklabels=chars, cmap='magma')
    
    plt.title(f"Activité d'Attention - Couche {layer} | Tête {head}\nTexte: '{text_str}'", fontsize=14, pad=20)
    plt.xlabel("Caractère Lu (La Cible à qui l'on pose la question)")
    plt.ylabel("Caractère Courant (Celui qui pose la question)")
    
    filename = f"heatmap_L{layer}_H{head}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"\n✅ Carte de chaleur magifiquement générée: {filename}")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*50)
    print("   BOÎTE À OUTILS D'INTERPRÉTABILITÉ (PARTIE 5)")
    print("="*50 + "\n")
    
    config, tokenizer, model = load_environment()
    
    # ==========================================
    #             MISE EN PRATIQUE
    # ==========================================
    # Modifiez ce mot pour tester le comportement sur d'autres contextes !
    texte_test = "Hello world!"
    
    tokens, matrices = analyze_text(texte_test, model, tokenizer)
    
    # ------------------------------------------
    # On trace la carte thermique. 
    # (Amusez-vous à changer de Layer de 0 à 5, et de Head de 0 à 5)
    # ------------------------------------------
    plot_heatmap(layer=0, head=2, text_str=texte_test, tokens=tokens, organized_matrices=matrices, tokenizer=tokenizer)
    
    print("\nAmusez-vous à utiliser 'plot_heatmap' pour chercher une tête qui :")
    print("- Regarde toujours le caractère précédent")
    print("- Associe les majuscules aux espaces.")
