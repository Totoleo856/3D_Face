import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from processor import process_video_with_callback
import overlay_generator  # le script séparé


def browse_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mov;*.avi")])
    if path:
        video_entry.delete(0, tk.END)
        video_entry.insert(0, path)

def browse_output_folder():
    path = filedialog.askdirectory()
    if path:
        output_entry.delete(0, tk.END)
        output_entry.insert(0, path)

def progress_callback(value):
    progress_var.set(value)

def start_processing():
    video_path = video_entry.get()
    output_folder = output_entry.get()
    if not video_path or not output_folder:
        messagebox.showwarning("Attention", "Veuillez remplir tous les champs.")
        return
    progress_var.set(0)

    fast_mode = fast_mode_var.get()
    use_one_euro = one_euro_var.get()
    use_kalman = kalman_var.get()
    min_cutoff = min_cutoff_var.get()
    beta = beta_var.get()

    def thread_func():
        try:
            process_video_with_callback(
                video_path, output_folder, progress_callback,
                fast_mode=fast_mode,
                use_one_euro=use_one_euro,
                one_euro_min_cutoff=min_cutoff,
                one_euro_beta=beta,
                use_kalman=use_kalman
            )
        except Exception as e:
            print(f"Erreur : {e}")

    threading.Thread(target=thread_func, daemon=True).start()

def create_overlay_button_callback():
    video_path = video_entry.get()
    output_folder = output_entry.get()
    if not video_path or not output_folder:
        messagebox.showwarning("Attention", "Veuillez remplir tous les champs.")
        return

    def thread_func():
        try:
            overlay_path = overlay_generator.generate_overlay(video_path, output_folder)
            messagebox.showinfo("Terminé", f"Overlay généré : {overlay_path}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    threading.Thread(target=thread_func, daemon=True).start()

# --- GUI ---
root = tk.Tk()
root.title("Face Mesh Processor")

# Video / output
tk.Label(root, text="Vidéo source:").grid(row=0, column=0, sticky="e")
video_entry = tk.Entry(root, width=50)
video_entry.grid(row=0, column=1)
tk.Button(root, text="Parcourir", command=browse_video).grid(row=0, column=2)

tk.Label(root, text="Dossier parent:").grid(row=1, column=0, sticky="e")
output_entry = tk.Entry(root, width=50)
output_entry.grid(row=1, column=1)
tk.Button(root, text="Parcourir", command=browse_output_folder).grid(row=1, column=2)

# Filters
one_euro_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Activer OneEuroFilter", variable=one_euro_var).grid(row=2, column=0, columnspan=2, sticky="w")

kalman_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Activer Kalman Filter", variable=kalman_var).grid(row=3, column=0, columnspan=2, sticky="w")

fast_mode_var = tk.BooleanVar(value=False)
tk.Checkbutton(root, text="Mode FAST (preview uniquement)", variable=fast_mode_var).grid(row=4, column=0, columnspan=2, sticky="w")

# OneEuro sliders
tk.Label(root, text="OneEuro min_cutoff").grid(row=5, column=0, sticky="e")
min_cutoff_var = tk.DoubleVar(value=1.0)
tk.Scale(root, variable=min_cutoff_var, from_=0.0, to=5.0, resolution=0.1, orient="horizontal").grid(row=5, column=1, sticky="we")

tk.Label(root, text="OneEuro beta").grid(row=6, column=0, sticky="e")
beta_var = tk.DoubleVar(value=0.005)
tk.Scale(root, variable=beta_var, from_=0.0, to=0.05, resolution=0.001, orient="horizontal").grid(row=6, column=1, sticky="we")

# Launch button
tk.Button(root, text="Lancer le traitement", command=start_processing, bg="green", fg="white").grid(row=7, column=0, columnspan=3, pady=10)

# Progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.grid(row=8, column=0, columnspan=3, sticky="we", padx=10)




# Ajouter le bouton dans le GUI
tk.Button(root, text="Créer Overlay", command=create_overlay_button_callback, bg="blue", fg="white").grid(row=9, column=0, columnspan=3, pady=10)


root.mainloop()
