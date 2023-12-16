#!/usr/bin/env python3

import tkinter as tk

window = tk.Tk()
window.title("Intrusion Detection")
window.minsize(width=500, height=300)
label = tk.Label(text="A Label", font=("Arial", 25, "bold"))
label.pack()
window.mainloop() # At the end of the loop
