import customtkinter as ctk
import numpy as np
import librosa
import sounddevice as sd
import customtkinter as ctk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import threading
import tkinter as tk

# Zmienne globalne na dane audio
y, sr = None, None

#Podstawowe ustawienia do GUI
gui = ctk.CTk()
ctk.set_appearance_mode("dark")
gui.geometry("800x900")
gui.title("Analiza dźwięków Kalimby")
gui.resizable(True, True)

gui.grid_columnconfigure(0, weight=1)
gui.grid_rowconfigure((0, 1, 2,3,4), weight=1)

czcionka="Consolas"

#Ustawienia ramki 1 (analiza z pliku audio)
ramka1 = ctk.CTkFrame(gui, width=800, height=200, corner_radius=30, fg_color="#272727", border_width=1,
                         border_color="dimgrey")
ramka1.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="nsew")
ramka1.grid_propagate(False)

ramka1.grid_columnconfigure((0,1), weight=1)
ramka1.grid_rowconfigure((0,1,2,3,4), weight=1)

#Zmieniające się labels (wstępne ustawienia)
label_audio_info = ctk.CTkLabel(gui, text="", font=(czcionka, 12))

label_analysis_result = ctk.CTkLabel(gui, text="", font=(czcionka, 14))
label_analysis_result.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

label_plik=ctk.CTkLabel(ramka1, text="Brak załącznika", font=(czcionka, 15))
label_plik.grid(row=2, column=0, columnspan=2, sticky="", padx=10, pady=5)

#Konsola w GUI
ctk.CTkLabel(gui, text="Konsola", font=(czcionka, 22)).grid(row=3, column=0, columnspan=2, sticky="", padx=10, pady=(20,0))
text_logs = ctk.CTkTextbox(gui, height=200, width=600, corner_radius=30,
          fg_color="#272727", text_color="snow", border_width=1, border_color="dimgrey", font=(czcionka, 12))
text_logs.grid(row=4, column=0, padx=20, pady=(0,20), sticky="ew")
text_logs.configure(state="disabled")


#Funkcje
#Funkcja do logów (wyświetla w konsoli w GUI)
def log_message(message):
    """Dodanie wiadomości do logów."""
    text_logs.configure(state="normal")  # Umożliwia edycję logów
    text_logs.insert(ctk.END, message + "\n")
    text_logs.see(ctk.END)
    text_logs.configure(state="disabled")  # Zabezpiecza przed edycją przez użytkownika

#Funkcja do załadowania pliku
def load_audio_file():
    """Funkcja do wgrywania pliku audio."""
    global y, sr
    file_path = filedialog.askopenfilename(
        title="Wybierz plik audio",
        filetypes=(
            ("Wszystkie pliki", "*.*"),
            ("Pliki WAV", "*.wav"),
            ("Pliki MP3", "*.mp3")
        )
    )
    if file_path:
        label_plik.configure(text=f"Wybrany plik: {file_path}")
        log_message(f"Wybrano plik: {file_path}")
        try:
            y, sr = librosa.load(file_path, sr=None)
            if y is not None and sr is not None:
                label_audio_info.configure(
                    text=f"Długość: {len(y) / sr:.2f} s, Próbek: {len(y)}, Częstotliwość próbkowania: {sr} Hz"
                )
                log_message(f"Pomyślnie wczytano plik. Długość: {len(y) / sr:.2f} s, Próbek: {len(y)}, Częstotliwość: {sr} Hz")
            else:
                label_plik.configure(text="Błąd: Nie udało się wczytać danych audio.")
                log_message("Błąd: Nie udało się wczytać danych audio.")
        except Exception as e:
            label_plik.configure(text=f"Błąd podczas ładowania pliku: {str(e)}")
            log_message(f"Błąd podczas ładowania pliku: {str(e)}")

#Info do dźwieków kalimby oraz ich analizy
def generate_kalimba_frequencies():
    """Generowanie częstotliwości dźwięków kalimby strojonej w C-dur."""
    kalimba_notes = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00,
        'A4': 440.00, 'B4': 493.88, 'C5': 523.25, 'D5': 587.33, 'E5': 659.25,
        'F5': 698.46, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77, 'C6': 1046.50,
        'D6': 1174.66, 'E6': 1318.51
    }
    return kalimba_notes

def find_closest_note(freq, kalimba_dict):
    """Znajdowanie najbliższego dźwięku dla danej częstotliwości."""
    closest_note = min(kalimba_dict.keys(), key=lambda note: abs(kalimba_dict[note] - freq))
    return closest_note, kalimba_dict[closest_note]

def analyze_audio(wyniki):
    """Funkcja do analizy FFT i rozpoznawania dźwięku kalimby."""
    global y, sr, positive_frequencies, positive_magnitudes, closest_note
    if y is None or sr is None:
        label_plik.configure(text="Brak wgranego pliku audio!")
        log_message("Błąd: Brak wgranego pliku audio!")
        return

    try:
        # Obliczenie FFT
        fft = np.fft.fft(y)
        frequencies = np.fft.fftfreq(len(fft), 1 / sr)
        magnitudes = np.abs(fft)

        # Filtrujemy tylko częstotliwości dodatnie
        positive_frequencies = frequencies[:len(frequencies) // 2]
        positive_magnitudes = magnitudes[:len(magnitudes) // 2]

        # Znajdujemy dominującą częstotliwość
        dominant_frequency = positive_frequencies[np.argmax(positive_magnitudes)]

        # Dopasowanie do najbliższego dźwięku kalimby
        kalimba_frequencies = generate_kalimba_frequencies()
        closest_note, note_freq = find_closest_note(dominant_frequency, kalimba_frequencies)

        # Wyświetlenie wyników
        if wyniki is wyniki2:
            wyniki2.configure(
                text=f"\n\n‧ ⋆⭒༺✩༻⭒⋆ ‧\nDominująca częstotliwość: {dominant_frequency:.2f} Hz\n"
                     f"Najbliższy dźwięk: {closest_note} ({note_freq:.2f} Hz)"
            )
            # Dodanie slidera
            slider = ctk.CTkSlider(ramka2, from_=0, to=1400, orientation="horizontal", width=800, corner_radius=30,
                                   height=25,
                                   state="disabled", button_color='dimgrey', border_width=2, border_color="dimgrey")
            slider.set(dominant_frequency)  # Ustawienie początkowej wartości na dominant_frequency
            slider.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

            def label_slider(Hz, column, sticky, columnspan):
                ctk.CTkLabel(ramka2, text=Hz, font=(czcionka, 15)).grid(row=3, column=column, sticky=sticky, columnspan=columnspan, padx=10, pady=(0, 10))

            label_slider("0 Hz",  0, "nw", 1)
            label_slider("1400 Hz",  1, "ne", 1)
            label_slider("350 Hz",  0, "n", 1)
            label_slider("1050 Hz",  1, "n", 1)
            label_slider("700 Hz", 0, "n", 2)

            #Funkcja do aktualizacji slidera
            def update_slider(value):
                slider.set(value)

            update_slider(dominant_frequency)

        elif wyniki is wyniki1:
            wyniki1.configure(
                text=f"‧ ⋆⭒༺✩༻⭒⋆ ‧\nDominująca częstotliwość: {dominant_frequency:.2f} Hz\n"
                     f"Najbliższy dźwięk: {closest_note} ({note_freq:.2f} Hz)"
            )


        log_message(f"Dominująca częstotliwość: {dominant_frequency:.2f} Hz. Najbliższy dźwięk: {closest_note} ({note_freq:.2f} Hz)")
    except Exception as e:
        label_analysis_result.configure(text=f"Błąd podczas analizy: {str(e)}")
        log_message(f"Błąd podczas analizy: {str(e)}")

#Funkcja do odczytu na żywo
def live_sound_recognition():
    """Rozpoznawanie dźwięku na żywo."""
    duration = 5  # Czas nagrywania w sekundach
    log_message("Rozpoczynanie nagrywania dźwięku na żywo...")
    try:
        audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
        sd.wait()  # Czekaj na zakończenie nagrywania
        global y, sr
        y = audio_data.flatten()
        sr = 44100
        label_audio_info.configure(
            text=f"Nagranie na żywo: Długość: {duration} s, Próbek: {len(y)}, Częstotliwość próbkowania: {sr} Hz"
        )
        log_message(f"Nagranie zakończone. Długość: {duration} s, Próbek: {len(y)}, Częstotliwość: {sr} Hz")
        analyze_audio()  # Automatyczna analiza po nagraniu
    except Exception as e:
        label_audio_info.configure(text=f"Błąd podczas nagrywania: {str(e)}")
        log_message(f"Błąd podczas nagrywania: {str(e)}")


recording = False

def start_live_sound_recognition():
    """Rozpoczyna nagrywanie dźwięku na żywo w pętli."""
    global recording
    recording = True
    log_message("Rozpoczęto rozpoznawanie dźwięku na żywo.")

    def record_loop():
        while recording:
            try:
                duration = 0.5  # Czas pojedynczego nagrania w sekundach
                audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
                sd.wait()  # Czekaj na zakończenie nagrywania
                global y, sr
                y = audio_data.flatten()
                sr = 44100
                log_message("Zarejestrowano próbkę audio, rozpoczynam analizę.")
                analyze_audio(wyniki2)  # Automatyczna analiza po nagraniu
            except Exception as e:
                log_message(f"Błąd podczas nagrywania: {str(e)}")
                break

    # Uruchomienie wątku dla pętli nagrywania
    threading.Thread(target=record_loop, daemon=True).start()

def stop_live_sound_recognition():
    """Zatrzymuje nagrywanie dźwięku na żywo."""
    global recording
    recording = False
    log_message("Zatrzymano rozpoznawanie dźwięku na żywo.")

def open_new_window():
    global positive_frequencies, positive_magnitudes, closest_note
    if 'positive_frequencies' not in globals() or 'positive_magnitudes' not in globals():
        log_message("Błąd: Brak danych do wyświetlenia wykresu.")
        return

    # Tworzymy nowe okno
    new_window = ctk.CTkToplevel(gui)
    new_window.title("Wykresy")
    new_window.geometry("800x600")

    # Tworzymy wykres Matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(positive_frequencies, positive_magnitudes, label=f'Najwyższa częstotliwość')
    ax.set_title(f"FFT dźwięku {closest_note}")
    ax.set_xlabel("Częstotliwość [Hz]")
    ax.set_ylabel("Amplituda")
    ax.legend()

    # Umieszczamy wykres w oknie za pomocą FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Dodajemy przycisk do zamknięcia okna
    button_close = ctk.CTkButton(new_window, text="Zamknij okno", command=new_window.destroy)
    button_close.pack(side=tk.BOTTOM, pady=10)



#GUI ciąg dalszy
tytul = (ctk.CTkLabel(gui, text="‧ ⋆⭒༺✩༻⭒⋆ ‧\nANALIZA DŹWIĘKÓW KALIMBY\n• *₊°‧✩˚༺🤍༻˚✩‧°₊ * •", font=(czcionka, 25))).grid(row=0, column=0, sticky="", padx=10, pady=10)

#Analiza z pliku audio GUI
wyniki1=ctk.CTkLabel(ramka1, text=(" "), font=(czcionka, 18))
wyniki1.grid(row=3, column=0, columnspan=2,sticky="", padx=10, pady=(0,20))

ctk.CTkLabel(ramka1, text="Analiza z pliku audio", font=(czcionka, 22)).grid(row=0, column=0, columnspan=2, sticky="", padx=10, pady=(20,5))
ctk.CTkButton(ramka1, text='Wgraj plik', font=(czcionka, 18), command=load_audio_file, width=500, height=40, corner_radius=30,
          fg_color="#353535", text_color="snow", border_width=2, border_color="dimgrey", hover_color='darkslategrey').grid(row=1, column=0, padx=20,
                                                                                               pady=10, sticky="")
ctk.CTkButton(ramka1, text='Analizuj', font=(czcionka, 18), command=(lambda: analyze_audio(wyniki1)), width=500, height=40, corner_radius=30,
          fg_color="#353535", text_color="snow", border_width=2, border_color="dimgrey", hover_color='darkslategrey').grid(row=1, column=1, padx=20,
                                                                                               pady=10, sticky="")

#Analiza na żywo GUI
ramka2 = ctk.CTkFrame(gui, corner_radius=30, fg_color="#272727", border_width=1,
                         border_color="dimgrey")
ramka2.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

ramka2.grid_columnconfigure((0,1), weight=1)
ramka2.grid_rowconfigure((0,1,2,3,4), weight=1)

ctk.CTkLabel(ramka2, text="Analiza na żywo", font=(czcionka, 22)).grid(row=0, column=0, columnspan=2, sticky="", padx=10, pady=(20,5))

ctk.CTkButton(ramka2, text='Start', font=(czcionka, 18), command=start_live_sound_recognition, width=200, height=40, corner_radius=30,
          fg_color="#385b36", text_color="snow", border_width=2, border_color="dimgrey", hover_color='#233b22').grid(row=1, column=0, padx=20,
                                                                                               pady=10, sticky="e")
ctk.CTkButton(ramka2, text='Stop', font=(czcionka, 18), command=stop_live_sound_recognition, width=200, height=40, corner_radius=30,
          fg_color="#5b3d36", text_color="snow", border_width=2, border_color="dimgrey",hover_color='#3a2622').grid(row=1, column=1, padx=20,
                                                                                               pady=10, sticky="w")
button_open_window = ctk.CTkButton(gui, text="Wyświetl wykresy", command=open_new_window)
button_open_window.grid(row=5, column=0, pady=20)

wyniki2=ctk.CTkLabel(ramka2, text=(""), font=(czcionka, 18))
wyniki2.grid(row=3, column=0, columnspan=2,sticky="", padx=10, pady=(0,20))


gui.mainloop()