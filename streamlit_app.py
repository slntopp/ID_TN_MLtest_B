## -------------------------------
## ====    Import smthng      ====
## -------------------------------

##Libraries for Streamlit
##--------------------------------
import streamlit as st
import altair as alt
import pydub
# import librosa.display
import librosa
import io
from scipy.io import wavfile
from PIL import Image

##Libraries for prediction
##--------------------------------
# import os
# import pathlib
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from IPython.display import Audio, display
# from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.keras import models
# import tensorflow_datasets as tfds
# import pickle



## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


## Page decorations
##--------------------------------


id_logo = Image.open("TypoMeshDarkFullat3x.png")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(id_logo)

st.markdown("<h1 style='text-align: center; color: grey;'>ML Audio Recognition App</h1>", 
            unsafe_allow_html=True)
st.header(" ")
st.header(" ")
# st.title('ML Audio recognition App :sunglasses:')
# st.write('Welcome')





## -------------------------------
## ====  Select and load data ====
## -------------------------------
# st.header("Select data to analyze")
st.markdown("<h2 style='text-align: center; color: grey;'>Select data to analyze</h2>", 
            unsafe_allow_html=True)


st.subheader("Select one of the samples")
# st.write('selector will be implemented')
selected_provided_file = st.selectbox(label="", 
                            options=["example of a cutting event", "example of a background sound"]
                            )


st.subheader("or Upload an audio file in WAV format")
st.write("if a file is uploaded, previously selected samples are not taken into account")

uploaded_audio_file = st.file_uploader(label="Select a short WAV file < 5 sec", 
                                        type="wav", 
                                        accept_multiple_files=False, 
                                        key=None, 
                                        help=None, 
                                        on_change=None, 
                                        args=None, 
                                        kwargs=None, 
                                        disabled=False)


def handle_uploaded_audio_file(uploaded_file):
    audio_dub = pydub.AudioSegment.from_file(
            file=uploaded_audio_file,
            format=uploaded_audio_file.name.split(".")[-1]
            )

    channel_sounds = audio_dub.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr1 = np.array(samples).T#.astype(np.float32)
    fp_arr2 = fp_arr1 /  np.iinfo(samples[0].typecode).max
    fp_arr3 = fp_arr2.astype(np.float32)

    audio_arr_f = fp_arr3[:, 0]
    audio_arr_sr_f = audio_dub.frame_rate
    return audio_arr_f, audio_arr_sr_f




## Data Switch is here
##--------------------------------
if uploaded_audio_file is not None:
    # st.write("YEP")
    audio_arr, audio_arr_sr = handle_uploaded_audio_file(uploaded_audio_file)
    # st.audio(uploaded_audio_file, format='audio/wav')
else:
    # st.write("NOPE")
    if selected_provided_file == "example of a cutting event":
        audio_arr, audio_arr_sr = librosa.load('03-CM01B_Vorne.wav', sr=48000)
    if selected_provided_file == "example of a background sound":
        audio_arr, audio_arr_sr = librosa.load('04-Schlitzen_am_LKW.wav', sr=44100)
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=audio_arr_sr, data=audio_arr)
    uploaded_audio_file = virtualfile
    # st.audio(virtualfile, format='audio/wav')

## for debugging
# st.code(audio_arr)
# st.code(audio_arr_sr)





## -------------------------------
## ====   Show selected data  ====
## -------------------------------

# st.subheader("Show the data selected for analysis")
st.header(" ")
st.header(" ")
st.markdown("<h2 style='text-align: center; color: grey;'>Show the data selected for analysis</h2>", 
            unsafe_allow_html=True)

# st.write("Listen the loaded data")
st.markdown(" ##### _Listen the loaded data_")
st.audio(uploaded_audio_file, format='audio/wav')
# st.write("Waveform of the loaded data")
st.markdown(" ##### _Waveform of the loaded data_")
st.line_chart(audio_arr)


# ----------------------------------------
# ==== Functions to make spectrograms ====
# ----------------------------------------

def get_spectrogram( waveform, sampling_rate ):
    waveform_1d = tf.squeeze(waveform)
    waveform_1d_shape = tf.shape(waveform_1d)
    n_samples  = waveform_1d_shape[0]
    spectrogram = tf.signal.stft(
                        tf.squeeze(waveform),
                        frame_length=tf.cast(n_samples/100, dtype=tf.int32),
                        frame_step=tf.cast(n_samples/100/4, dtype=tf.int32),
                        )
    spectrogram = tf.abs(spectrogram)
    l2m = tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins=125,
                        num_spectrogram_bins=tf.shape(spectrogram)[1],
                        sample_rate=sampling_rate,
                        lower_edge_hertz=0,
                        upper_edge_hertz=22000,
                        )
    spectrogram = tf.matmul(spectrogram, l2m)
    spectrogram = tf.math.divide(spectrogram, tf.math.reduce_max(spectrogram) )
    spectrogram = tf.math.add(spectrogram, tf.math.reduce_min(spectrogram) )
    spectrogram = tf.math.add(spectrogram, 0.01 )
    spectrogram = tf.math.log(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.transpose(spectrogram, perm=(1,0,2))
    spectrogram = spectrogram[::-1, :, :]
    return spectrogram


spectrogram_shape_to_analyze = (64*2*1, 64*4*1)

def spectrogram_resize(spectrogram):
    return tf.image.resize(spectrogram, spectrogram_shape_to_analyze)



# ----------------------------------------
# ==== Create Dataset of spectrograms ====
# ----------------------------------------


spectrogram_arr = get_spectrogram(audio_arr, audio_arr_sr)
spectrogram_arr_resized = spectrogram_resize(spectrogram_arr)


## Show spectrogram
##--------------------------------
st.markdown(" ##### _Spectrogram of the  loaded data_")
fig_sp, ax_sp = plt.subplots(1,1, figsize=(5, 2))
ax_sp.imshow(spectrogram_arr_resized)
st.pyplot(fig_sp)

# st.image(spectrogram_arr_resized)




## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



## -------------------------------
## ====    Apply ML model     ====
## -------------------------------
# st.header("Analysis with ML model")
st.header(" ")
st.header(" ")
st.markdown("<h2 style='text-align: center; color: grey;'>Analysis with ML model</h2>", 
            unsafe_allow_html=True)
# st.subheader("Select a model")
# st.subheader("Predict using selected model")




# ----------------------------------
# ==== Load ML model and see it ====
# ----------------------------------

## Load the model
##--------------------------------
reloaded_model = tf.keras.models.load_model("./tf_models/modelTN2/modelTN2")

## Check model architecture
##--------------------------------
model_summary_stringlist = []
reloaded_model.summary(print_fn=lambda x: model_summary_stringlist.append(x))
short_model_summary = "\n".join(model_summary_stringlist)
# print(short_model_summary)
st.markdown(" ##### _ML model architecture_")
st.code(body=short_model_summary, language="Python")



# -----------------------------------
# ==== Predict with loaded Model ====
# -----------------------------------

y_pred_1 = reloaded_model.predict(np.expand_dims(spectrogram_arr_resized, 0))
audio_data_predicted_label = 1 - np.round(y_pred_1[0,0], decimals=2) #1-val bcoz model trained as 0=event, 1=bkg

print(f"Predicted Label: {audio_data_predicted_label}")
st.subheader("Prediction:")

pred_index = np.round(audio_data_predicted_label, decimals=0).astype(int)
results_options = ['No cutting sound detected.',
                    'Canvas cutting is DETECTED.']

st.markdown(f"### _{results_options[pred_index]}_")
# st.write(f"Result: {results_options[pred_index]}")

