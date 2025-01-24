# ----------------------------------------------------------------------------
# --subject: 'S01' 'S02' 'S03' 'S04' 'S05' 'S06' 'S07' 'S08' 'S09'
# --wav_type: 'mel' 'wav2vec' 'hubert'
# --word_type: 'fasttext' 'bert'
# --aug: None 'time_noise' 'freq_noise'
# ----------------------------------------------------------------------------
python -u ../src/train.py \
--subject 'S01' \
--wav_type 'mel' \
--word_type 'fasttext' \
--aug 'time_noise'