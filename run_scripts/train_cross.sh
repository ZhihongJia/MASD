# ----------------------------------------------------------------------------
# --subject: 'S01' 'S02' 'S03' 'S04' 'S05' 'S06' 'S07' 'S08' 'S09'
# --wav_type: 'mel' 'wav2vec' 'hubert'
# --word_type: 'fasttext' 'bert'
# ----------------------------------------------------------------------------
python -u ../src/train_cross.py \
--subject 'S01' \
--wav_type 'mel' \
--word_type 'fasttext' 