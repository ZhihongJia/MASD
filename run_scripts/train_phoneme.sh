# ----------------------------------------------------------------------------
# --subject: 'S01' 'S02' 'S03' 'S04' 'S05' 'S06' 'S07' 'S08' 'S09'
# --label_type: 'initial_class' 'tone_class' 'initial_class_8' 'final_class'
# --wav_type: 'mel' 'wav2vec' 'hubert'
# ----------------------------------------------------------------------------
python -u ../src/train_phoneme.py \
--subject 'S01' \
--label_type 'initial_class' \
--wav_type 'mel'