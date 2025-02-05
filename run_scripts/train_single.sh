# ----------------------------------------------------------------------------
# --subject: 'S01' 'S02' 'S03' 'S04' 'S05' 'S06' 'S07' 'S08' 'S09'
# --label_type: 'word' 'initial_class' 'tone_class' 'initial_class_8' 'final_class'
# --aug: None 'time_noise' 'freq_noise'
# --noise_type: 'gaussian' 'poisson' 'pink' 'salt_and_pepper'
# ----------------------------------------------------------------------------
python -u ../src/train_single.py \
--subject 'S01' \
--label_type 'word' \
--aug 'time_noise' \
--noise_type 'gaussian'