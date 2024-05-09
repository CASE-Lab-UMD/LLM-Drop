##############################################################################
MODEL=llama-2-7b

# run AWQ search (optional; we provided the pre-computed results)
python awq/entry.py \
    --model_path $MODEL \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq ../awq_cache/$MODEL-w4-g128.pt
