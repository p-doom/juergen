from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def test_vllm_tree_is_drained_before_teacher_forced_gpu_load():
    stage = (HERE / "typing_eval_stage.sh").read_text()
    generation = stage.index('"$PY" "${A[experiment_dir]}/typing_evaluate.py"')
    shutdown = stage.index("shutdown_vllm\n", generation)
    teacher = stage.index('"$PY" "${A[experiment_dir]}/typing_teacher_forced.py"')

    assert "setsid --wait uv run --no-sync vllm serve" in stage
    assert 'kill -TERM -- "-$VLLM_PID"' in stage
    assert 'kill -KILL -- "-$VLLM_PID"' in stage
    assert 'kill -0 -- "-$VLLM_PID"' in stage
    assert generation < shutdown < teacher


def test_teacher_forced_load_is_explicitly_bfloat16():
    source = (HERE / "typing_teacher_forced.py").read_text()
    load = source.index("Qwen3VLForConditionalGeneration.from_pretrained(")
    move = source.index(').to(torch.device("cuda:0")).eval()', load)
    assert "dtype=torch.bfloat16" in source[load:move]
