Please reference the environment setup in:
./requirements.txt

All script are under ./scripts

Script config setup:
(1) Select the optimizer algo, set optim_cfg=dual_adam
(2) Control the dual loss optimizer rate, eg:
forget_coeff=0.1
regularization_coeff=1.0
(3) To get the eval result at all epoch, refer to: ./scripts/tofu_phi1-5_eval

Parse result:
parse_output.py to collect from a single task dir
parse_output_multi.py to collect multi-task results get mean/std

