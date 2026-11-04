# Large-batch NZ up trial

Round7 kept source-oriented prepared BF16 NZ up at tokens<=64: all15 checks
passed and full Graph improved1.0788-1.2399x at the changed acceptance points.
Up tasks are7.38-8.88us there, compared with21.225/21.633us ND at B128/256.

Additional round8 changes only the maximum token count using the already
prepared up NZ buffer: candidate2048, same-binary control64. No new allocation,
native V build, activation conversion, precision threshold or other stage
change. Preparation/refresh follow round7 and stay outside capture.
Run `matrix_trial.py --up-large-control` with the existing fixed15 cases and
full Graph20 AB/BA pairs, then fresh fused pairs and complete serial profile.
Keep large-batch NZ only if current B128/256 full Graph improves with unchanged
correctness and no extra execution conversion task; otherwise reject the range.
