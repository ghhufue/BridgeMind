# SL v3 rule-bidding 3-hour training backup

This branch contains actual binary checkpoints, logs, frozen source, data and SHA256 manifests.
Clone this branch and run `python restore.py --destination /root/autodl-tmp/BridgeMind`.
Recreate the recorded Python/CUDA environment before using --resume.
No SSH credentials or private deploy keys are included.
