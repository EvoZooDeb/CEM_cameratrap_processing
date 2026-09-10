• Several advertised two-stage configurations cannot load, and video detections are never classified because frames are neither saved nor resolved correctly. The grouped best_28 detector also
  applies incompatible classifiers to half of its detections.

  Full review comments:

  - [P1] Load mdv6 and DeepFaune with compatible backends — /home/gabor/Documents/motolla/hoi/kameracsapda/CEM_cameratrap_processing/felis_cli/core.py:135-138
    When mdv6 or deepfaune is selected, these mapped checkpoints are passed directly to Ultralytics YOLO; the supplied MegaDetector checkpoint is a YOLOv5 checkpoint and the DeepFaune
    checkpoint contains unsupported custom types, so current unconstrained Ultralytics installations raise TypeError during construction and two advertised detector options cannot run. Use
    their compatible PytorchWildlife loaders or provide supported Ultralytics exports.

  - [P1] Enable image saving when requesting video frames — /home/gabor/Documents/motolla/hoi/kameracsapda/CEM_cameratrap_processing/felis_cli/core.py:164-165
    For two-stage video input, setting save_frames=True while passing save=False does not create frame images in Ultralytics, because its frame writer is only invoked through the normal save
    path. Classification therefore finds labels but no source frames, skips every video detection, and aggregation falls back to unrelated single-stage class names; enable saving or extract
    the required frames explicitly.

  - [P1] Preserve the media stem when resolving saved frames — /home/gabor/Documents/motolla/hoi/kameracsapda/CEM_cameratrap_processing/felis_cli/core.py:317-322
    Even when video frames already exist, Ultralytics names them <stem>_frames/<stem>_<frame>.jpg, matching label files named <stem>_<frame>.txt. Removing <stem>_ here makes the lookup search
    for <frame>.jpg, so _source_for_label returns None and all video classifications are skipped.

  - [P1] Dispatch best_28 groups to their matching classifiers — /home/gabor/Documents/motolla/hoi/kameracsapda/CEM_cameratrap_processing/felis_cli/core.py:198-203
    For best_28, class 0 is Artiodactyla and class 3 is Carnivora, but both are placed in the same animal set and sent through the single configured classifier. Selecting 2_artiodactyla
    consequently assigns artiodactyl species to carnivore crops, while selecting 2_carnivora does the reverse; dispatch each detector class to its corresponding model or exclude the unmatched
    group.

  - [P2] Honor the configured device for Keras classifiers — /home/gabor/Documents/motolla/hoi/kameracsapda/CEM_cameratrap_processing/felis_cli/core.py:253-259
    When a Keras classifier is selected, cfg.predict.device is never applied while loading or invoking TensorFlow. On GPU hosts, --device cpu can still allocate the default GPU, and --device
    cuda:1 does not select GPU 1, potentially causing contention or out-of-memory failures; configure TensorFlow visibility/device placement consistently with the CLI setting.

  - [P2] Expose device selection on the classify command — /home/gabor/Documents/motolla/hoi/kameracsapda/CEM_cameratrap_processing/felis_cli/commands/classify.py:22-27
    Standalone classification constructs DeepFaune using cfg.predict.device, which defaults to cuda:0, but this parser does not accept --device. Consequently felis classify --device cpu is
    rejected and CPU-only users must modify configuration or environment variables instead of using the documented CLI-style override.

  - [P2] Honor cancellation during classification — /home/gabor/Documents/motolla/hoi/kameracsapda/CEM_cameratrap_processing/felis_cli/commands/run.py:138-140
    If SIGTERM or the cancellation file appears during the new classification stage, the signal handler only sets a flag that classify never checks, and execution continues through every crop
    before aggregation. Large camera-trap batches can therefore ignore shutdown for hours and be killed before partial finalization; pass a cancellation callback into classification and
    finalize the completed subset.
