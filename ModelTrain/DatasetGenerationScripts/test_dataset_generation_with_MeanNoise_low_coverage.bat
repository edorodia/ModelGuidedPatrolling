mkdir "ModelTrain/test_MeanNoise_Shekel_Datasets_low_coverage"

::MeanNoise	-	False	False	none ##############################
start cmd /k ^
    title Command_1                                                                                                                                                             ^& ^
    mkdir "ModelTrain/test_MeanNoise_Shekel_Datasets_low_coverage/MeanNoise_-_False_False_none"                                                                                            ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py   --max_frames 30  --drone_noise MeanNoise   --importance_asv_read none  --name_file 1_val   --N_episodes 50    --set test    ^& ^
    copy ModelTrain\Data\gts_shekel_test_1_val.npy ModelTrain\test_MeanNoise_Shekel_Datasets_low_coverage\MeanNoise_-_False_False_none\gts_shekel_test.npy                               ^& ^
    copy ModelTrain\Data\trajectories_shekel_test_1_val.npy ModelTrain\test_MeanNoise_Shekel_Datasets_low_coverage\MeanNoise_-_False_False_none\trajectories_shekel_test.npy

::MeanNoise	-	True	False	none ############################
start cmd /k ^
    title Command_2                                                                                                                                                                                             ^& ^
    mkdir "ModelTrain/test_MeanNoise_Shekel_Datasets_low_coverage/MeanNoise_-_True_False_none"                                                                                                                             ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py   --max_frames 30  --drone_noise MeanNoise   --influence_drone_visited_map   --importance_asv_read none  --name_file 2_val   --N_episodes 50    --set test    ^& ^
    copy ModelTrain\Data\gts_shekel_test_2_val.npy ModelTrain\test_MeanNoise_Shekel_Datasets_low_coverage\MeanNoise_-_True_False_none\gts_shekel_test.npy                                                                ^& ^
    copy ModelTrain\Data\trajectories_shekel_test_2_val.npy ModelTrain\test_MeanNoise_Shekel_Datasets_low_coverage\MeanNoise_-_True_False_none\trajectories_shekel_test.npy
