mkdir "ModelTrain/Training_FishEyeNoise_Shekel_Datasets"

::FishEyeNoise	-	False	False	none ##############################
start cmd /k ^
    title Command_1                                                                                                                                                                                                     ^& ^
    mkdir "ModelTrain/Training_FishEyeNoise_Shekel_Datasets/FishEyeNoise_-_False_False_none"                                                                                                                            ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py                                         --drone_noise FishEyeNoise                                      --importance_asv_read none  --name_file 4           ^& ^
    copy ModelTrain\Data\gts_shekel_train_4.npy ModelTrain\Training_FishEyeNoise_Shekel_Datasets\FishEyeNoise_-_False_False_none\gts_shekel_train.npy                                                                   ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_4.npy ModelTrain\Training_FishEyeNoise_Shekel_Datasets\FishEyeNoise_-_False_False_none\trajectories_shekel_train.npy

::FishEyeNoise	-	True	False	none ############################
start cmd /k ^
    title Command_2                                                                                                                                                                                                     ^& ^
    mkdir "ModelTrain/Training_FishEyeNoise_Shekel_Datasets/FishEyeNoise_-_True_False_none"                                                                                                                             ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py                                         --drone_noise FishEyeNoise      --influence_drone_visited_map   --importance_asv_read none  --name_file 6           ^& ^
    copy ModelTrain\Data\gts_shekel_train_6.npy ModelTrain\Training_FishEyeNoise_Shekel_Datasets\FishEyeNoise_-_True_False_none\gts_shekel_train.npy                                                                    ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_6.npy ModelTrain\Training_FishEyeNoise_Shekel_Datasets\FishEyeNoise_-_True_False_none\trajectories_shekel_train.npy