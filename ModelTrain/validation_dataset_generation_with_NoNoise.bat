mkdir "ModelTrain/Training_NoNoise_Shekel_Datasets"

::NoNoise	-	False	False	none ##############################
start cmd /k ^
    title Command_4                                                                                                                                                                                             ^& ^
    mkdir "ModelTrain/Training_NoNoise_Shekel_Datasets/NoNoise_-_False_False_none"                                                                                                                                       ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise                                                                   --importance_asv_read none --name_file 4                ^& ^
    copy ModelTrain\Data\gts_shekel_train_4.npy ModelTrain\Training_NoNoise_Shekel_Datasets\NoNoise_-_False_False_none\gts_shekel_train.npy                                                                                                   ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_4.npy ModelTrain\Training_NoNoise_Shekel_Datasets\NoNoise_-_False_False_none\trajectories_shekel_train.npy

::NoNoise	-	True	False	none ############################
start cmd /k ^
    title Command_6                                                                                                                                                                                              ^& ^
    mkdir "ModelTrain/Training_NoNoise_Shekel_Datasets/NoNoise_-_True_False_none"                                                                                                                                         ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise   --influence_drone_visited_map                                   --importance_asv_read none --name_file 6                 ^& ^
    copy ModelTrain\Data\gts_shekel_train_6.npy ModelTrain\Training_NoNoise_Shekel_Datasets\NoNoise_-_True_False_none\gts_shekel_train.npy                                                                                                     ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_6.npy ModelTrain\Training_NoNoise_Shekel_Datasets\NoNoise_-_True_False_none\trajectories_shekel_train.npy
