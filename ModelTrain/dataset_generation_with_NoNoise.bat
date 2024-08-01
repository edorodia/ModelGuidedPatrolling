mkdir "ModelTrain/NoNoise_Shekel_Datasets"

::NoNoise	-	True	True	miopic
start cmd /k ^
    title Command_1                                                                                                                                                                                             ^& ^
    mkdir "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_True_True_miopic"                                                                                                                                       ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise   --influence_drone_visited_map   --influence_asv_visited_map     --importance_asv_read miopic       --name_file 1        ^& ^
    copy ModelTrain\Data\gts_shekel_train_1.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_True_True_miopic\gts_shekel_train.npy                                                                                                  ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_1.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_True_True_miopic\trajectories_shekel_train.npy

::NoNoise	-	True	True	none
start cmd /k ^
    title Command_2                                                                                                                                                                                             ^& ^
    mkdir "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_True_True_none"                                                                                                                                         ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise   --influence_drone_visited_map   --influence_asv_visited_map     --importance_asv_read none       --name_file 2          ^& ^
    copy ModelTrain\Data\gts_shekel_train_2.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_True_True_none\gts_shekel_train.npy                                                                                                    ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_2.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_True_True_none\trajectories_shekel_train.npy

::NoNoise	-	False	False	miopic
start cmd /k ^
    title Command_3                                                                                                                                                                                             ^& ^
    mkdir "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_False_False_miopic"                                                                                                                                     ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise                                                                   --importance_asv_read miopic    --name_file 3           ^& ^
    copy ModelTrain\Data\gts_shekel_train_3.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_False_False_miopic\gts_shekel_train.npy                                                                                                 ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_3.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_False_False_miopic\trajectories_shekel_train.npy

::NoNoise	-	False	False	none
start cmd /k ^
    title Command_4                                                                                                                                                                                             ^& ^
    mkdir "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_False_False_none"                                                                                                                                       ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise                                                                   --importance_asv_read none --name_file 4                ^& ^
    copy ModelTrain\Data\gts_shekel_train_4.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_False_False_none\gts_shekel_train.npy                                                                                                   ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_4.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_False_False_none\trajectories_shekel_train.npy

::NoNoise	-	True	False	miopic
start cmd /k ^
    title Command_5                                                                                                                                                                                              ^& ^
    mkdir "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_True_False_miopic"                                                                                                                                       ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise   --influence_drone_visited_map                                   --importance_asv_read miopic --name_file 5               ^& ^
    copy ModelTrain\Data\gts_shekel_train_5.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_True_False_miopic\gts_shekel_train.npy                                                                                                   ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_5.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_True_False_miopic\trajectories_shekel_train.npy

::NoNoise	-	True	False	none
start cmd /k ^
    title Command_6                                                                                                                                                                                              ^& ^
    mkdir "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_True_False_none"                                                                                                                                         ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise   --influence_drone_visited_map                                   --importance_asv_read none --name_file 6                 ^& ^
    copy ModelTrain\Data\gts_shekel_train_6.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_True_False_none\gts_shekel_train.npy                                                                                                     ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_6.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_True_False_none\trajectories_shekel_train.npy

::NoNoise	-	False	True	miopic
start cmd /k ^
    title Command_7                                                                                                                                                                                              ^& ^
    mkdir "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_False_True_miopic"                                                                                                                                       ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise                                   --influence_asv_visited_map     --importance_asv_read miopic --name_file 7               ^& ^
    copy ModelTrain\Data\gts_shekel_train_7.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_False_True_miopic\gts_shekel_train.npy                                                                                                   ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_7.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_False_True_miopic\trajectories_shekel_train.npy

::NoNoise	-	False	True	none
start cmd /k ^
    title Command_8                                                                                                                                                                                              ^& ^
    mkdir "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_False_True_none"                                                                                                                                         ^& ^
    python .\ModelTrain\static_dataset_creation-2_in_het.py     --drone_noise NoNoise                                   --influence_asv_visited_map     --importance_asv_read none --name_file 8                 ^& ^
    copy ModelTrain\Data\gts_shekel_train_8.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_False_True_none\gts_shekel_train.npy                                                                                                     ^& ^
    copy ModelTrain\Data\trajectories_shekel_train_8.npy ModelTrain\NoNoise_Shekel_Datasets\NoNoise_-_False_True_none\trajectories_shekel_train.npy
